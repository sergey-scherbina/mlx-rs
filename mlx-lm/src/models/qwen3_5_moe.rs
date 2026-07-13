//! Qwen3.6 / Qwen3-Next MoE text model (`qwen3_5_moe`, e.g. Qwen3.6-35B-A3B).
//!
//! Same hybrid backbone as [`super::qwen3_5`] (output-gated full attention every
//! 4th layer + GatedDeltaNet linear layers, both reused verbatim), but every
//! layer's MLP is a sparse MoE block: a router `gate` over `num_experts`, a
//! fused `SwitchGLU` of the top-k experts (reused from [`super::qwen3_moe`]), and
//! a shared expert gated by `sigmoid(shared_expert_gate(x))`. The router gate and
//! shared-expert gate are 8-bit quantized (the rest are 4-bit), so they are held
//! as raw quantized linears (`quantized_matmul`) outside `nn::quantize`.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{
        argpartition_axis, expand_dims_axes, indexing::take_along_axis, indexing::IndexOp,
        indexing::NewAxis, quantized_matmul, softmax_axis,
    },
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    cache::ConcatKeyValueCache,
    error::Error,
    models::{
        qwen3::{Mlp, QuantizationConfig},
        qwen3_5::{Attention, GatedDeltaNet, LayerCache, LinearSnap, ModelArgs as Qwen35Args},
        qwen3_moe::SwitchGlu,
    },
    utils::rope::FloatOrString,
};

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    // Pure-MoE configs omit this (no dense MLP); only the reused dense backbone
    // type carries it, where it is unused for MoE layers.
    #[serde(default)]
    pub intermediate_size: Option<i32>,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<crate::models::qwen3_5::RopeParameters>,
    pub full_attention_interval: i32,
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    pub tie_word_embeddings: bool,
    // MoE
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: i32,
    pub shared_expert_intermediate_size: i32,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    pub quantization: Option<QuantizationConfig>,
}

impl ModelArgs {
    fn is_linear(&self, layer_idx: i32) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }

    /// The shared Qwen3.6 backbone fields, so the reused attention / GatedDeltaNet
    /// constructors take a plain `qwen3_5::ModelArgs`.
    fn base(&self) -> Qwen35Args {
        Qwen35Args {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size.unwrap_or(0),
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            partial_rotary_factor: self.partial_rotary_factor,
            rope_parameters: self.rope_parameters.clone(),
            full_attention_interval: self.full_attention_interval,
            linear_num_value_heads: self.linear_num_value_heads,
            linear_num_key_heads: self.linear_num_key_heads,
            linear_key_head_dim: self.linear_key_head_dim,
            linear_value_head_dim: self.linear_value_head_dim,
            linear_conv_kernel_dim: self.linear_conv_kernel_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_scaling: self.rope_scaling.clone(),
            quantization: self.quantization.clone(),
        }
    }
}

/// A raw quantized linear (`quantized_matmul`) for the 8-bit router / shared
/// gates, which `nn::quantize` (uniform 4-bit) must not touch.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QuantLinear {
    pub group_size: i32,
    pub bits: i32,
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
}

impl QuantLinear {
    fn new(group_size: i32, bits: i32) -> Self {
        Self {
            group_size,
            bits,
            weight: Param::new(array!(0.0)),
            scales: Param::new(array!(0.0)),
            biases: Param::new(array!(0.0)),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        quantized_matmul(
            x,
            &self.weight.value,
            &self.scales.value,
            &self.biases.value,
            true,
            self.group_size,
            self.bits,
        )
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct SparseMoeBlock {
    pub top_k: i32,
    pub num_experts: i32,
    pub norm_topk_prob: bool,

    #[param]
    pub gate: QuantLinear,
    #[param]
    pub switch_mlp: SwitchGlu,
    #[quantizable]
    #[param]
    pub shared_expert: Mlp,
    #[param]
    pub shared_expert_gate: QuantLinear,
}

impl SparseMoeBlock {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let gs = args.quantization.as_ref().map(|q| q.group_size).unwrap_or(64);
        // The MoE experts (switch_mlp + shared_expert) are quantized at EXPERT_BITS, which can DIFFER
        // from the model's global `bits`: a DWQ / mixed-precision checkpoint keeps attention + embed +
        // lm_head at 8-bit but the experts at 4-bit (better quality at the same size). The old code fed
        // switch_mlp the global `bits` (8 for DWQ), so its `gather_qmm` saw a 4-bit-checkpoint vs
        // 8-bit-structure shape mismatch and the model couldn't run at all. gate + shared_expert_gate
        // stay 8-bit (the existing per-leaf hardcodes). The standard uniform-4-bit 35B ALSO has 4-bit
        // experts, so 4 is correct for both — and the model-level uniform `nn::quantize` still applies
        // the global `bits` to attention/embed/lm_head.
        const EXPERT_BITS: i32 = 4;
        Ok(Self {
            top_k: args.num_experts_per_tok,
            num_experts: args.num_experts,
            norm_topk_prob: args.norm_topk_prob,
            gate: QuantLinear::new(gs, 8),
            switch_mlp: SwitchGlu::new(gs, EXPERT_BITS),
            // Pre-quantize the shared expert at EXPERT_BITS. Its leaves are `MaybeQuantized`, so the
            // model-level uniform `nn::quantize` (run at the global `bits`) idempotently SKIPS them —
            // they stay at EXPERT_BITS instead of being forced to the global width.
            shared_expert: mlx_rs::nn::quantize(
                Mlp::new(args.hidden_size, args.shared_expert_intermediate_size)?,
                gs,
                EXPERT_BITS,
            )?,
            shared_expert_gate: QuantLinear::new(gs, 8),
        })
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gates = softmax_axis(&self.gate.forward(x)?, -1, true)?;
        let k = self.top_k;
        let inds = argpartition_axis(&gates, -k, -1)?;
        let inds = inds.index((.., .., (self.num_experts - k)..));
        let scores = take_along_axis(&gates, &inds, -1)?;
        let scores = if self.norm_topk_prob {
            scores.divide(&scores.sum_axes(&[-1], true)?)?
        } else {
            scores
        };

        let y = self.switch_mlp.forward(x, &inds)?;
        let y = y
            .multiply(&expand_dims_axes(&scores, &[-1])?)?
            .sum_axes(&[-2], false)?;

        let shared = self.shared_expert.forward(x)?;
        let shared = nn::sigmoid(&self.shared_expert_gate.forward(x)?)?.multiply(&shared)?;
        y.add(&shared)
    }

    fn training_mode(&mut self, mode: bool) {
        self.shared_expert.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    pub is_linear: bool,
    #[quantizable]
    #[param]
    pub self_attn: Option<Attention>,
    #[quantizable]
    #[param]
    pub linear_attn: Option<GatedDeltaNet>,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[quantizable]
    #[param]
    pub mlp: SparseMoeBlock,
}

impl DecoderLayer {
    fn new(args: &ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let base = args.base();
        let is_linear = args.is_linear(layer_idx);
        let (self_attn, linear_attn) = if is_linear {
            (None, Some(GatedDeltaNet::new(&base)?))
        } else {
            (Some(Attention::new(&base)?), None)
        };
        Ok(Self {
            is_linear,
            self_attn,
            linear_attn,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            mlp: SparseMoeBlock::new(args)?,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        causal: bool,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let r = match (self.is_linear, cache) {
            (false, LayerCache::Full(kv)) => {
                self.self_attn
                    .as_mut()
                    .unwrap()
                    .forward(&normed, causal, Some(kv))?
            }
            (true, LayerCache::Linear { conv, state }) => self
                .linear_attn
                .as_mut()
                .unwrap()
                .forward(&normed, conv, state)?,
            _ => return Err(Exception::custom("qwen3_5_moe: cache/layer kind mismatch")),
        };
        let h = x.add(&r)?;
        let m = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(&m)
    }

    fn training_mode(&mut self, mode: bool) {
        // Inference-only; the reused attention / GDN training hooks are private.
        self.mlp.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Qwen3_5MoeModel {
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3_5MoeModel {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }

    fn init_cache(&self) -> Vec<LayerCache> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_linear {
                    LayerCache::Linear {
                        conv: None,
                        state: None,
                    }
                } else {
                    LayerCache::Full(ConcatKeyValueCache::new())
                }
            })
            .collect()
    }

    fn forward(&mut self, inputs: &Array, cache: &mut [LayerCache]) -> Result<Array, Exception> {
        let mut h = self.embed_tokens.forward(inputs)?;
        // Multimodal (Qwen3.5-VL MoE, e.g. Qwen3.6-35B): splice the vision-tower output
        // onto the image-token block. Shared with the dense model via `apply_mm_splice`.
        h = crate::models::qwen3_5::apply_mm_splice(&h)?;
        let t = h.shape()[1];
        // Prefill uses fused causal SDPA; decode (T==1) needs no mask. See qwen3_5.
        let causal = t > 1;
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(&h, causal, c)?;
        }
        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for l in &mut self.layers {
            l.training_mode(mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: Qwen3_5MoeModel,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Qwen3_5MoeModel::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        inputs: &Array,
        cache: &mut [LayerCache],
    ) -> Result<Array, Exception> {
        let out = self.model.forward(inputs, cache)?;
        self.project(&out)
    }

    /// Project hidden states to vocab logits (`lm_head`, or the tied embedding).
    fn project(&mut self, hidden: &Array) -> Result<Array, Exception> {
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(hidden),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(hidden),
                MaybeQuantized::Quantized(q) => q.as_linear(hidden),
            },
        }
    }

    /// Chunked prompt prefill (returns last-position logits). See
    /// [`super::qwen3_5::Model::prefill`] — same mechanism, MoE backbone;
    /// `lm_head` runs only on the final position.
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut [LayerCache],
    ) -> Result<Array, Exception> {
        self.prefill_chunked(inputs, cache, crate::models::qwen3_5::prefill_chunk_size())
    }

    /// [`prefill`](Self::prefill) with an explicit chunk size (for tests).
    pub fn prefill_chunked(
        &mut self,
        inputs: &Array,
        cache: &mut [LayerCache],
        chunk: i32,
    ) -> Result<Array, Exception> {
        self.prefill_cancellable(inputs, cache, chunk, &|| false)
            .map(|o| o.expect("prefill not cancelled"))
    }

    /// [`prefill_chunked`](Self::prefill_chunked) that polls `should_cancel`
    /// between chunks; returns `Ok(None)` if it fired. See `qwen3_5`.
    pub fn prefill_cancellable(
        &mut self,
        inputs: &Array,
        cache: &mut [LayerCache],
        chunk: i32,
        should_cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Array>, Exception> {
        let t = inputs.shape()[1];
        let mut start = 0;
        let last_hidden = loop {
            if should_cancel() {
                return Ok(None);
            }
            let end = (start + chunk).min(t);
            let piece = inputs.index((.., start..end));
            let hidden = self.model.forward(&piece, cache)?;
            if end == t {
                let l = end - start;
                break hidden.index((.., (l - 1)..l, ..));
            }
            let mut to_eval: Vec<&Array> = Vec::new();
            for c in cache.iter() {
                c.collect_eval(&mut to_eval);
            }
            mlx_rs::transforms::eval(to_eval)?;
            start = end;
        };
        Ok(Some(self.project(&last_hidden)?))
    }

    pub fn init_cache(&self) -> Vec<LayerCache> {
        self.model.init_cache()
    }

    pub fn training_mode(&mut self, mode: bool) {
        self.model.training_mode(mode);
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct WrappedConfig {
    text_config: ModelArgs,
    quantization: Option<QuantizationConfig>,
}

pub fn load_qwen3_5_moe_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let text = std::fs::read_to_string(model_dir.join("config.json"))?;
    let (mut args, quantization): (ModelArgs, Option<QuantizationConfig>) =
        match serde_json::from_str::<WrappedConfig>(&text) {
            Ok(w) => {
                let q = w.text_config.quantization.clone().or(w.quantization);
                (w.text_config, q)
            }
            Err(_) => {
                let a: ModelArgs = serde_json::from_str(&text)?;
                let q = a.quantization.clone();
                (a, q)
            }
        };
    args.quantization = quantization.clone();
    let mut model = Model::new(args)?;

    // Uniform 4-bit for the MaybeQuantized leaves (attention, shared_expert,
    // embed, lm_head, GatedDeltaNet projections). The raw 8-bit gates and the
    // 4-bit SwitchGLU experts are not Quantizable and load directly.
    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    fn load_weights_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let mut params = model.parameters_mut().flatten();
        let param_keys: HashSet<String> = params.keys().map(|k| k.to_string()).collect();
        let mut matched = 0usize;
        for (key, mut value) in loaded {
            let key = if let Some(k) = key.strip_prefix("language_model.") {
                k.to_string()
            } else if key.starts_with("vision_tower.") || key.starts_with("visual.") {
                continue;
            } else {
                key
            };
            if key.ends_with("conv1d.weight") && *value.shape().last().unwrap() != 1 {
                value = value.move_axis(2, 1)?;
            }
            let mapped = match key.strip_suffix(".weight") {
                Some(prefix) if param_keys.contains(&format!("{prefix}.inner.weight")) => {
                    format!("{prefix}.inner.weight")
                }
                _ => key,
            };
            if let Some(param) = params.get_mut(mapped.as_str()) {
                **param = value;
                matched += 1;
            }
        }
        Ok(matched)
    }

    let weights_index = model_dir.join("model.safetensors.index.json");
    let mut matched = 0usize;
    if weights_index.exists() {
        let json = std::fs::read_to_string(weights_index)?;
        let weight_map: WeightMap = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = weight_map.weight_map.values().collect();
        for weight_file in weight_files {
            matched += load_weights_remapped(&mut model, &model_dir.join(weight_file))?;
        }
    } else {
        let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            shards.push(model_dir.join("model.safetensors"));
        }
        for shard in shards {
            matched += load_weights_remapped(&mut model, &shard)?;
        }
    }
    if std::env::var("ROZUM_MLX_DEBUG").is_ok() {
        eprintln!("LOADED {matched} params (qwen3_5_moe)");
    }
    model.eval()?;
    Ok(model)
}

/// Greedy/categorical token iterator (prefill once, then single-token decode).
pub struct Generate<'a> {
    model: &'a mut Model,
    cache: Vec<LayerCache>,
    sampler: crate::models::qwen3::SamplerOpts,
    /// Generated-token history for the repetition penalty (see qwen3_5).
    history: Vec<u32>,
    state: GenState<'a>,
    /// Polled between prefill chunks for mid-prefill cancellation (see qwen3_5).
    should_cancel: Box<dyn Fn() -> bool + Send>,
    /// Linear-state snapshot at the conversation boundary, for prefix reuse (qwen3_5).
    prefill_snapshot: Option<Vec<LinearSnap>>,
    /// Trailing generation-prompt length; the snapshot is taken before it (qwen3_5).
    gen_prompt_len: i32,
    /// Multimodal (Qwen3.5-VL MoE) context — see qwen3_5::MmContext. Present for VLM
    /// checkpoints of the MoE arch (e.g. Qwen3.6-35B-A3B). Single-pass mm prefill
    /// (splice + prompt M-RoPE, no chunk/reuse) + per-decode-step scalar-position M-RoPE.
    mm: Option<crate::models::qwen3_5::MmContext>,
}

enum GenState<'a> {
    Prefill(&'a Array),
    Decode(Array),
}

impl<'a> Generate<'a> {
    pub fn new(model: &'a mut Model, temp: f32, prompt_token: &'a Array) -> Self {
        let cache = model.init_cache();
        Self::with_cache(model, temp, prompt_token, cache)
    }

    /// Start generation from a pre-populated cache (prefix reuse): prefill only
    /// `prompt_token` (the new suffix) on top of `cache`. See qwen3_5.
    pub fn with_cache(
        model: &'a mut Model,
        temp: f32,
        prompt_token: &'a Array,
        cache: Vec<LayerCache>,
    ) -> Self {
        Self {
            model,
            cache,
            sampler: crate::models::qwen3::SamplerOpts::with_temp(temp),
            history: Vec::new(),
            state: GenState::Prefill(prompt_token),
            should_cancel: Box::new(|| false),
            prefill_snapshot: None,
            gen_prompt_len: 0,
            mm: None,
        }
    }

    /// Set the trailing generation-prompt length (snapshot at the conv boundary).
    pub fn set_gen_prompt_len(&mut self, n: i32) {
        self.gen_prompt_len = n.max(0);
    }

    /// Attach multimodal context (vision splice + M-RoPE). Single-pass prefill, no
    /// prefix reuse. Mirror of qwen3_5::Generate::set_mm_context.
    pub fn set_mm_context(&mut self, mm: crate::models::qwen3_5::MmContext) {
        self.mm = Some(mm);
    }

    /// Consume the iterator, returning the advanced cache + the conversation-boundary
    /// Linear snapshot (`None` if prefill never completed). See qwen3_5.
    pub fn into_cache_and_snapshot(self) -> (Vec<LayerCache>, Option<Vec<LinearSnap>>) {
        (self.cache, self.prefill_snapshot)
    }

    /// Install a cancellation predicate, polled between prefill chunks.
    pub fn set_cancel(&mut self, should_cancel: Box<dyn Fn() -> bool + Send>) {
        self.should_cancel = should_cancel;
    }

    /// Set top-p / top-k / repeat-penalty (see qwen3_5).
    pub fn set_sampler(&mut self, top_p: f32, top_k: i32, repeat_penalty: f32) {
        self.sampler.top_p = top_p;
        self.sampler.top_k = top_k;
        self.sampler.repeat_penalty = repeat_penalty;
    }
}

impl Iterator for Generate<'_> {
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        macro_rules! tri {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e.into())),
                }
            };
        }
        let (inputs, is_prefill) = match &self.state {
            GenState::Prefill(p) => ((*p).clone(), true),
            GenState::Decode(y) => (y.index((.., NewAxis)), false),
        };
        let logits = if is_prefill {
            if let Some((splice, pcos, psin)) = self
                .mm
                .as_ref()
                .map(|mm| (mm.splice.clone(), mm.prompt_cos.clone(), mm.prompt_sin.clone()))
            {
                // Multimodal: single-pass prefill with vision splice(s) + prompt M-RoPE
                // (no chunking / prefix reuse). Mirror of qwen3_5::Generate. Project only
                // the LAST position — the image-padded positions never feed the vocab head.
                crate::models::qwen3_5::set_mm_splice(Some(splice));
                crate::models::qwen3_5::set_mrope_cossin(Some((pcos, psin)));
                let l = match self.model.model.forward(&inputs, &mut self.cache) {
                    Ok(h) => {
                        let last = h.index((.., -1.., ..));
                        self.model.project(&last)
                    }
                    Err(e) => Err(e),
                };
                crate::models::qwen3_5::set_mm_splice(None);
                crate::models::qwen3_5::set_mrope_cossin(None);
                self.prefill_snapshot = Some(self.cache.iter().map(|c| c.snapshot()).collect());
                tri!(l)
            } else {
            // Snapshot the Linear state at the conversation boundary (prompt len −
            // gen_prompt_len), not the very end — see qwen3_5 for the rationale.
            let chunk = crate::models::qwen3_5::prefill_chunk_size();
            let t = inputs.shape()[1];
            let split = t - self.gen_prompt_len;
            if self.gen_prompt_len > 0 && split >= 0 && split < t {
                if split > 0 {
                    let conv = inputs.index((.., 0..split));
                    if tri!(self.model.prefill_cancellable(
                        &conv,
                        &mut self.cache,
                        chunk,
                        &*self.should_cancel,
                    ))
                    .is_none()
                    {
                        return None;
                    }
                }
                self.prefill_snapshot =
                    Some(self.cache.iter().map(|c| c.snapshot()).collect());
                let tail = inputs.index((.., split..t));
                tri!(self.model.forward(&tail, &mut self.cache))
            } else {
                let l = match tri!(self.model.prefill_cancellable(
                    &inputs,
                    &mut self.cache,
                    chunk,
                    &*self.should_cancel,
                )) {
                    Some(l) => l,
                    None => return None,
                };
                self.prefill_snapshot =
                    Some(self.cache.iter().map(|c| c.snapshot()).collect());
                l
            }
            }
        } else if let Some((pos, rd, theta)) = self.mm.as_mut().map(|mm| mm.next_decode_step()) {
            // Multimodal decode: install this step's scalar-position M-RoPE.
            let (dc, ds) =
                crate::models::qwen3_5_vision::mrope_cos_sin(&[(pos, pos, pos)], rd, theta);
            crate::models::qwen3_5::set_mrope_cossin(Some((
                Array::from_slice(&dc, &[1, rd]),
                Array::from_slice(&ds, &[1, rd]),
            )));
            let l = self.model.forward(&inputs, &mut self.cache);
            crate::models::qwen3_5::set_mrope_cossin(None);
            tri!(l)
        } else {
            tri!(self.model.forward(&inputs, &mut self.cache))
        };
        let recent: &[u32] = if self.sampler.repeat_penalty != 1.0 {
            crate::models::qwen3::repeat_window(&self.history)
        } else {
            &[]
        };
        let y = tri!(crate::models::qwen3::sample_with(
            &logits.index((.., -1, ..)),
            &self.sampler,
            recent,
        ));
        if self.sampler.repeat_penalty != 1.0 {
            tri!(mlx_rs::transforms::eval([&y]));
            self.history.push(y.index(0).item::<u32>());
        }
        self.state = GenState::Decode(y.clone());
        Some(Ok(y))
    }
}
