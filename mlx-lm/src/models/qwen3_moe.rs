//! Qwen3-MoE (e.g. Qwen3-30B-A3B): dense Qwen3 attention, sparse MoE MLP.
//!
//! Each decoder layer's MLP is a router (`gate`) over `num_experts` experts,
//! top-`k` of which run as a fused `SwitchGLU`. The attention block is byte-for
//! -byte the dense Qwen3 one, so it is reused verbatim (incl. the L=1 RoPE fix).
//! Experts are AFQ-quantized and evaluated with `gather_qmm`. For prefill (many
//! routed slots) the tokens are sorted by expert and `gather_qmm` is told the
//! indices are sorted (`sorted_indices=true`), so each expert's rows are accessed
//! contiguously — a pure memory-access optimization (byte-identical output),
//! mirroring Python `mlx_lm`'s `SwitchGLU` `_gather_sort`/`_scatter_unsort`. At
//! decode (T=1, few slots) sorting is skipped, exactly as Python does.

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
        argpartition_axis, argsort_axis, expand_dims_axes, gather_qmm, unflatten,
        indexing::{take_along_axis, IndexOp},
        softmax_axis,
    },
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    cache::KeyValueCache,
    error::Error,
    models::qwen3::{
        Attention, AttentionInput, ModelArgs as Qwen3Args, ModelInput, QuantizationConfig,
    },
    utils::{create_attention_mask, AttentionMask},
};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub head_dim: i32,
    pub tie_word_embeddings: bool,
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: i32,
    pub decoder_sparse_step: i32,
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    pub rope_scaling: Option<HashMap<String, crate::utils::rope::FloatOrString>>,
    pub quantization: Option<QuantizationConfig>,
    /// Bit width of the MoE router (`mlp.gate`) when it differs from the rest — some
    /// checkpoints (e.g. Qwen3-Coder) quantize the gate at 8-bit while everything
    /// else is 4-bit. Not in config.json directly; derived from its per-tensor quant
    /// override at load (0 = use the top-level bits). See `load_qwen3_moe_model`.
    #[serde(skip)]
    pub router_bits: i32,
}

impl ModelArgs {
    /// The shared Qwen3 fields, so the dense attention/embedding/norm code is
    /// reused without duplicating its construction.
    fn base(&self) -> Qwen3Args {
        Qwen3Args {
            model_type: self.model_type.clone(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_scaling: self.rope_scaling.clone(),
            quantization: self.quantization.clone(),
        }
    }

    fn is_sparse(&self, layer_idx: i32) -> bool {
        !self.mlp_only_layers.contains(&layer_idx)
            && self.num_experts > 0
            && (layer_idx + 1) % self.decoder_sparse_step == 0
    }
}

/// One expert-batched quantized linear: `weight`/`scales`/`biases` are 3-D
/// `[num_experts, out, in]`. `gather_qmm` selects per-token experts by index.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QSwitchLinear {
    pub group_size: i32,
    pub bits: i32,
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub biases: Param<Array>,
}

impl QSwitchLinear {
    fn new(group_size: i32, bits: i32) -> Self {
        // Placeholders; overwritten by the checkpoint at load time.
        Self {
            group_size,
            bits,
            weight: Param::new(array!(0.0)),
            scales: Param::new(array!(0.0)),
            biases: Param::new(array!(0.0)),
        }
    }

    fn forward(&self, x: &Array, indices: &Array, sorted: bool) -> Result<Array, Exception> {
        gather_qmm(
            x,
            &self.weight.value,
            &self.scales.value,
            &self.biases.value,
            None,
            indices,
            true,
            self.group_size,
            self.bits,
            sorted,
            None, // mode: affine (default)
        )
    }
}

/// Fused gated MLP over experts: `down(silu(gate(x)) * up(x))`, batched by the
/// per-token expert indices.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SwitchGlu {
    #[param]
    pub gate_proj: QSwitchLinear,
    #[param]
    pub up_proj: QSwitchLinear,
    #[param]
    pub down_proj: QSwitchLinear,
}

impl SwitchGlu {
    pub fn new(group_size: i32, bits: i32) -> Self {
        Self {
            gate_proj: QSwitchLinear::new(group_size, bits),
            up_proj: QSwitchLinear::new(group_size, bits),
            down_proj: QSwitchLinear::new(group_size, bits),
        }
    }

    /// `x`: `[B, L, D]`, `indices`: `[B, L, k]` -> `[B, L, k, D]`.
    pub fn forward(&self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        let x = expand_dims_axes(x, &[-2, -3])?; // [B, L, 1, 1, D]

        // Many routed slots (prefill): sort the per-(token,slot) rows by expert id
        // so `gather_qmm` reads each expert's weights contiguously, then scatter the
        // result back. Identical math (a permutation undone by `inv_order`); only the
        // memory-access order changes. Skipped at decode (T=1) where it does not pay.
        if indices.size() >= 64 {
            let sh = indices.shape().to_vec(); // [B, L, k]
            let (b, l, k) = (sh[0], sh[1], sh[2]);
            let idx_flat = indices.flatten(0, -1)?; // [B*L*k]
            let order = argsort_axis(&idx_flat, -1)?; // [B*L*k], ascending by expert
            let inv_order = argsort_axis(&order, -1)?; // inverse permutation
            let token_of_slot = order.floor_divide(Array::from_int(k))?; // [B*L*k]
            let xs = x.flatten(0, -3)?.take_axis(&token_of_slot, 0)?; // [B*L*k, 1, D]
            let idx_sorted = idx_flat.take_axis(&order, 0)?; // [B*L*k], sorted

            let x_up = self.up_proj.forward(&xs, &idx_sorted, true)?;
            let x_gate = self.gate_proj.forward(&xs, &idx_sorted, true)?;
            let act = nn::silu(&x_gate)?.multiply(&x_up)?;
            let out = self.down_proj.forward(&act, &idx_sorted, true)?; // [B*L*k, 1, D]

            let out = out.take_axis(&inv_order, 0)?; // undo the sort
            return unflatten(&out, 0, &[b, l, k])?.squeeze_axes(&[-2]); // [B, L, k, D]
        }

        let x_up = self.up_proj.forward(&x, indices, false)?;
        let x_gate = self.gate_proj.forward(&x, indices, false)?;
        let act = nn::silu(&x_gate)?.multiply(&x_up)?;
        let out = self.down_proj.forward(&act, indices, false)?;
        out.squeeze_axes(&[-2])
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct SparseMoeBlock {
    pub top_k: i32,
    pub num_experts: i32,
    pub norm_topk_prob: bool,

    #[quantizable]
    #[param]
    pub gate: MaybeQuantized<nn::Linear>,
    #[param]
    pub switch_mlp: SwitchGlu,
}

impl SparseMoeBlock {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let gate = nn::LinearBuilder::new(args.hidden_size, args.num_experts)
            .bias(false)
            .build()?;
        let q = args.quantization.as_ref();
        let (group_size, bits) = q.map(|q| (q.group_size, q.bits)).unwrap_or((64, 4));
        // The router may use a different bit width (Qwen3-Coder: 8-bit gate). When it
        // does, PRE-quantize the gate at its own bits so the uniform `nn::quantize`
        // at load leaves it alone; otherwise leave it Original for that pass to handle.
        let gate = if args.router_bits > 0 && args.router_bits != bits {
            use mlx_rs::quantization::Quantizable as _;
            MaybeQuantized::new(gate).quantize_with(|m| m.try_into_quantized(group_size, args.router_bits))?
        } else {
            MaybeQuantized::Original(gate)
        };
        Ok(Self {
            top_k: args.num_experts_per_tok,
            num_experts: args.num_experts,
            norm_topk_prob: args.norm_topk_prob,
            gate,
            switch_mlp: SwitchGlu::new(group_size, bits),
        })
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gates = self.gate.forward(x)?;
        let gates = softmax_axis(&gates, -1, true)?;

        let k = self.top_k;
        // argpartition puts the top-k in the last k columns (unsorted among
        // themselves, which is fine — scores re-weight them).
        let inds = argpartition_axis(&gates, -k, -1)?;
        let inds = inds.index((.., .., (self.num_experts - k)..));
        let scores = take_along_axis(&gates, &inds, -1)?;
        let scores = if self.norm_topk_prob {
            let denom = scores.sum_axes(&[-1], true)?;
            scores.divide(&denom)?
        } else {
            scores
        };

        let y = self.switch_mlp.forward(x, &inds)?;
        let weighted = y.multiply(&expand_dims_axes(&scores, &[-1])?)?;
        weighted.sum_axes(&[-2], false)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: SparseMoeBlock,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    fn new(args: &ModelArgs, base: &Qwen3Args) -> Result<Self, Exception> {
        let self_attn = Attention::new(base)?;
        let mlp = SparseMoeBlock::new(args)?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let self_attn_input = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        };
        let r = self.self_attn.forward(self_attn_input)?;
        let h = x.add(r)?;

        let r = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Qwen3MoeModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3MoeModel {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let base = args.base();
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        // Every layer here is a sparse MoE block. Dense layers (via
        // `mlp_only_layers` / `decoder_sparse_step`) are not yet wired, so fail
        // loud rather than silently building the wrong MLP.
        if (0..args.num_hidden_layers).any(|i| !args.is_sparse(i)) {
            return Err(Exception::custom(
                "qwen3_moe: dense (mlp_only) layers not yet supported",
            ));
        }
        let layers = (0..args.num_hidden_layers)
            .map(|_| DecoderLayer::new(args, &base))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for Qwen3MoeModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only `Array` mask is supported"));
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <DecoderLayer as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: Qwen3MoeModel,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Qwen3MoeModel::new(&args)?;
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

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input)?;
        match self.lm_head.as_mut() {
            Some(lm_head) => lm_head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out),
                MaybeQuantized::Quantized(q) => q.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Qwen3MoeModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_qwen3_moe_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let config_text = std::fs::read_to_string(model_dir.join("config.json"))?;
    let mut args: ModelArgs = serde_json::from_str(&config_text)?;
    // Detect a per-tensor MoE-router (`mlp.gate`) quant override — some checkpoints
    // (Qwen3-Coder) quantize the gate at 8-bit while the rest is 4-bit. Without this
    // the uniform `nn::quantize` below builds a 4-bit gate and the 8-bit checkpoint
    // weights don't fit (shape mismatch). 0 ⇒ use the top-level bits (uniform).
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&config_text) {
        if let Some(q) = v.get("quantization").and_then(|q| q.as_object()) {
            if let Some(gate_bits) = q
                .iter()
                .find(|(k, _)| k.ends_with(".mlp.gate"))
                .and_then(|(_, gv)| gv.get("bits"))
                .and_then(|b| b.as_i64())
            {
                args.router_bits = gate_bits as i32;
            }
        }
    }
    let quantization = args.quantization.clone();
    let mut model = Model::new(args)?;

    // Quantize the MaybeQuantized<Linear>/Embedding leaves (attention q/k/v/o,
    // router gate, embed/lm_head). The hand-built SwitchGLU experts are not
    // Quantizable and are left as raw arrays for `gather_qmm`.
    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    // Target-aware key remap. mlx-rs QuantizedLinear/Embedding nest the packed
    // weight at `<p>.inner.weight`, but checkpoints store `<p>.weight`. Remap
    // `<p>.weight` -> `<p>.inner.weight` ONLY when that inner param exists in the
    // model; the SwitchGLU experts keep `<p>.weight` (no `.inner`), so the same
    // `.scales`-sibling heuristic must not rewrite them. load_safetensors is
    // non-strict, so an unmatched key silently leaves a random init -> garbage.
    fn load_weights_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let mut params = model.parameters_mut().flatten();
        let param_keys: HashSet<String> = params.keys().map(|k| k.to_string()).collect();
        let mut matched = 0usize;
        for (key, value) in loaded {
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
        matched = load_weights_remapped(&mut model, &model_dir.join("model.safetensors"))?;
    }
    if std::env::var("ROZUM_MLX_DEBUG").is_ok() {
        eprintln!("LOADED {matched} params (qwen3_moe)");
    }
    model.eval()?;
    Ok(model)
}

/// Greedy/categorical token iterator, an exact mirror of `qwen3::Generate` for
/// the MoE model: prefill once, then single-token decode. Reuses the proven
/// `qwen3::{GenerateState, sample}` so the prefill/decode shape flow is shared.
pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    sampler: crate::models::qwen3::SamplerOpts,
    history: Vec<u32>,
    state: crate::models::qwen3::GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(
        model: &'a mut Model,
        cache: &'a mut Vec<Option<C>>,
        temp: f32,
        prompt_token: &'a Array,
    ) -> Self {
        Self {
            model,
            cache,
            sampler: crate::models::qwen3::SamplerOpts::with_temp(temp),
            history: Vec::new(),
            state: crate::models::qwen3::GenerateState::Prefill { prompt_token },
        }
    }

    /// Set top-p / top-k / repeat-penalty (temp came from `new`).
    pub fn set_sampler(&mut self, top_p: f32, top_k: i32, repeat_penalty: f32) {
        self.sampler.top_p = top_p;
        self.sampler.top_k = top_k;
        self.sampler.repeat_penalty = repeat_penalty;
    }
}

impl<C> Iterator for Generate<'_, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use crate::models::qwen3::{repeat_window, sample_with, GenerateState};
        use mlx_rs::ops::indexing::{IndexOp, NewAxis};

        macro_rules! tri {
            ($e:expr) => {
                match $e {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e.into())),
                }
            };
        }

        macro_rules! record {
            ($y:expr) => {
                if self.sampler.repeat_penalty != 1.0 {
                    tri!(mlx_rs::transforms::eval([&$y]));
                    self.history
                        .push(tri!($y.reshape(&[-1])).index(0).item::<u32>());
                }
            };
        }

        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                // Chunked prefill: bound the per-chunk forward to `[chunk, ctx]` instead of
                // `[T, T]`; eval the cache between chunks to free activations (MLX lazy-skips
                // `lm_head` on discarded intermediate chunks). Byte-identical to a single
                // forward of the last position. See qwen3::Generate for the rationale.
                let t = prompt_token.shape()[1];
                let chunk = crate::models::qwen3_5::prefill_chunk_size();
                let mut start = 0;
                let logits = loop {
                    let end = (start + chunk).min(t);
                    let piece = prompt_token.index((.., start..end));
                    let l = tri!(self.model.forward(ModelInput {
                        inputs: &piece,
                        mask: None,
                        cache: self.cache,
                    }));
                    if end == t {
                        break l;
                    }
                    let mut to_eval: Vec<&Array> = Vec::new();
                    for c in self.cache.iter().flatten() {
                        c.collect_eval(&mut to_eval);
                    }
                    tri!(mlx_rs::transforms::eval(to_eval));
                    start = end;
                };
                let recent: &[u32] = if self.sampler.repeat_penalty != 1.0 {
                    repeat_window(&self.history)
                } else {
                    &[]
                };
                let y = tri!(sample_with(
                    &logits.index((.., -1, ..)),
                    &self.sampler,
                    recent
                ));
                record!(y);
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let logits = tri!(self.model.forward(ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                }));
                let recent: &[u32] = if self.sampler.repeat_penalty != 1.0 {
                    repeat_window(&self.history)
                } else {
                    &[]
                };
                let y = tri!(sample_with(&logits, &self.sampler, recent));
                record!(y);
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}
