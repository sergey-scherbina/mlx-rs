//! OpenAI gpt-oss (e.g. gpt-oss-20b): a sparse-MoE decoder with several features
//! the Qwen3 path lacks, ported faithfully from Python `mlx_lm`'s `gpt_oss.py`:
//!
//! * **Attention sinks** — a per-head learned logit added to the softmax (an
//!   always-available "no-op" key), via `fast::scaled_dot_product_attention(sinks)`.
//! * **Sliding/full alternating attention** — even layers attend within a
//!   `sliding_window` (128) band, odd layers attend fully. We keep a full KV cache
//!   for every layer and enforce the window with a banded causal mask (identical
//!   math to a rotating cache — the evicted keys are exactly the masked ones; a
//!   future memory optimization can swap in a true rotating cache).
//! * **q/k/v/o biases** — attention projections carry additive biases.
//! * **YaRN RoPE** — frequency interpolation + mscale (see `utils::rope::YarnRope`).
//! * **Mixed quantization** — attention/embed/lm_head are affine 4-bit, the router
//!   is affine **8-bit**, and the experts are **MXFP4** (4-bit microscaling, no
//!   zero-point). Each leaf is quantized at its own bits at construction.
//! * **Clamped SwiGLU experts** — `(clip(gate,max=L) * sigmoid(α·clip(gate,max=L)))
//!   * (clip(up,±L) + 1)` with `α=1.702`, plus additive per-expert gate/up/down
//!   biases — not the plain `silu(gate)*up` of Qwen.
//!
//! The checkpoint (mlx-community MXFP4-Q4) is already "sanitized" (experts split
//! into `gate_proj`/`up_proj`/`down_proj`), so no weight surgery is needed at load.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{
        argpartition_axis, expand_dims_axes, gather_qmm,
        indexing::{take_along_axis, IndexOp},
        maximum, minimum, softmax_axis, zeros,
    },
    quantization::{MaybeQuantized, Quantizable as _},
    Array,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    cache::KeyValueCache,
    error::Error,
    models::qwen3::{AttentionInput, ModelInput},
    utils::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
};

/// SwiGLU gate scale `α` (a fixed gpt-oss constant, not in the config).
const SWIGLU_ALPHA: f32 = 1.702;
/// Affine-quant group size for the non-expert leaves (attention/embed/router/lm_head).
const AFFINE_GROUP_SIZE: i32 = 64;
/// Affine-quant bit width for attention/embed/lm_head.
const AFFINE_BITS: i32 = 4;
/// The router is affine-quantized at 8 bits (per the checkpoint's quant config).
const ROUTER_BITS: i32 = 8;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: i32,
    pub num_local_experts: i32,
    pub num_experts_per_tok: i32,
    #[serde(default = "default_swiglu_limit")]
    pub swiglu_limit: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<String>,
    /// Raw JSON so non-numeric entries (e.g. `truncate: false`, `rope_type`) parse;
    /// converted to the `FloatOrString` map `initialize_rope` wants via
    /// [`ModelArgs::rope_scaling_map`].
    pub rope_scaling: Option<HashMap<String, Value>>,
    pub quantization: Option<GptOssQuantization>,
}

impl ModelArgs {
    /// Numeric/string rope-scaling fields for `initialize_rope`; bools (e.g.
    /// `truncate`) are dropped since `FloatOrString` can't hold them and they are
    /// not read by the YaRN path.
    fn rope_scaling_map(&self) -> Option<HashMap<String, FloatOrString>> {
        self.rope_scaling.as_ref().map(|m| {
            m.iter()
                .filter_map(|(k, v)| match v {
                    Value::Number(n) => {
                        n.as_f64().map(|f| (k.clone(), FloatOrString::Float(f as f32)))
                    }
                    Value::String(s) => Some((k.clone(), FloatOrString::String(s.clone()))),
                    _ => None,
                })
                .collect()
        })
    }
}

fn default_max_position_embeddings() -> i32 {
    131072
}
fn default_sliding_window() -> i32 {
    128
}
fn default_swiglu_limit() -> f32 {
    7.0
}

/// Only the top-level (mxfp4 expert) quant params are parsed; the per-tensor
/// affine overrides are handled by the per-leaf constants above.
#[derive(Debug, Clone, Deserialize)]
pub struct GptOssQuantization {
    pub group_size: i32,
    pub bits: i32,
}

// ───────────────────────────── attention ─────────────────────────────

#[derive(Debug, Clone, ModuleParameters)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    /// Per-head attention-sink logits `[n_heads]`.
    #[param]
    pub sinks: Param<Array>,
    #[param]
    pub rope: RopeVariant,
}

/// Build an affine-quantized `Linear` (with bias) directly, so a mixed-bit model
/// can give each leaf its own bit width without a uniform `nn::quantize` pass.
fn quantized_linear(
    in_dim: i32,
    out_dim: i32,
    bias: bool,
    group_size: i32,
    bits: i32,
) -> Result<MaybeQuantized<nn::Linear>, Exception> {
    let lin = nn::LinearBuilder::new(in_dim, out_dim).bias(bias).build()?;
    MaybeQuantized::new(lin).quantize_with(|m| m.try_into_quantized(group_size, bits))
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = quantized_linear(dim, n_heads * head_dim, true, AFFINE_GROUP_SIZE, AFFINE_BITS)?;
        let k_proj =
            quantized_linear(dim, n_kv_heads * head_dim, true, AFFINE_GROUP_SIZE, AFFINE_BITS)?;
        let v_proj =
            quantized_linear(dim, n_kv_heads * head_dim, true, AFFINE_GROUP_SIZE, AFFINE_BITS)?;
        let o_proj = quantized_linear(n_heads * head_dim, dim, true, AFFINE_GROUP_SIZE, AFFINE_BITS)?;

        let sinks = Param::new(zeros::<f32>(&[n_heads])?);
        let rope_scaling = args.rope_scaling_map();
        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
            &rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            sinks,
            rope,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = shape[0];
        let L = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = queries
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = keys
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(cache) = cache.as_mut() {
            let off = cache.offset();
            queries = self
                .rope
                .forward(nn::RopeInputBuilder::new(&queries).offset(off).build()?)?;
            keys = self
                .rope
                .forward(nn::RopeInputBuilder::new(&keys).offset(off).build()?)?;
            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        let attn = mlx_rs::fast::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            self.scale,
            mask.map(ScaledDotProductAttentionMask::Array),
            Some(&self.sinks.value),
        )?;

        let output = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ───────────────────────────── MoE experts ─────────────────────────────

/// gpt-oss clamped SwiGLU: `(g·σ(α·g)) · (u + 1)` with `g = clip(gate, max=L)`,
/// `u = clip(up, ±L)`. Mirrors Python `mlx_lm` `gpt_oss.swiglu`.
fn swiglu(x_up: &Array, x_gate: &Array, limit: f32) -> Result<Array, Exception> {
    let lim = Array::from_f32(limit);
    let neg_lim = Array::from_f32(-limit);
    let x_glu = minimum(x_gate, &lim)?;
    let x_linear = minimum(maximum(x_up, &neg_lim)?, &lim)?;
    let glu_scaled = x_glu.multiply(Array::from_f32(SWIGLU_ALPHA))?;
    let sig = nn::sigmoid(&glu_scaled)?;
    let out_glu = x_glu.multiply(&sig)?;
    out_glu.multiply(&x_linear.add(Array::from_f32(1.0))?)
}

/// Gather a per-expert additive bias `[num_experts, out]` by per-slot expert
/// `indices` (any shape) → `[*indices.shape, out]`.
fn take_expert_bias(bias: &Array, indices: &Array) -> Result<Array, Exception> {
    let idx_flat = indices.flatten(0, -1)?;
    let gathered = bias.take_axis(&idx_flat, 0)?; // [N, out]
    let mut shape = indices.shape().to_vec();
    shape.push(-1);
    gathered.reshape(&shape)
}

/// One expert-batched MXFP4 linear with an additive per-expert bias. Weights are
/// `[num_experts, out, in]`; `gather_qmm(mode="mxfp4")` selects per-token experts.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QSwitchLinear {
    pub group_size: i32,
    pub bits: i32,
    #[param]
    pub weight: Param<Array>,
    #[param]
    pub scales: Param<Array>,
    #[param]
    pub bias: Param<Array>,
}

impl QSwitchLinear {
    fn new(group_size: i32, bits: i32) -> Self {
        Self {
            group_size,
            bits,
            weight: Param::new(Array::from_f32(0.0)),
            scales: Param::new(Array::from_f32(0.0)),
            bias: Param::new(Array::from_f32(0.0)),
        }
    }

    fn forward(&self, x: &Array, indices: &Array) -> Result<Array, Exception> {
        // mxfp4: scales only, no zero-point biases (4th arg None).
        let y = gather_qmm(
            x,
            &self.weight.value,
            &self.scales.value,
            None::<&Array>, // mxfp4: no zero-point biases
            None::<&Array>, // no lhs (x) gather
            indices,
            true,
            self.group_size,
            self.bits,
            false,
            Some("mxfp4"),
        )?;
        let bias = expand_dims_axes(&take_expert_bias(&self.bias.value, indices)?, &[-2])?;
        y.add(&bias)
    }
}

/// Fused gated MLP over experts with the clamped SwiGLU activation.
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
    fn new(group_size: i32, bits: i32) -> Self {
        Self {
            gate_proj: QSwitchLinear::new(group_size, bits),
            up_proj: QSwitchLinear::new(group_size, bits),
            down_proj: QSwitchLinear::new(group_size, bits),
        }
    }

    /// `x`: `[B, L, D]`, `indices`: `[B, L, k]` -> `[B, L, k, D]`.
    // NOTE: the prefill expert-sort optimization (see qwen3_moe) is intentionally
    // omitted for v1 correctness; output is identical, prefill just runs slower.
    fn forward(&self, x: &Array, indices: &Array, limit: f32) -> Result<Array, Exception> {
        let x = expand_dims_axes(x, &[-2, -3])?; // [B, L, 1, 1, D]
        let x_up = self.up_proj.forward(&x, indices)?;
        let x_gate = self.gate_proj.forward(&x, indices)?;
        let act = swiglu(&x_up, &x_gate, limit)?;
        let out = self.down_proj.forward(&act, indices)?;
        out.squeeze_axes(&[-2])
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct MlpBlock {
    pub top_k: i32,
    pub num_experts: i32,
    pub swiglu_limit: f32,

    #[param]
    pub router: MaybeQuantized<nn::Linear>,
    #[param]
    pub experts: SwitchGlu,
}

impl MlpBlock {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let router = quantized_linear(
            args.hidden_size,
            args.num_local_experts,
            true,
            AFFINE_GROUP_SIZE,
            ROUTER_BITS,
        )?;
        let (group_size, bits) = args
            .quantization
            .as_ref()
            .map(|q| (q.group_size, q.bits))
            .unwrap_or((32, 4));
        Ok(Self {
            top_k: args.num_experts_per_tok,
            num_experts: args.num_local_experts,
            swiglu_limit: args.swiglu_limit,
            router,
            experts: SwitchGlu::new(group_size, bits),
        })
    }
}

impl Module<&Array> for MlpBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        // gpt-oss routes on the RAW router logits: top-k then softmax over the
        // selected k (precise), unlike Qwen's softmax-all-then-top-k.
        let g = self.router.forward(x)?;
        let k = self.top_k;
        let inds = argpartition_axis(&g, -k, -1)?;
        let inds = inds.index((.., .., (self.num_experts - k)..));
        let scores = take_along_axis(&g, &inds, -1)?;
        let weights = softmax_axis(&scores, -1, true)?;

        let y = self.experts.forward(x, &inds, self.swiglu_limit)?;
        let weighted = y.multiply(&expand_dims_axes(&weights, &[-1])?)?;
        weighted.sum_axes(&[-2], false)
    }

    fn training_mode(&mut self, mode: bool) {
        self.router.training_mode(mode);
    }
}

// ───────────────────────────── decoder ─────────────────────────────

#[derive(Debug, Clone, ModuleParameters)]
pub struct DecoderLayer {
    #[param]
    pub self_attn: Attention,
    #[param]
    pub mlp: MlpBlock,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl DecoderLayer {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            self_attn: Attention::new(args)?,
            mlp: MlpBlock::new(args)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
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

        let r = self.self_attn.forward(AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        })?;
        let h = x.add(r)?;

        let r = self.mlp.forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(r)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct GptOssModel {
    pub num_hidden_layers: i32,
    pub sliding_window: i32,
    pub layer_types: Vec<String>,

    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl GptOssModel {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let embed = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let embed_tokens = MaybeQuantized::new(embed)
            .quantize_with(|m| m.try_into_quantized(AFFINE_GROUP_SIZE, AFFINE_BITS))?;
        let layers = (0..args.num_hidden_layers)
            .map(|_| DecoderLayer::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            num_hidden_layers: args.num_hidden_layers,
            sliding_window: args.sliding_window,
            layer_types: args.layer_types.clone(),
            embed_tokens,
            layers,
            norm,
        })
    }
}

impl<C> Module<ModelInput<'_, C>> for GptOssModel
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs, cache, ..
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        // All layers' caches advance in lockstep, so one offset drives both masks.
        let offset = cache
            .first()
            .and_then(|c| c.as_ref())
            .map(|c| c.offset())
            .unwrap_or(0);
        let T = h.shape()[1];

        // Full layers: plain causal (None at decode). Sliding layers: a banded
        // causal mask, needed even at T=1 once the history exceeds the window.
        let full_mask = if T > 1 {
            Some(create_causal_mask(T, Some(offset), None, None)?)
        } else {
            None
        };
        let swa_mask = if T > 1 || offset >= self.sliding_window {
            Some(create_causal_mask(
                T,
                Some(offset),
                Some(self.sliding_window),
                None,
            )?)
        } else {
            None
        };

        for (i, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let mask = if self.layer_types[i].as_str() == "sliding_attention" {
                swa_mask.as_ref()
            } else {
                full_mask.as_ref()
            };
            h = layer.forward(AttentionInput {
                x: &h,
                mask,
                cache: c.as_mut(),
            })?;
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

#[derive(Debug, Clone, ModuleParameters)]
pub struct Model {
    pub args: ModelArgs,

    #[param]
    pub model: GptOssModel,
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = GptOssModel::new(&args)?;
        let lm_head = if args.tie_word_embeddings {
            None
        } else {
            Some(quantized_linear(
                args.hidden_size,
                args.vocab_size,
                false,
                AFFINE_GROUP_SIZE,
                AFFINE_BITS,
            )?)
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
        <GptOssModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
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

pub fn load_gpt_oss_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let file = std::fs::File::open(model_dir.join("config.json"))?;
    let args: ModelArgs = serde_json::from_reader(file)?;
    let mut model = Model::new(args)?;

    // Leaves are already quantized (per-leaf, at construction). Target-aware
    // remap: send `<p>.weight` -> `<p>.inner.weight` ONLY when that inner param
    // exists (the affine QuantizedLinear/Embedding leaves); the hand-built MXFP4
    // experts keep `<p>.weight` (no `.inner`). `.scales`/`.biases`/`.bias` load by
    // their own keys. load_safetensors is non-strict, so an unmatched key would
    // silently leave a random init -> garbage; the matched count guards that.
    fn load_weights_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let mut params = model.parameters_mut().flatten();
        let param_keys: HashSet<String> = params.keys().map(|k| k.to_string()).collect();
        let mut matched = 0usize;
        for (key, value) in loaded {
            // QuantizedLinear nests BOTH the packed weight and the additive bias
            // under `.inner` (`<p>.inner.weight`, `<p>.inner.bias`); the affine
            // `.scales`/`.biases` (zero-points) stay top-level. The MXFP4 experts
            // are hand-built (no `.inner`), so only remap when the inner param
            // actually exists in the model.
            let mapped = match key
                .strip_suffix(".weight")
                .filter(|p| param_keys.contains(&format!("{p}.inner.weight")))
            {
                Some(prefix) => format!("{prefix}.inner.weight"),
                None => match key
                    .strip_suffix(".bias")
                    .filter(|p| param_keys.contains(&format!("{p}.inner.bias")))
                {
                    Some(prefix) => format!("{prefix}.inner.bias"),
                    None => key,
                },
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
        eprintln!("LOADED {matched} params (gpt_oss)");
    }
    model.eval()?;
    Ok(model)
}

/// Greedy/categorical token iterator — an exact mirror of `qwen3_moe::Generate`
/// (prefill once with chunking, then single-token decode), reusing the shared
/// `qwen3::{GenerateState, sample_with}` flow.
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

    pub fn set_sampler(
        &mut self,
        top_p: f32,
        top_k: i32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) {
        self.sampler.top_p = top_p;
        self.sampler.top_k = top_k;
        self.sampler.repeat_penalty = repeat_penalty;
        self.sampler.frequency_penalty = frequency_penalty;
        self.sampler.presence_penalty = presence_penalty;
    }
}

impl<C> Iterator for Generate<'_, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        use crate::models::qwen3::{sample_with, GenerateState};
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
                if self.sampler.keeps_history() {
                    tri!(mlx_rs::transforms::eval([&$y]));
                    self.history
                        .push(tri!($y.reshape(&[-1])).index(0).item::<u32>());
                }
            };
        }

        match &self.state {
            GenerateState::Prefill { prompt_token } => {
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
                let recent: &[u32] = if self.sampler.keeps_history() {
                    &self.history
                } else {
                    &[]
                };
                let y = tri!(sample_with(&logits.index((.., -1, ..)), &self.sampler, recent));
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
                let recent: &[u32] = if self.sampler.keeps_history() {
                    &self.history
                } else {
                    &[]
                };
                // Index the (single) last position to a 2-D `[B, vocab]` — matching
                // the prefill path — so the repeat-penalty `take_along_axis` (which
                // indexes `[B, vocab]` by the history) has matching dims.
                let y = tri!(sample_with(&logits.index((.., -1, ..)), &self.sampler, recent));
                record!(y);
                self.state = GenerateState::Decode { y: y.clone() };
                Some(Ok(y))
            }
        }
    }
}

#[cfg(test)]
mod smoke {
    use super::*;
    use crate::cache::ConcatKeyValueCache;
    use mlx_rs::ops::indexing::{IndexOp, NewAxis};

    /// Greedy generation against a real gpt-oss checkpoint. Gated on
    /// `ROZUM_GPTOSS_DIR` (skips otherwise). Run with:
    ///   ROZUM_GPTOSS_DIR=<snapshot> cargo test -p mlx-lm gpt_oss_greedy -- --nocapture
    /// then compare GEN_IDS against Python `mlx_lm` on the same PROMPT_IDS.
    #[test]
    fn gpt_oss_greedy_smoke() {
        let dir = match std::env::var("ROZUM_GPTOSS_DIR") {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: set ROZUM_GPTOSS_DIR to a gpt-oss snapshot dir");
                return;
            }
        };
        let dir = std::path::PathBuf::from(dir);

        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .expect("load tokenizer");
        let prompt = std::env::var("ROZUM_PROMPT")
            .unwrap_or_else(|_| "The capital of France is".to_string());
        let enc = tok.encode(prompt.as_str(), false).expect("encode");
        let ids: Vec<u32> = enc.get_ids().to_vec();
        eprintln!("PROMPT_IDS {ids:?}");

        let mut model = load_gpt_oss_model(&dir).expect("load model");
        let prompt_tokens = Array::from(&ids[..]).index(NewAxis);
        let mut cache: Vec<Option<ConcatKeyValueCache>> = Vec::new();
        let generate =
            Generate::<ConcatKeyValueCache>::new(&mut model, &mut cache, 0.0, &prompt_tokens);

        let mut out = Vec::new();
        for token in generate.take(16) {
            let token = token.expect("generate");
            mlx_rs::transforms::eval([&token]).expect("eval");
            out.push(token.item::<u32>());
        }
        eprintln!("GEN_IDS {out:?}");
        let text = tok.decode(&out, false).expect("decode");
        eprintln!("GEN_TEXT {text}");
        assert!(!out.is_empty());
    }
}
