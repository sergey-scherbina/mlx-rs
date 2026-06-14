//! Qwen3.6 / Qwen3-Next dense text model (`qwen3_5`, e.g. Qwen3.6-27B).
//!
//! Hybrid stack: every `full_attention_interval`-th layer is output-gated full
//! attention (partial RoPE), the rest are GatedDeltaNet linear-attention layers
//! (depthwise Conv1d short-conv + the delta-rule scan in [`super::gated_delta`]).
//! Caches are heterogeneous per layer (KV for full attention, conv+recurrent
//! state for the linear layers). Single-stream (batch 1), so the SSM mask is None
//! and the conv cache is just the trailing `kernel-1` positions. RMSNorm weights
//! carry the Qwen3.6 `+1` convention (applied at load).

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::{concatenate_axis, indexing::IndexOp, indexing::NewAxis, split_sections},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::Error,
    models::{
        gated_delta::gated_delta_update,
        qwen3::{Mlp, QuantizationConfig},
    },
    utils::rope::{initialize_rope, FloatOrString, RopeVariant},
};

/// Default prompt-prefill chunk size (tokens). Caps the full-attention
/// `[chunk, ctx]` causal-mask + score peak instead of `[T, T]` for a long prompt.
const PREFILL_CHUNK_DEFAULT: i32 = 2048;
/// Floor for a chunk size; tiny chunks only add per-chunk overhead.
const PREFILL_CHUNK_MIN: i32 = 256;

/// Prefill chunk size, env-tunable via `ROZUM_MLX_PREFILL_CHUNK` (floored to
/// [`PREFILL_CHUNK_MIN`]). Shared by the dense and MoE Qwen3.6 prefill paths.
pub fn prefill_chunk_size() -> i32 {
    std::env::var("ROZUM_MLX_PREFILL_CHUNK")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .filter(|&v| v > 0)
        .map(|v| v.max(PREFILL_CHUNK_MIN))
        .unwrap_or(PREFILL_CHUNK_DEFAULT)
}

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
    pub max_position_embeddings: i32,
    // rope_theta / partial_rotary_factor live either top-level or (on Qwen3.6)
    // nested under `rope_parameters`; resolved via the accessors below.
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    pub full_attention_interval: i32,
    pub linear_num_value_heads: i32,
    pub linear_num_key_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f32>,
}

impl ModelArgs {
    fn is_linear(&self, layer_idx: i32) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }
    fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10_000_000.0)
    }
    fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| r.partial_rotary_factor)
            .or(self.partial_rotary_factor)
            .unwrap_or(0.25)
    }
    fn rotary_dim(&self) -> i32 {
        (self.head_dim as f32 * self.partial_rotary_factor()) as i32
    }
}

/// Weightless RMS norm over the last axis (`mx.fast.rms_norm(x, None, eps)`).
/// Uses the null-weight fast kernel — no per-call ones weight (which would add a
/// `Full`+`AsType` to the graph every call).
fn rms_norm_weightless(x: &Array, eps: f32) -> Result<Array, Exception> {
    mlx_rs::fast::rms_norm_no_weight(x, eps)
}

/// RMSNorm with an optional SwiGLU-style gate: `silu(gate) * rms_norm(x, w)`.
#[derive(Debug, Clone, ModuleParameters)]
pub struct RmsNormGated {
    pub eps: f32,
    #[param]
    pub weight: Param<Array>,
}

impl RmsNormGated {
    fn new(dims: i32, eps: f32) -> Self {
        Self {
            eps,
            weight: Param::new(Array::ones::<f32>(&[dims]).unwrap()),
        }
    }

    fn forward(&self, x: &Array, gate: Option<&Array>) -> Result<Array, Exception> {
        let normed = mlx_rs::fast::rms_norm(x, &self.weight.value, self.eps)?;
        match gate {
            Some(g) => nn::silu(g)?.multiply(&normed),
            None => Ok(normed),
        }
    }
}

// ─── Full attention (output-gated, partial RoPE) ─────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub q_norm: nn::RmsNorm,
    #[param]
    pub k_norm: nn::RmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let d = args.hidden_size;
        let nh = args.num_attention_heads;
        let nkv = args.num_key_value_heads;
        let hd = args.head_dim;
        let q_proj = nn::LinearBuilder::new(d, nh * hd * 2).bias(false).build()?;
        let k_proj = nn::LinearBuilder::new(d, nkv * hd).bias(false).build()?;
        let v_proj = nn::LinearBuilder::new(d, nkv * hd).bias(false).build()?;
        let o_proj = nn::LinearBuilder::new(nh * hd, d).bias(false).build()?;
        let q_norm = nn::RmsNormBuilder::new(hd).eps(args.rms_norm_eps).build()?;
        let k_norm = nn::RmsNormBuilder::new(hd).eps(args.rms_norm_eps).build()?;
        let rope = initialize_rope(
            args.rotary_dim(),
            args.rope_theta(),
            false,
            &None,
            args.max_position_embeddings,
        )?;
        Ok(Self {
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            scale: (hd as f32).sqrt().recip(),
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        x: &Array,
        causal: bool,
        cache: Option<&mut ConcatKeyValueCache>,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let (B, L) = (shape[0], shape[1]);

        // q_proj -> queries + gate (split the doubled head dim).
        let qg = self
            .q_proj
            .forward(x)?
            .reshape(&[B, L, self.n_heads, self.head_dim * 2])?;
        let parts = split_sections(&qg, &[self.head_dim], -1)?;
        let queries = &parts[0]; // [B,L,nh,hd]
        let gate = parts[1].reshape(&[B, L, -1])?; // [B,L,nh*hd]

        let mut queries = self
            .q_norm
            .forward(queries)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x)?
                    .reshape(&[B, L, self.n_kv_heads, -1])?,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let (keys, values) = if let Some(cache) = cache {
            let qin = nn::RopeInputBuilder::new(&queries)
                .offset(cache.offset())
                .build()?;
            queries = self.rope.forward(qin)?;
            let kin = nn::RopeInputBuilder::new(&keys)
                .offset(cache.offset())
                .build()?;
            keys = self.rope.forward(kin)?;
            cache.update_and_fetch(keys, values)?
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
            (keys, values)
        };

        // Fused causal SDPA: MLX's built-in causal mode skips the masked upper
        // triangle and avoids an explicit `[L, ctx]` mask array (queries align to
        // the last `L` of the cached keys). Decode (L==1) needs no mask.
        let mask = causal.then_some(mlx_rs::fast::ScaledDotProductAttentionMask::Causal);
        let attn =
            mlx_rs::fast::scaled_dot_product_attention(queries, keys, values, self.scale, mask, None)?;
        let out = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        let gated = out.multiply(&nn::sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
    }
}

// ─── GatedDeltaNet linear attention ──────────────────────────────────────────

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
#[allow(non_snake_case)]
pub struct GatedDeltaNet {
    pub num_v_heads: i32,
    pub num_k_heads: i32,
    pub head_k_dim: i32,
    pub head_v_dim: i32,
    pub key_dim: i32,
    pub value_dim: i32,
    pub conv_dim: i32,
    pub conv_kernel_size: i32,

    #[quantizable]
    #[param]
    pub in_proj_qkv: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub in_proj_z: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub in_proj_b: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub in_proj_a: MaybeQuantized<nn::Linear>,
    #[param]
    pub conv1d: nn::Conv1d,
    #[param]
    pub A_log: Param<Array>,
    #[param]
    pub dt_bias: Param<Array>,
    #[param]
    pub norm: RmsNormGated,
    #[quantizable]
    #[param]
    pub out_proj: MaybeQuantized<nn::Linear>,
}

impl GatedDeltaNet {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let hidden = args.hidden_size;
        let num_v_heads = args.linear_num_value_heads;
        let num_k_heads = args.linear_num_key_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let k = args.linear_conv_kernel_dim;

        let conv1d = nn::Conv1dBuilder::new(conv_dim, conv_dim, k)
            .groups(conv_dim)
            .bias(false)
            .padding(0)
            .build()?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size: k,
            in_proj_qkv: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden, conv_dim)
                    .bias(false)
                    .build()?,
            ),
            in_proj_z: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden, value_dim)
                    .bias(false)
                    .build()?,
            ),
            in_proj_b: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden, num_v_heads)
                    .bias(false)
                    .build()?,
            ),
            in_proj_a: MaybeQuantized::Original(
                nn::LinearBuilder::new(hidden, num_v_heads)
                    .bias(false)
                    .build()?,
            ),
            conv1d,
            A_log: Param::new(Array::ones::<f32>(&[num_v_heads])?),
            dt_bias: Param::new(Array::ones::<f32>(&[num_v_heads])?),
            norm: RmsNormGated::new(head_v_dim, args.rms_norm_eps),
            out_proj: MaybeQuantized::Original(
                nn::LinearBuilder::new(value_dim, hidden)
                    .bias(false)
                    .build()?,
            ),
        })
    }

    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        conv_state: &mut Option<Array>,
        rec_state: &mut Option<Array>,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let (B, S) = (shape[0], shape[1]);

        let qkv = self.in_proj_qkv.forward(inputs)?; // [B,S,conv_dim]
        let z =
            self.in_proj_z
                .forward(inputs)?
                .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;
        let b = self.in_proj_b.forward(inputs)?; // [B,S,num_v_heads]
        let a = self.in_proj_a.forward(inputs)?;

        let prev = match conv_state.take() {
            Some(s) => s,
            None => Array::zeros::<f32>(&[B, self.conv_kernel_size - 1, self.conv_dim])?
                .as_dtype(inputs.dtype())?,
        };
        let conv_input = concatenate_axis(&[&prev, &qkv], 1)?; // [B,S+k-1,conv_dim]
        let n_keep = self.conv_kernel_size - 1;
        let total = conv_input.shape()[1];
        *conv_state = Some(conv_input.index((.., (total - n_keep)..total, ..)));

        let conv_out = nn::silu(&self.conv1d.forward(&conv_input)?)?; // [B,S,conv_dim]
        let qkv_parts = split_sections(&conv_out, &[self.key_dim, 2 * self.key_dim], -1)?;
        let q = qkv_parts[0].reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let k = qkv_parts[1].reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let v = qkv_parts[2].reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        // Per-head L2 (weightless) normalize + the delta-rule scaling on q/k.
        // Scale by a scalar IN q/k's dtype (Python multiplies by a python float,
        // which keeps bf16); a strong f32 `Array::from_f32` would promote the whole
        // stream to f32 -> ~1000 spurious weight-casts/token downstream. See
        // docs/mlx-gd-bug: the f32 leak started here.
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let qn = rms_norm_weightless(&q, 1e-6)?;
        let q = qn.multiply(Array::from_f32(inv_scale * inv_scale).as_dtype(qn.dtype())?)?;
        let kn = rms_norm_weightless(&k, 1e-6)?;
        let k = kn.multiply(Array::from_f32(inv_scale).as_dtype(kn.dtype())?)?;

        let (out, new_state) = gated_delta_update(
            &q,
            &k,
            &v,
            &a,
            &b,
            &self.A_log.value,
            &self.dt_bias.value,
            rec_state.take(),
        )?;
        *rec_state = Some(new_state);

        let normed = self.norm.forward(&out, Some(&z))?; // [B,S,num_v_heads,head_v_dim]
        let result = self.out_proj.forward(&normed.reshape(&[B, S, -1])?);
        if std::env::var("ROZUM_GDN_DEBUG").is_ok() {
            let l2 = |x: &Array| -> f32 {
                x.index((0, -1))
                    .square()
                    .and_then(|s| s.sum(None))
                    .and_then(|s| s.sqrt())
                    .map(|s| s.item::<f32>())
                    .unwrap_or(f32::NAN)
            };
            let conv_l2 = conv_out
                .index((0, -1, ..))
                .square()
                .and_then(|s| s.sum(None))
                .and_then(|s| s.sqrt())
                .map(|s| s.item::<f32>())
                .unwrap_or(f32::NAN);
            eprintln!(
                "GDN qkv_l2={:.3} conv_l2={conv_l2:.3} delta_l2={:.3} norm_l2={:.3} out_l2={:.3}",
                qkv.index((0, -1, ..))
                    .square()
                    .and_then(|s| s.sum(None))
                    .and_then(|s| s.sqrt())
                    .map(|s| s.item::<f32>())
                    .unwrap_or(f32::NAN),
                l2(&out),
                l2(&normed),
                result.as_ref().map(|r| l2(r)).unwrap_or(f32::NAN),
            );
        }
        result
    }

    fn training_mode(&mut self, mode: bool) {
        self.in_proj_qkv.training_mode(mode);
        self.in_proj_z.training_mode(mode);
        self.in_proj_b.training_mode(mode);
        self.in_proj_a.training_mode(mode);
        self.out_proj.training_mode(mode);
    }
}

// ─── Heterogeneous per-layer cache ───────────────────────────────────────────

pub enum LayerCache {
    Full(ConcatKeyValueCache),
    Linear {
        conv: Option<Array>,
        state: Option<Array>,
    },
}

impl LayerCache {
    /// Push this layer's live cache arrays into `out`. Used to eval all caches
    /// between prefill chunks: forcing them materializes the whole chunk forward
    /// (each layer's cache depends on the previous layer's output), so the chunk's
    /// activations are freed before the next chunk and the deferred graph does not
    /// span the whole prompt.
    pub fn collect_eval<'a>(&'a self, out: &mut Vec<&'a Array>) {
        match self {
            LayerCache::Full(kv) => out.extend(kv.state_arrays()),
            LayerCache::Linear { conv, state } => {
                out.extend(conv.iter());
                out.extend(state.iter());
            }
        }
    }

    /// Drop a `Full` (KV) layer back to its first `len` positions for prefix reuse;
    /// `Linear` is a no-op here — its recurrent state can't be truncated, so it is
    /// restored from a [`LinearSnap`] snapshot instead (see [`Self::snapshot`]).
    pub fn truncate(&mut self, len: i32) {
        if let LayerCache::Full(kv) = self {
            kv.truncate(len);
        }
    }

    /// Deep-copy this layer's recurrent (`Linear`) state into an independent buffer
    /// for a prefix snapshot, taken at the end of prefill (offset == prompt len).
    /// `deep_clone` forces materialization + a fresh buffer, so the snapshot survives
    /// any buffer donation by the subsequent decode steps. `Full` needs no snapshot
    /// (its KV buffer is kept live and truncated).
    pub fn snapshot(&self) -> LinearSnap {
        match self {
            LayerCache::Full(_) => LinearSnap::Full,
            LayerCache::Linear { conv, state } => LinearSnap::Linear {
                conv: conv.as_ref().map(|a| {
                    let _ = a.eval();
                    a.deep_clone()
                }),
                state: state.as_ref().map(|a| {
                    let _ = a.eval();
                    a.deep_clone()
                }),
            },
        }
    }

    /// Restore a `Linear` layer's recurrent state from a snapshot (deep-cloned again
    /// so the persisted snapshot stays pristine for the next reuse). `Full` is a
    /// no-op (handled by [`Self::truncate`]).
    pub fn restore(&mut self, snap: &LinearSnap) {
        if let (
            LayerCache::Linear { conv, state },
            LinearSnap::Linear { conv: sc, state: ss },
        ) = (self, snap)
        {
            *conv = sc.as_ref().map(|a| a.deep_clone());
            *state = ss.as_ref().map(|a| a.deep_clone());
        }
    }
}

/// A deep-copied snapshot of a layer's recurrent (`Linear`) state at the end of
/// prefill, for cross-request prefix reuse. `Full` (KV) layers carry no snapshot —
/// their KV buffer is kept live and truncated to the shared prefix; `Linear` layers
/// can't be truncated (the GatedDeltaNet state is a running summary), so their small
/// conv + recurrent state is deep-copied and restored on the next reuse.
#[derive(Clone)]
pub enum LinearSnap {
    Full,
    Linear { conv: Option<Array>, state: Option<Array> },
}

// ─── Decoder layer / model ───────────────────────────────────────────────────

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
    pub mlp: Mlp,
}

impl DecoderLayer {
    fn new(args: &ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let is_linear = args.is_linear(layer_idx);
        let (self_attn, linear_attn) = if is_linear {
            (None, Some(GatedDeltaNet::new(args)?))
        } else {
            (Some(Attention::new(args)?), None)
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
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
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
            _ => return Err(Exception::custom("qwen3_5: cache/layer kind mismatch")),
        };
        let h = x.add(&r)?;
        let m = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(&m)
    }

    fn training_mode(&mut self, mode: bool) {
        if let Some(a) = &mut self.self_attn {
            a.training_mode(mode);
        }
        if let Some(l) = &mut self.linear_attn {
            l.training_mode(mode);
        }
        self.mlp.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Qwen3_5Model {
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3_5Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
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
        let t = h.shape()[1];
        // Prefill (T>1) uses fused causal SDPA; decode (T==1) needs no mask. MLX's
        // causal mode handles the KV-cache offset (queries align to the last T keys),
        // so no explicit `[T, ctx]` mask array is built.
        let causal = t > 1;
        let dbg = std::env::var("ROZUM_LAYER_DEBUG").is_ok();
        if dbg {
            let l2 = h
                .index((0, -1, ..))
                .square()?
                .sum(None)?
                .sqrt()?
                .item::<f32>();
            eprintln!("EMBED last_l2={l2:.4}");
        }
        for (i, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            h = layer.forward(&h, causal, c)?;
            if dbg && i < 6 {
                let l2 = h
                    .index((0, -1, ..))
                    .square()?
                    .sum(None)?
                    .sqrt()?
                    .item::<f32>();
                let kind = if layer.is_linear { "linear" } else { "full" };
                eprintln!("LAYER {i} ({kind}) last_l2={l2:.4}");
            }
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
    pub model: Qwen3_5Model,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Qwen3_5Model::new(&args)?;
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

    /// Prefill a (possibly long) prompt, returning logits for the LAST position
    /// only (`[B, 1, vocab]`) — all `Generate` needs to sample the first token.
    /// The prompt is processed in chunks of [`prefill_chunk_size`] so the
    /// full-attention layers bound their `[chunk, ctx]` causal-mask + score peak
    /// instead of `[T, T]`; the caches advance across chunks and are eval'd
    /// between them to free each chunk's activations. The GatedDeltaNet layers
    /// are already O(1) memory. `lm_head` is applied only to the final position
    /// (the per-chunk hidden states feed only the caches), so the big vocab
    /// projection never runs on discarded positions. The result is byte-identical
    /// to a single-pass `forward` of the last position (per-position attention +
    /// sequential delta scan are position-local; chunking only changes when
    /// intermediates are freed).
    pub fn prefill(
        &mut self,
        inputs: &Array,
        cache: &mut [LayerCache],
    ) -> Result<Array, Exception> {
        self.prefill_chunked(inputs, cache, prefill_chunk_size())
    }

    /// [`prefill`](Self::prefill) with an explicit chunk size (the env-driven
    /// default goes through `prefill`). Exposed so tests can compare chunked vs
    /// single-pass output.
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
    /// between chunks; returns `Ok(None)` if it fired (the caller ends the run).
    pub fn prefill_cancellable(
        &mut self,
        inputs: &Array,
        cache: &mut [LayerCache],
        chunk: i32,
        should_cancel: &dyn Fn() -> bool,
    ) -> Result<Option<Array>, Exception> {
        let t = inputs.shape()[1];
        let mut start = 0;
        // Run the backbone (no lm_head) per chunk; keep only the final position's
        // hidden state, then project that one position.
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
            // Force this chunk's caches (-> its whole forward), freeing the
            // chunk's activations before the next chunk and keeping the deferred
            // graph from spanning the prompt.
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

/// `config.json` is a multimodal wrapper; the text model lives under
/// `text_config`. The top-level `quantization` applies to the text weights.
#[derive(Debug, Clone, Deserialize)]
struct WrappedConfig {
    text_config: ModelArgs,
    quantization: Option<QuantizationConfig>,
}

const NORM_PLUS_ONE_SUFFIXES: &[&str] = &[
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
];

pub fn load_qwen3_5_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let text = std::fs::read_to_string(model_dir.join("config.json"))?;
    // Either a multimodal wrapper (text_config) or a bare text config.
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

    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    fn load_weights_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let mut params = model.parameters_mut().flatten();
        let param_keys: HashSet<String> = params.keys().map(|k| k.to_string()).collect();
        let mut matched = 0usize;
        for (key, mut value) in loaded {
            // Multimodal checkpoint: text weights live under `language_model.`;
            // skip the vision tower. A bare text checkpoint has no prefix.
            let key = if let Some(k) = key.strip_prefix("language_model.") {
                k.to_string()
            } else if key.starts_with("vision_tower.") || key.starts_with("visual.") {
                continue;
            } else {
                key
            };
            // NOTE: Python `sanitize` only applies the RMSNorm `+1` convention
            // (and expert fusion) when the checkpoint is in raw, unfused form
            // (`...experts.0.up_proj.weight` present). The mlx-community 4bit
            // checkpoints are already sanitized (norm weights centered at ~1, so
            // adding 1 would double them), so we do NOT add it here. The raw
            // path is revisited for the unfused MoE variant.
            let _ = NORM_PLUS_ONE_SUFFIXES;
            // HF depthwise conv weight [C,1,k] -> MLX Conv1d [C,k,1] (self-guards
            // on the trailing dim, so a pre-baked [C,k,1] is left untouched).
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
        // No index.json: load every `model-*.safetensors` shard, else the
        // single `model.safetensors`.
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
        eprintln!("LOADED {matched} params (qwen3_5)");
    }
    model.eval()?;
    Ok(model)
}

/// Greedy/categorical token iterator (prefill once, then single-token decode).
/// Owns the heterogeneous cache.
pub struct Generate<'a> {
    model: &'a mut Model,
    cache: Vec<LayerCache>,
    sampler: crate::models::qwen3::SamplerOpts,
    /// Generated-token history for the repetition penalty (only maintained when
    /// `sampler.repeat_penalty != 1.0`).
    history: Vec<u32>,
    state: GenState<'a>,
    /// Cooperative cancel: polled between prefill chunks so a client disconnect
    /// is honored mid-prefill (a long prompt can take seconds), not only between
    /// decode tokens. Default never cancels; the host wires it via `set_cancel`.
    should_cancel: Box<dyn Fn() -> bool + Send>,
    /// Linear (GatedDeltaNet) state snapshot taken at the END of prefill (offset ==
    /// prompt len), for cross-request prefix reuse. Filled on the first (prefill)
    /// `next()`; `None` if prefill was cancelled. The host persists it alongside the
    /// (advanced) cache to restore the recurrent state next turn.
    prefill_snapshot: Option<Vec<LinearSnap>>,
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
    /// `prompt_token` (the new suffix) on top of `cache` instead of from scratch.
    /// The host has already truncated the `Full` layers to the shared prefix and
    /// restored the `Linear` layers from a snapshot (see [`LayerCache`]).
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
        }
    }

    /// Consume the iterator, returning the (now advanced) cache + the Linear-state
    /// snapshot taken at end-of-prefill (`None` if prefill never completed). For
    /// prefix reuse next turn: truncate the `Full` layers + restore `Linear` from
    /// the snapshot.
    pub fn into_cache_and_snapshot(self) -> (Vec<LayerCache>, Option<Vec<LinearSnap>>) {
        (self.cache, self.prefill_snapshot)
    }

    /// Install a cancellation predicate, polled between prefill chunks. When it
    /// returns true mid-prefill the iterator ends (`next() -> None`).
    pub fn set_cancel(&mut self, should_cancel: Box<dyn Fn() -> bool + Send>) {
        self.should_cancel = should_cancel;
    }

    /// Set top-p / top-k / repeat-penalty (temp came from `new`). `top_k<=0`,
    /// `top_p>=1`, `repeat_penalty==1` disable the respective filter.
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
        // Prefill chunks the prompt (bounded full-attention peak) and returns the
        // last position only; decode (T=1) is a plain forward. Prefill is
        // cancellable between chunks -> None ends the iteration on a mid-prefill
        // cancel.
        let logits = if is_prefill {
            match tri!(self.model.prefill_cancellable(
                &inputs,
                &mut self.cache,
                prefill_chunk_size(),
                &*self.should_cancel,
            )) {
                Some(l) => l,
                None => return None,
            }
        } else {
            tri!(self.model.forward(&inputs, &mut self.cache))
        };
        // At end of prefill (offset == prompt len, no decode token in the cache yet)
        // snapshot the recurrent `Linear` state for cross-request prefix reuse.
        if is_prefill {
            self.prefill_snapshot = Some(self.cache.iter().map(|c| c.snapshot()).collect());
        }
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
        // Maintain history only when the penalty is active (needs the token id ->
        // an eval; the host re-uses the now-materialized token, so no double sync).
        if self.sampler.repeat_penalty != 1.0 {
            tri!(mlx_rs::transforms::eval([&y]));
            self.history.push(y.index(0).item::<u32>());
        }
        self.state = GenState::Decode(y.clone());
        Some(Ok(y))
    }
}
