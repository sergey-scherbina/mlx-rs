//! GLM-4.7-Flash (`model_type: "glm4_moe_lite"`) — absorbed MLA + DeepSeek-V3 MoE.
//! Ported from Python `mlx_lm.models.glm4_moe_lite` (+ `mla.py` MultiLinear). Target: GLM-4.7-Flash.
//! Absorbed MLA (embed_q/unembed_out = per-head QuantizedMultiLinear, compressed kv_latent+k_pe
//! cache, pe_scores SDPA mask, L==1 decode vs L>1 prefill) + V3 noaux_tc routing (sigmoid +
//! e_score_correction_bias). Reuses deepseek_v2's MLP/MoE-experts/Model/load. Spec:
//! rozum:docs/specs/glm4-moe-lite-native.md. COMPILES; byte-parity vs Python is slot-gated.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, ModuleParametersExt},
    nn,
    ops::{
        argpartition_axis, expand_dims_axes,
        indexing::{take_along_axis, IndexOp, NewAxis},

    },
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    cache::KeyValueCache,
    error::Error,
    models::qwen3::{repeat_window, sample_with, GenerateState, QuantizationConfig, SamplerOpts},
    models::qwen3_moe::SwitchGlu,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant, YarnRope},
        AttentionMask,
    },
};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub rope_theta: f32,
    #[serde(default = "d_max_pos")]
    pub max_position_embeddings: i32,
    #[serde(default = "d_true")]
    pub tie_word_embeddings: bool,
    // ── MLA latent dims ──────────────────────────────────────────────────────────────────────
    /// `None` (config `null`, e.g. DeepSeek-Coder-V2-Lite) ⇒ no q low-rank: a plain `q_proj`.
    pub q_lora_rank: Option<i32>,
    pub kv_lora_rank: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub v_head_dim: i32,
    #[serde(default = "d_true")]
    pub attention_bias: bool,
    // ── DeepSeek MoE ─────────────────────────────────────────────────────────────────────────
    #[serde(default)]
    pub n_routed_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub n_shared_experts: i32,
    /// First k layers use the dense MLP, the rest MoE (HF: `first_k_dense_replace`).
    #[serde(default, alias = "first_k_dense_replace")]
    pub first_k_dense_layers: i32,
    #[serde(default = "d_one_i")]
    pub moe_layer_freq: i32,
    #[serde(default = "d_one_f32")]
    pub routed_scaling_factor: f32,
    #[serde(default = "d_true")]
    pub norm_topk_prob: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    pub quantization: Option<QuantizationConfig>,
}
fn d_max_pos() -> i32 { 4096 }
fn d_true() -> bool { true }
fn d_one_i() -> i32 { 1 }
fn d_one_f32() -> f32 { 1.0 }

/// DeepSeek YaRN attention-scale multiplier: `0.1 * mscale * ln(scale) + 1` (1.0 when scale<=1).
fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 { 1.0 } else { 0.1 * mscale * scale.ln() + 1.0 }
}
/// Read a numeric field out of the `rope_scaling` map.
fn rope_f(rs: &HashMap<String, FloatOrString>, k: &str) -> Option<f32> {
    match rs.get(k) {
        Some(FloatOrString::Float(v)) => Some(*v),
        _ => None,
    }
}

impl ModelArgs {
    fn is_moe(&self, layer_idx: i32) -> bool {
        self.n_routed_experts > 0
            && layer_idx >= self.first_k_dense_layers
            && layer_idx % self.moe_layer_freq == 0
    }
}

// ── Per-head quantized linear (mlx_lm.models.mla.QuantizedMultiLinear) — weight [H, out, in] ────
// GLM-4.7-Flash's `embed_q`/`unembed_out` are per-head MultiLinear; the 4bit checkpoint quantizes
// them. Loaded raw (like SwitchGlu's QSwitchLinear) — NOT via nn::quantize.
#[derive(Debug, Clone, ModuleParameters)]
pub struct QMultiLinear {
    pub group_size: i32,
    pub bits: i32,
    #[param] pub weight: mlx_rs::module::Param<Array>,
    #[param] pub scales: mlx_rs::module::Param<Array>,
    #[param] pub biases: mlx_rs::module::Param<Array>,
}
impl QMultiLinear {
    fn new(group_size: i32, bits: i32) -> Self {
        Self {
            group_size,
            bits,
            weight: mlx_rs::module::Param::new(array!(0.0)),
            scales: mlx_rs::module::Param::new(array!(0.0)),
            biases: mlx_rs::module::Param::new(array!(0.0)),
        }
    }
    /// `transpose=true`: x @ Wᵀ (in→out); `false`: x @ W (out→in). Per-head batched.
    fn forward(&self, x: &Array, transpose: bool) -> Result<Array, Exception> {
        mlx_rs::ops::quantized_matmul(
            x,
            &self.weight.value,
            &self.scales.value,
            &self.biases.value,
            transpose,
            self.group_size,
            self.bits,
        )
    }
}

// ── GLM-4.7-Flash absorbed MLA (mlx_lm.models.glm4_moe_lite.Glm4MoeLiteAttention) ───────────────
// Absorbed form: `embed_q`/`unembed_out` (per-head) fold kv_b into attention; the cache stores the
// COMPRESSED latent (kv_latent + k_pe), and the rope part is a separate additive score (pe_scores)
// passed as the SDPA mask. Decode (L==1) and prefill (L>1) take different absorption paths.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MlaAttention {
    pub num_heads: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub q_head_dim: i32,
    pub v_head_dim: i32,
    pub kv_lora_rank: i32,
    pub scale: f32,

    #[quantizable] #[param] pub q_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable] #[param] pub q_a_proj: Option<MaybeQuantized<nn::Linear>>,
    #[param] pub q_a_layernorm: Option<nn::RmsNorm>,
    #[quantizable] #[param] pub q_b_proj: Option<MaybeQuantized<nn::Linear>>,
    #[quantizable] #[param] pub kv_a_proj_with_mqa: MaybeQuantized<nn::Linear>,
    #[param] pub kv_a_layernorm: nn::RmsNorm,
    #[param] pub embed_q: QMultiLinear,
    #[param] pub unembed_out: QMultiLinear,
    #[quantizable] #[param] pub o_proj: MaybeQuantized<nn::Linear>,
    #[param] pub rope: RopeVariant,
}

impl MlaAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let d = args.hidden_size;
        let h = args.num_attention_heads;
        let q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim;
        // scale = q_head_dim^-0.5 (× YaRN mscale² if rope_scaling.mscale_all_dim — GLM-4.7-Flash has
        // rope_scaling=null, so this is a no-op here; kept for the yarn'd GLM variants).
        let mut scale = (q_head_dim as f32).powf(-0.5);
        if let Some(rs) = &args.rope_scaling {
            let mscale_all_dim = rope_f(rs, "mscale_all_dim").unwrap_or(0.0);
            if mscale_all_dim != 0.0 {
                let factor = rope_f(rs, "factor").unwrap_or(1.0);
                let m = yarn_get_mscale(factor, mscale_all_dim);
                scale *= m * m;
            }
        }
        let lin = |i, o, b| nn::LinearBuilder::new(i, o).bias(b).build();
        let rms = |dim| nn::RmsNormBuilder::new(dim).eps(args.rms_norm_eps).build();
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) = match args.q_lora_rank {
            Some(qr) if qr > 0 => (
                None,
                Some(MaybeQuantized::Original(lin(d, qr, args.attention_bias)?)),
                Some(rms(qr)?),
                Some(MaybeQuantized::Original(lin(qr, h * q_head_dim, false)?)),
            ),
            _ => (Some(MaybeQuantized::Original(lin(d, h * q_head_dim, false)?)), None, None, None),
        };
        let kv_a_proj_with_mqa = lin(d, args.kv_lora_rank + args.qk_rope_head_dim, args.attention_bias)?;
        let o_proj = lin(h * args.v_head_dim, d, args.attention_bias)?;
        let (gs, bits) = args.quantization.as_ref().map(|q| (q.group_size, q.bits)).unwrap_or((64, 4));
        let rope = match args.rope_scaling.as_ref() {
            Some(rs) if matches!(rs.get("type"), Some(FloatOrString::String(s)) if s == "yarn") => {
                let factor = rope_f(rs, "factor").unwrap_or(1.0);
                let beta_fast = rope_f(rs, "beta_fast").unwrap_or(32.0);
                let beta_slow = rope_f(rs, "beta_slow").unwrap_or(1.0);
                let orig_max = rope_f(rs, "original_max_position_embeddings")
                    .map(|v| v as i32).unwrap_or(args.max_position_embeddings);
                let mut yr = YarnRope::new(
                    args.qk_rope_head_dim, true, args.rope_theta, factor, orig_max, beta_fast, beta_slow,
                )?;
                let msc = rope_f(rs, "mscale").unwrap_or(1.0);
                let msc_all = rope_f(rs, "mscale_all_dim").unwrap_or(0.0);
                yr.mscale = yarn_get_mscale(factor, msc) / yarn_get_mscale(factor, msc_all);
                RopeVariant::Yarn(yr)
            }
            _ => initialize_rope(args.qk_rope_head_dim, args.rope_theta, true, &args.rope_scaling,
                args.max_position_embeddings)?,
        };
        Ok(Self {
            num_heads: h,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            q_head_dim,
            v_head_dim: args.v_head_dim,
            kv_lora_rank: args.kv_lora_rank,
            scale,
            q_proj, q_a_proj, q_a_layernorm, q_b_proj,
            kv_a_proj_with_mqa: MaybeQuantized::Original(kv_a_proj_with_mqa),
            kv_a_layernorm: rms(args.kv_lora_rank)?,
            embed_q: QMultiLinear::new(gs, bits),
            unembed_out: QMultiLinear::new(gs, bits),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for MlaAttention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;
        let s = x.shape();
        let (B, L) = (s[0], s[1]);
        let rope_d = self.qk_rope_head_dim;
        let nope = self.qk_nope_head_dim;

        let q = match self.q_proj.as_mut() {
            Some(qp) => qp.forward(x)?,
            None => {
                let qa = self.q_a_proj.as_mut().unwrap().forward(x)?;
                let qa = self.q_a_layernorm.as_mut().unwrap().forward(&qa)?;
                self.q_b_proj.as_mut().unwrap().forward(&qa)?
            }
        };
        let q = q.reshape(&[B, L, self.num_heads, self.q_head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((.., .., .., 0..nope)); // [B,H,L,nope]
        let mut q_pe = q.index((.., .., .., nope..(nope + rope_d))); // [B,H,L,rope_d]

        let ckv = self.kv_a_proj_with_mqa.forward(x)?;
        let compressed_kv = ckv.index((.., .., 0..self.kv_lora_rank));
        let mut k_pe = ckv
            .index((.., .., self.kv_lora_rank..(self.kv_lora_rank + rope_d)))
            .reshape(&[B, L, 1, rope_d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,1,L,rope_d]
        // kv_latent [B,1,L,kv_lora_rank]
        let kv_latent = self
            .kv_a_layernorm
            .forward(&compressed_kv)?
            .reshape(&[B, L, 1, self.kv_lora_rank])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let offset = cache.as_ref().map(|c| c.offset()).unwrap_or(0);
        q_pe = self.rope.forward(nn::RopeInputBuilder::new(&q_pe).offset(offset).build()?)?;
        k_pe = self.rope.forward(nn::RopeInputBuilder::new(&k_pe).offset(offset).build()?)?;

        // Cache the COMPRESSED latent + rope key (both single-head, concat on the seq axis).
        let (kv_latent, k_pe) = match cache.as_mut() {
            Some(c) => c.update_and_fetch(kv_latent, k_pe)?,
            None => (kv_latent, k_pe),
        };

        // Rope-part scores: pe_scores = (q_pe*scale) @ k_peᵀ  → [B,H,L,L'] (k_pe head=1 broadcasts).
        let pe_scores = mlx_rs::ops::matmul(
            &q_pe.multiply(array!(self.scale))?,
            &k_pe.transpose_axes(&[0, 1, 3, 2])?,
        )?;
        // Causal mask → -inf on the masked positions (the boolean mask comes from create_attention_mask).
        let pe_scores = match mask {
            Some(m) => {
                let neg = array!(f32::MIN);
                mlx_rs::ops::r#where(m, &pe_scores, &neg)?
            }
            None => pe_scores,
        };
        // SDPA requires the additive mask to promote to the output (bf16) dtype — the f32 scale
        // multiply + f32::MIN where-fill leave pe_scores in f32, which does NOT promote to bf16.
        let pe_scores = pe_scores.as_dtype(q_nope.dtype())?;

        // L==1 (decode): absorb q into latent, attend in latent space, unembed the output.
        // L>1  (prefill): unembed the latent into per-head k/v, attend in nope/v space.
        let output = if L == 1 {
            let q_a = self.embed_q.forward(&q_nope, true)?; // [B,H,1,kv_lora_rank]
            let out = crate::utils::scaled_dot_product_attention(
                q_a, kv_latent.clone(), kv_latent, None::<&mut C>, self.scale, Some(&pe_scores),
            )?;
            self.unembed_out.forward(&out, true)? // [B,H,1,v_head_dim]
        } else {
            let k = self.embed_q.forward(&kv_latent, false)?; // [B,H,L',nope]
            let v = self.unembed_out.forward(&kv_latent, true)?; // [B,H,L',v_head_dim]
            crate::utils::scaled_dot_product_attention(
                q_nope, k, v, None::<&mut C>, self.scale, Some(&pe_scores),
            )?
        };
        let out = output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;
        self.o_proj.forward(&out)
    }

    fn training_mode(&mut self, mode: bool) {
        if let Some(m) = self.q_proj.as_mut() { m.training_mode(mode); }
        if let Some(m) = self.q_a_proj.as_mut() { m.training_mode(mode); }
        if let Some(m) = self.q_a_layernorm.as_mut() { m.training_mode(mode); }
        if let Some(m) = self.q_b_proj.as_mut() { m.training_mode(mode); }
        self.kv_a_proj_with_mqa.training_mode(mode);
        self.kv_a_layernorm.training_mode(mode);
        self.o_proj.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ── Dense MLP (also the shared expert) — split gate/up/down (NOT fused) ─────────────────────────
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DeepseekV2MLP {
    #[quantizable] #[param] pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable] #[param] pub up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable] #[param] pub down_proj: MaybeQuantized<nn::Linear>,
}
impl DeepseekV2MLP {
    pub fn new(hidden: i32, inter: i32) -> Result<Self, Exception> {
        let lin = |i, o| nn::LinearBuilder::new(i, o).bias(false).build().map(MaybeQuantized::Original);
        Ok(Self { gate_proj: lin(hidden, inter)?, up_proj: lin(hidden, inter)?, down_proj: lin(inter, hidden)? })
    }
}
impl Module<&Array> for DeepseekV2MLP {
    type Output = Array;
    type Error = Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = nn::silu(&self.gate_proj.forward(x)?)?.multiply(&self.up_proj.forward(x)?)?;
        self.down_proj.forward(&h)
    }
    fn training_mode(&mut self, m: bool) {
        self.gate_proj.training_mode(m); self.up_proj.training_mode(m); self.down_proj.training_mode(m);
    }
}

// ── GLM/DeepSeek-V3 router gate: raw weight [n_experts, hidden] + e_score_correction_bias ────────
#[derive(Debug, Clone, ModuleParameters)]
pub struct MoEGate {
    #[param] pub weight: mlx_rs::module::Param<Array>,
    #[param] pub e_score_correction_bias: mlx_rs::module::Param<Array>,
}
impl MoEGate {
    fn new() -> Self {
        Self {
            weight: mlx_rs::module::Param::new(array!(0.0)),
            e_score_correction_bias: mlx_rs::module::Param::new(array!(0.0)),
        }
    }
    fn logits(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::ops::matmul(x, &self.weight.value.transpose_axes(&[1, 0])?)
    }
}

// ── GLM-4.7-Flash MoE: V3 `noaux_tc` router (sigmoid + correction-bias select) + experts + shared ─
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DeepseekV2MoE {
    pub top_k: i32,
    pub num_experts: i32,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    #[param] pub gate: MoEGate,
    #[param] pub switch_mlp: SwitchGlu,
    #[quantizable] #[param] pub shared_experts: Option<DeepseekV2MLP>,
}
impl DeepseekV2MoE {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let (gs, bits) = args.quantization.as_ref().map(|q| (q.group_size, q.bits)).unwrap_or((64, 4));
        let shared = if args.n_shared_experts > 0 {
            Some(DeepseekV2MLP::new(args.hidden_size, args.moe_intermediate_size * args.n_shared_experts)?)
        } else {
            None
        };
        Ok(Self {
            top_k: args.num_experts_per_tok,
            num_experts: args.n_routed_experts,
            routed_scaling_factor: args.routed_scaling_factor,
            norm_topk_prob: args.norm_topk_prob,
            gate: MoEGate::new(),
            switch_mlp: SwitchGlu::new(gs, bits),
            shared_experts: shared,
        })
    }
}
impl Module<&Array> for DeepseekV2MoE {
    type Output = Array;
    type Error = Exception;
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // DeepSeek-V3 `noaux_tc` (n_group=1): sigmoid scores; select by scores+correction_bias; weight
        // by the ORIGINAL sigmoid scores; norm_topk_prob; × routed_scaling_factor.
        let orig = mlx_rs::ops::sigmoid(&self.gate.logits(x)?)?;
        let sel = orig.add(&self.gate.e_score_correction_bias.value)?;
        let k = self.top_k;
        let inds = argpartition_axis(&sel, -k, -1)?;
        let inds = inds.index((.., .., (self.num_experts - k)..));
        let mut scores = take_along_axis(&orig, &inds, -1)?;
        if self.norm_topk_prob {
            let denom = scores.sum_axes(&[-1], true)?.add(array!(1e-20))?;
            scores = scores.divide(&denom)?;
        }
        let scores = scores.multiply(array!(self.routed_scaling_factor))?;
        let y = self.switch_mlp.forward(x, &inds)?; // [B,L,k,D]
        let mut y = y.multiply(&expand_dims_axes(&scores, &[-1])?)?.sum_axes(&[-2], false)?;
        if let Some(se) = self.shared_experts.as_mut() {
            y = y.add(&se.forward(x)?)?;
        }
        Ok(y)
    }
    fn training_mode(&mut self, m: bool) {
        let _ = m; // MoEGate is plain params (no training mode)
        if let Some(se) = self.shared_experts.as_mut() { se.training_mode(m); }
    }
}

// ── Decoder layer: MLA + (dense | MoE) FFN, standard 2-norm pre-norm residual ───────────────────
// One of `mlp_moe`/`mlp_dense` is Some per layer (chosen by `is_moe`). Derive-friendly (Option<M>);
// the load remap rewrites checkpoint `…mlp.X` → `…mlp_moe.X` / `…mlp_dense.X` (whichever exists).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    #[quantizable] #[param] pub self_attn: MlaAttention,
    #[quantizable] #[param] pub mlp_moe: Option<DeepseekV2MoE>,
    #[quantizable] #[param] pub mlp_dense: Option<DeepseekV2MLP>,
    #[param] pub input_layernorm: nn::RmsNorm,
    #[param] pub post_attention_layernorm: nn::RmsNorm,
}
impl DecoderLayer {
    fn new(args: &ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        let rms = || nn::RmsNormBuilder::new(args.hidden_size).eps(args.rms_norm_eps).build();
        let (mlp_moe, mlp_dense) = if args.is_moe(layer_idx) {
            (Some(DeepseekV2MoE::new(args)?), None)
        } else {
            (None, Some(DeepseekV2MLP::new(args.hidden_size, args.intermediate_size)?))
        };
        Ok(Self {
            self_attn: MlaAttention::new(args)?,
            mlp_moe,
            mlp_dense,
            input_layernorm: rms()?,
            post_attention_layernorm: rms()?,
        })
    }
}
impl<C> Module<AttentionInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Array, Exception> {
        let AttentionInput { x, mask, cache } = input;
        let r = self.self_attn.forward(AttentionInput { x: &self.input_layernorm.forward(x)?, mask, cache })?;
        let h = x.add(r)?;
        let ffn_in = self.post_attention_layernorm.forward(&h)?;
        let r = match (self.mlp_moe.as_mut(), self.mlp_dense.as_mut()) {
            (Some(moe), _) => moe.forward(&ffn_in)?,
            (_, Some(dense)) => dense.forward(&ffn_in)?,
            _ => unreachable!("a layer has exactly one FFN"),
        };
        h.add(r)
    }
    fn training_mode(&mut self, m: bool) {
        <MlaAttention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, m);
        if let Some(x) = self.mlp_moe.as_mut() { x.training_mode(m); }
        if let Some(x) = self.mlp_dense.as_mut() { x.training_mode(m); }
        self.input_layernorm.training_mode(m);
        self.post_attention_layernorm.training_mode(m);
    }
}

// ── Model + load (mirror glm4/qwen3_moe) ───────────────────────────────────────────────────────
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DeepseekV2Inner {
    pub num_hidden_layers: i32,
    #[quantizable] #[param] pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable] #[param] pub layers: Vec<DecoderLayer>,
    #[param] pub norm: nn::RmsNorm,
}
impl DeepseekV2Inner {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers).map(|i| DecoderLayer::new(args, i)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(args.vocab_size, args.hidden_size)?),
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size).eps(args.rms_norm_eps).build()?,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}
impl<C> Module<ModelInput<'_, C>> for DeepseekV2Inner
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;
    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Array, Exception> {
        let ModelInput { inputs, mask, cache } = input;
        let mut h = self.embed_tokens.forward(inputs)?;
        let mask = match mask {
            Some(m) => Some(m.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                _ => None,
            },
        };
        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(AttentionInput { x: &h, mask: mask.as_ref(), cache: c.as_mut() })?;
        }
        self.norm.forward(&h)
    }
    fn training_mode(&mut self, m: bool) {
        self.embed_tokens.training_mode(m);
        for l in &mut self.layers { <DecoderLayer as Module<AttentionInput<'_, C>>>::training_mode(l, m); }
        self.norm.training_mode(m);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable] #[param] pub model: DeepseekV2Inner,
    #[quantizable] #[param] pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}
impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = DeepseekV2Inner::new(&args)?;
        let lm_head = (!args.tie_word_embeddings)
            .then(|| nn::LinearBuilder::new(args.hidden_size, args.vocab_size).bias(false).build().map(MaybeQuantized::Original))
            .transpose()?;
        Ok(Self { args, model, lm_head })
    }
    pub fn model_type(&self) -> &str { &self.args.model_type }
}
impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;
    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Array, Exception> {
        let out = self.model.forward(input)?;
        match self.lm_head.as_mut() {
            Some(h) => h.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out),
                MaybeQuantized::Quantized(e) => e.as_linear(&out),
            },
        }
    }
    fn training_mode(&mut self, m: bool) {
        <DeepseekV2Inner as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, m);
        if let Some(h) = &mut self.lm_head { h.training_mode(m); }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn get_glm4_moe_lite_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    Ok(serde_json::from_reader(std::fs::File::open(model_dir.as_ref().join("config.json"))?)?)
}

pub fn load_glm4_moe_lite_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_glm4_moe_lite_model_args(model_dir)?;
    let quantization = args.quantization.clone();
    let mut model = Model::new(args)?;
    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    // AFQ remap + per-layer FFN field remap: a dense layer's `…mlp.X` lives under `…mlp_dense.X`,
    // a MoE layer's under `…mlp_moe.X`; pick whichever param key actually exists. `.weight` →
    // `.inner.weight` for quantized leaves (SwitchGlu experts keep `.weight`, no `.inner`).
    fn load_weights_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let mut params = model.parameters_mut().flatten();
        let keys: HashSet<String> = params.keys().map(|k| k.to_string()).collect();
        let mut matched = 0usize;
        for (key, value) in loaded {
            // 1) route `.mlp.` → `.mlp_moe.` / `.mlp_dense.` to whichever exists for this layer.
            let routed = if let Some(pos) = key.find(".mlp.") {
                let (pre, post) = key.split_at(pos); // post = ".mlp.<rest>"
                let rest = &post[5..];
                let moe = format!("{pre}.mlp_moe.{rest}");
                let dense = format!("{pre}.mlp_dense.{rest}");
                let probe = |s: &str| keys.contains(s) || keys.contains(&s.replacen(".weight", ".inner.weight", 1));
                if probe(&moe) { moe } else if probe(&dense) { dense } else { key.clone() }
            } else {
                key.clone()
            };
            // 2) `.weight` → `.inner.weight` for quantized leaves.
            let mapped = match routed.strip_suffix(".weight") {
                Some(p) if keys.contains(&format!("{p}.inner.weight")) => format!("{p}.inner.weight"),
                _ => routed,
            };
            if let Some(param) = params.get_mut(mapped.as_str()) {
                **param = value;
                matched += 1;
            }
        }
        Ok(matched)
    }

    let index = model_dir.join("model.safetensors.index.json");
    let mut matched = 0usize;
    if index.exists() {
        let wm: WeightMap = serde_json::from_str(&std::fs::read_to_string(index)?)?;
        for f in wm.weight_map.values().collect::<HashSet<_>>() {
            matched += load_weights_remapped(&mut model, &model_dir.join(f))?;
        }
    } else {
        matched = load_weights_remapped(&mut model, &model_dir.join("model.safetensors"))?;
    }
    if std::env::var("ROZUM_MLX_DEBUG").is_ok() {
        eprintln!("LOADED {matched} params (glm4_moe_lite)");
    }
    model.eval()?;
    Ok(model)
}

// ── Generate (mirror glm4/qwen3_moe; reuse qwen3 sampler + state) ───────────────────────────────
pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    sampler: SamplerOpts,
    history: Vec<u32>,
    state: GenerateState<'a>,
}
impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    pub fn new(model: &'a mut Model, cache: &'a mut Vec<Option<C>>, temp: f32, prompt_token: &'a Array) -> Self {
        Self { model, cache, sampler: SamplerOpts::with_temp(temp), history: Vec::new(), state: GenerateState::Prefill { prompt_token } }
    }
    pub fn set_sampler(&mut self, top_p: f32, top_k: i32, repeat_penalty: f32) {
        self.sampler.top_p = top_p; self.sampler.top_k = top_k; self.sampler.repeat_penalty = repeat_penalty;
    }
}
impl<C> Iterator for Generate<'_, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;
    fn next(&mut self) -> Option<Self::Item> {
        macro_rules! tri { ($e:expr) => { match $e { Ok(v) => v, Err(e) => return Some(Err(e.into())) } }; }
        let inputs = match &self.state {
            GenerateState::Prefill { prompt_token } => (*prompt_token).clone(),
            GenerateState::Decode { y } => y.index((.., NewAxis)),
        };
        let logits = tri!(self.model.forward(ModelInput { inputs: &inputs, mask: None, cache: self.cache }));
        let recent: &[u32] = if self.sampler.repeat_penalty != 1.0 { repeat_window(&self.history) } else { &[] };
        let y = tri!(sample_with(&logits.index((.., -1, ..)), &self.sampler, recent));
        if self.sampler.repeat_penalty != 1.0 {
            tri!(mlx_rs::transforms::eval([&y]));
            self.history.push(tri!(y.reshape(&[-1])).index(0).item::<u32>());
        }
        self.state = GenerateState::Decode { y: y.clone() };
        Some(Ok(y))
    }
}

#[allow(dead_code)]
fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp { 0.0 => argmax_axis!(logits, -1), _ => mlx_rs::random::categorical(&logits.multiply(array!(1.0 / temp))?, None, None, None) }
}
