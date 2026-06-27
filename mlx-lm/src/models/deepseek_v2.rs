//! DeepSeek-V2 (`model_type: "deepseek_v2"`) — **MLA** latent attention + DeepSeek MoE.
//! Ported from Python `mlx_lm.models.deepseek_v2`. Target: DeepSeek-Coder-V2-Lite (16B-A2.4B → very
//! low peak RAM). This file is ALSO the shared MLA home for GLM-4.7-Flash (`glm4_moe_lite`), which
//! adds `embed_q`/`unembed_out` + V3 routing on top. Spec: `rozum:docs/specs/mlx-mla-attention.md`.
//!
//! PORT STATUS: WIP — MLA attention + ModelArgs transcribed from the reference; MoE/model/load and
//! YaRN-mscale exactness are TODO. **NOT registered in `models/mod.rs`** (fork keeps building);
//! compile + byte-parity vs Python are slot-gated (load a model). Do not wire into rozum until green.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::{concatenate_axis, indexing::IndexOp},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;

use crate::{
    cache::KeyValueCache,
    utils::rope::{initialize_rope, FloatOrString, RopeVariant},
};
use std::collections::HashMap;

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
    // ── MLA latent dims (the heart of the port) ─────────────────────────────────────────────
    pub q_lora_rank: i32,         // 1536 (DSv2) / 768 (GLM-Flash); 0 ⇒ no q low-rank (plain q_proj)
    pub kv_lora_rank: i32,        // 512
    pub qk_nope_head_dim: i32,    // 128 (DSv2) / 192 (GLM-Flash)
    pub qk_rope_head_dim: i32,    // 64
    pub v_head_dim: i32,          // 128 (DSv2) / 256 (GLM-Flash)
    #[serde(default = "d_true")]
    pub attention_bias: bool,
    // ── DeepSeek MoE (handled in the MoE block, reusing qwen3_moe::SwitchGlu) ───────────────
    #[serde(default)]
    pub n_routed_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub n_shared_experts: i32,
    #[serde(default)]
    pub first_k_dense_layers: i32, // a.k.a. `first_k_dense_replace` in some configs
    #[serde(default = "d_one_f32")]
    pub routed_scaling_factor: f32,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>, // YaRN: factor, mscale, mscale_all_dim
    pub quantization: Option<crate::models::qwen3::QuantizationConfig>,
}
fn d_max_pos() -> i32 { 4096 }
fn d_true() -> bool { true }
fn d_one_f32() -> f32 { 1.0 }

/// MLA: compress Q (q_a→norm→q_b) and KV (kv_a→norm→kv_b) through low-rank latents; carry a small
/// decoupled RoPE part (`qk_rope_head_dim`) shared MQA-style across heads. KV cache stores the
/// compressed keys+values → tiny cache → low peak RAM. Forward mirrors the reference 1:1.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct MlaAttention {
    pub num_heads: i32,
    pub qk_nope_head_dim: i32,
    pub qk_rope_head_dim: i32,
    pub q_head_dim: i32,
    pub v_head_dim: i32,
    pub kv_lora_rank: i32,
    pub scale: f32,

    #[quantizable] #[param] pub q_a_proj: MaybeQuantized<nn::Linear>,
    #[param] pub q_a_layernorm: nn::RmsNorm,
    #[quantizable] #[param] pub q_b_proj: MaybeQuantized<nn::Linear>,
    #[quantizable] #[param] pub kv_a_proj_with_mqa: MaybeQuantized<nn::Linear>,
    #[param] pub kv_a_layernorm: nn::RmsNorm,
    #[quantizable] #[param] pub kv_b_proj: MaybeQuantized<nn::Linear>,
    #[quantizable] #[param] pub o_proj: MaybeQuantized<nn::Linear>,
    #[param] pub rope: RopeVariant,
}

impl MlaAttention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let d = args.hidden_size;
        let h = args.num_attention_heads;
        let q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim;
        // scale = q_head_dim^-0.5  (× YaRN mscale² when rope_scaling.mscale_all_dim — TODO: apply).
        let scale = (q_head_dim as f32).powf(-0.5);
        let lin = |i, o, b| nn::LinearBuilder::new(i, o).bias(b).build();
        let rms = |dim| nn::RmsNormBuilder::new(dim).eps(args.rms_norm_eps).build();
        // q low-rank (q_lora_rank>0 path; the q_lora_rank==0 plain-q_proj path is a TODO variant).
        let q_a_proj = lin(d, args.q_lora_rank, args.attention_bias)?;
        let q_b_proj = lin(args.q_lora_rank, h * q_head_dim, false)?;
        let kv_a_proj_with_mqa = lin(d, args.kv_lora_rank + args.qk_rope_head_dim, args.attention_bias)?;
        let kv_b_proj = lin(args.kv_lora_rank, h * (args.qk_nope_head_dim + args.v_head_dim), false)?;
        let o_proj = lin(h * args.v_head_dim, d, false)?;
        // Decoupled RoPE acts only on the qk_rope_head_dim part. YaRN scaling via rope_scaling.
        // TODO(parity): the reference uses a YaRN mscale on the embedding too — verify initialize_rope
        // reproduces DeepseekV2YarnRotaryEmbedding, else add the YaRN variant.
        let rope = initialize_rope(
            args.qk_rope_head_dim,
            args.rope_theta,
            false, // DeepSeek uses non-traditional rope on the decoupled part
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;
        Ok(Self {
            num_heads: h,
            qk_nope_head_dim: args.qk_nope_head_dim,
            qk_rope_head_dim: args.qk_rope_head_dim,
            q_head_dim,
            v_head_dim: args.v_head_dim,
            kv_lora_rank: args.kv_lora_rank,
            scale,
            q_a_proj: MaybeQuantized::Original(q_a_proj),
            q_a_layernorm: rms(args.q_lora_rank)?,
            q_b_proj: MaybeQuantized::Original(q_b_proj),
            kv_a_proj_with_mqa: MaybeQuantized::Original(kv_a_proj_with_mqa),
            kv_a_layernorm: rms(args.kv_lora_rank)?,
            kv_b_proj: MaybeQuantized::Original(kv_b_proj),
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
        let (nope, rope_d, vd) = (self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim);

        // q = q_b(q_a_norm(q_a(x)))  →  [B,L,H,q_head_dim] → [B,H,L,q_head_dim]
        let q = self.q_a_proj.forward(x)?;
        let q = self.q_a_layernorm.forward(&q)?;
        let q = self.q_b_proj.forward(&q)?;
        let q = q.reshape(&[B, L, self.num_heads, self.q_head_dim])?.transpose_axes(&[0, 2, 1, 3])?;
        let q_nope = q.index((.., .., .., 0..nope));
        let mut q_pe = q.index((.., .., .., nope..(nope + rope_d)));

        // compressed_kv | k_pe  =  kv_a_proj_with_mqa(x)
        let ckv = self.kv_a_proj_with_mqa.forward(x)?;
        let compressed_kv = ckv.index((.., .., 0..self.kv_lora_rank));
        let mut k_pe = ckv
            .index((.., .., self.kv_lora_rank..(self.kv_lora_rank + rope_d)))
            .reshape(&[B, L, 1, rope_d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,1,L,rope_d]  (MQA: one head)

        // kv = kv_b(kv_a_norm(compressed_kv)) → [B,L,H,nope+vd] → [B,H,L,nope+vd] → split
        let kv = self.kv_a_layernorm.forward(&compressed_kv)?;
        let kv = self.kv_b_proj.forward(&kv)?;
        let kv = kv.reshape(&[B, L, self.num_heads, nope + vd])?.transpose_axes(&[0, 2, 1, 3])?;
        let k_nope = kv.index((.., .., .., 0..nope));
        let values = kv.index((.., .., .., nope..(nope + vd)));

        // RoPE on the decoupled parts; broadcast k_pe to all heads. (offset from cache.)
        let offset = cache.as_ref().map(|c| c.offset()).unwrap_or(0);
        q_pe = self.rope.forward(nn::RopeInputBuilder::new(&q_pe).offset(offset).build()?)?;
        k_pe = self.rope.forward(nn::RopeInputBuilder::new(&k_pe).offset(offset).build()?)?;
        let k_pe = mlx_rs::ops::repeat_axis(&k_pe, self.num_heads, 1)?; // [B,H,L,rope_d]

        let queries = concatenate_axis(&[q_nope, q_pe], -1)?;
        let keys_full = concatenate_axis(&[k_nope, k_pe], -1)?;
        let (keys, values) = match cache.as_mut() {
            Some(c) => c.update_and_fetch(keys_full, values)?,
            None => (keys_full, values),
        };

        let out = crate::utils::scaled_dot_product_attention(
            queries, keys, values, cache, self.scale, mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;
        self.o_proj.forward(&out)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_a_proj.training_mode(mode);
        self.q_a_layernorm.training_mode(mode);
        self.q_b_proj.training_mode(mode);
        self.kv_a_proj_with_mqa.training_mode(mode);
        self.kv_a_layernorm.training_mode(mode);
        self.kv_b_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

// ── TODO (resume-cold; see spec) ────────────────────────────────────────────────────────────────
// - YaRN rope exactness: replicate DeepseekV2YarnRotaryEmbedding (mscale on x; interpolated freqs).
//   Verify initialize_rope(rope_scaling) matches; else add a YaRN RopeVariant. Apply mscale² to
//   `scale` when rope_scaling.mscale_all_dim is set.
// - DecoderLayer: input_layernorm → MlaAttention → residual; post_attention_layernorm → FFN →
//   residual. FFN = dense DeepseekV2MLP for layer < first_k_dense_layers, else MoE.
// - MoE block: reuse qwen3_moe::{QSwitchLinear, SwitchGlu}; router = softmax greedy top-k (DSv2);
//   + shared expert (n_shared_experts) added to routed output; × routed_scaling_factor.
//   (The V3 sigmoid + e_score_correction_bias variant is for glm4_moe_lite, ref deepseek_v3.py.)
// - Model + load: embed/final-norm/lm_head; AFQ remap (experts pre-stacked `mlp.switch_mlp.*`);
//   register `"deepseek_v2"` in models/mod.rs; add Generate (mirror qwen3_moe::Generate).
// - rozum wiring: LoadedModel::DeepseekV2 + dispatch + bump fork rev (after compile+parity).
