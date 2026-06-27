//! GLM-4 **MoE** (Zhipu/Z.ai) — `model_type: "glm4_moe"` (GLM-4.5-Air, GLM-4.6) and
//! `"glm4_moe_lite"` (GLM-4.7-Flash). PORT SCAFFOLD — see `rozum:docs/specs/glm4-moe-native.md`.
//!
//! PORT STATUS: PARKED scaffold, **not registered in `models/mod.rs`** (fork keeps building).
//!
//! ⚠️ CHECKPOINT FINDING (2026-06-27): the two GLM-MoE families differ by ATTENTION:
//!   - `glm4_moe` (GLM-4.5-Air / GLM-4.6) = standard **GQA** → glm4.rs attention reuse holds, BUT
//!     these are too big for the 36 GiB host.
//!   - `glm4_moe_lite` (**GLM-4.7-Flash**, the only one that fits) = **MLA** (DeepSeek-V2 latent
//!     attention: `q_a_proj`/`q_a_layernorm`/`q_b_proj`, `kv_a_proj_with_mqa`/`kv_a_layernorm`;
//!     q_lora 768 / kv_lora 512 / qk_nope 192 / qk_rope 64 / v_head_dim 256). glm4.rs attention does
//!     NOT apply. → HIGH effort, shares MLA with the deepseek_v2 port (do the MLA kernel once).
//!   Also: dense layer (first_k_dense_layers=1) FFN is SPLIT `mlp.{gate,up,down}_proj`, not glm4's
//!   fused `gate_up_proj`. MoE: `mlp.switch_mlp.*` + `mlp.shared_experts.*` +
//!   `mlp.gate.e_score_correction_bias`. See rozum:docs/specs/glm4-moe-native.md.
//!
//! The dense/GQA reuse notes below apply ONLY to `glm4_moe` (unrunnable here). For `glm4_moe_lite`
//! the attention must be MLA — finish only after the shared MLA kernel exists.
//!
//! = `glm4.rs` (GQA path only: partial RoPE `Attention`, 4-norm sandwich, embedding/final-norm,
//!   AFQ-remap `load_*`) + the dense `Mlp` REPLACED, per layer, by a MoE FFN.
//!
//! The MoE block is NOT a plain copy of `qwen3_moe::SparseMoeBlock`. GLM-4 MoE routes
//! DeepSeek-V3-style; the differences vs Qwen3-MoE (flat softmax top-k) are the real work:
//!   1. **sigmoid** gate scoring (not softmax)
//!   2. **grouped top-k** (`n_group` / `topk_group`) — limit selection to top groups
//!   3. **`e_score_correction_bias`** added for SELECTION only (not to combine weights)
//!   4. **`routed_scaling_factor`** scales the combined routed output
//!   5. **shared expert(s)** (`n_shared_experts`) — dense MLP output ADDED to routed output
//!      (reuse `qwen3_5_moe.rs` if it already implements a shared expert)
//!   6. **`first_k_dense_layers`** — first k layers are dense `Mlp`, the rest MoE
//!   7. **MTP** (`num_nextn_predict_layers`) — parse then DROP for greedy inference

use serde::Deserialize;

// NOTE: bring the same imports glm4.rs + qwen3_moe.rs use (Array, Exception, nn, MaybeQuantized,
// QuantizationConfig, gather_qmm, argpartition_axis, take_along_axis, expand_dims_axes, sigmoid,
// Module, ModuleParameters, Quantizable, RmsNorm, …). Reuse `qwen3_moe::{QSwitchLinear, SwitchGlu}`
// VERBATIM (AFQ experts via gather_qmm + sorted-prefill path) — only the router changes.

/// GLM-4 MoE config. Dense GLM-4 fields (hidden_size, num_hidden_layers, num_attention_heads,
/// num_key_value_heads, head_dim, partial_rotary_factor, rms_norm_eps, rope_theta, rope_scaling,
/// vocab_size, intermediate_size, quantization) are identical to `glm4::ModelArgs` — copy them.
/// The MoE-specific additions:
#[derive(Debug, Clone, Deserialize)]
pub struct MoeArgs {
    /// Routed-expert count (HF: `n_routed_experts`).
    #[serde(alias = "n_routed_experts")]
    pub num_experts: i32,
    pub num_experts_per_tok: i32,
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub n_shared_experts: i32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    /// Grouped routing (DeepSeek-style). `n_group <= 1` ⇒ plain top-k (the `glm4_moe_lite` path).
    #[serde(default = "one")]
    pub n_group: i32,
    #[serde(default = "one")]
    pub topk_group: i32,
    #[serde(default = "one_f32")]
    pub routed_scaling_factor: f32,
    /// First k decoder layers use the dense `Mlp`; the rest use the MoE block.
    #[serde(default)]
    pub first_k_dense_layers: i32,
    /// Router quant bit-width when it differs from the rest (e.g. 8-bit gate).
    #[serde(default)]
    pub router_bits: i32,
    /// Multi-token-prediction layers — PARSED THEN IGNORED (greedy inference drops them).
    #[serde(default)]
    pub num_nextn_predict_layers: i32,
}
fn one() -> i32 { 1 }
fn one_f32() -> f32 { 1.0 }

// ── GLM MoE router ───────────────────────────────────────────────────────────────────────────
//
// Adapt `qwen3_moe::SparseMoeBlock`. Reference (Qwen3-MoE forward, to TRANSFORM not copy):
//     let gates = softmax_axis(&self.gate.forward(x)?, -1, true)?;     // GLM: sigmoid, no softmax
//     let inds  = argpartition_axis(&gates, -k, -1)?...                 // GLM: + correction-bias,
//     let scores = take_along_axis(&gates, &inds, -1)?;                 //      + grouped top-k
//     let y = self.switch_mlp.forward(x, &inds)?;                       // (SwitchGlu — reuse as-is)
//     y * scores  → sum over experts                                    // GLM: * routed_scaling,
//                                                                       //      + shared_expert(x)
//
// GLM forward (target):
//   logits   = gate(x)
//   scores   = sigmoid(logits)
//   sel      = scores + e_score_correction_bias                        // selection score only
//   grouped top-k over sel: keep only experts in the top `topk_group` of `n_group` groups,
//                           then top-k experts within the kept groups
//   weights  = gather(scores, idx); if norm_topk_prob { weights /= sum(weights) }
//   weights *= routed_scaling_factor
//   y        = Σ weights · SwitchGlu(x, idx)  +  shared_expert(x)
//
// `e_score_correction_bias` is a learned param `mlp.gate.e_score_correction_bias` (load it).

/// GLM-4 MoE block. (struct/fields/new/forward to be filled — see the router math above.)
pub struct GlmSparseMoeBlock {
    // gate: MaybeQuantized<nn::Linear>,            // router, may be `router_bits`-quantized
    // e_score_correction_bias: Array,             // [num_experts]
    // switch_mlp: qwen3_moe::SwitchGlu,           // routed experts (reuse)
    // shared_expert: Option<glm4::Mlp>,           // n_shared_experts > 0
    // top_k / num_experts / n_group / topk_group / norm_topk_prob / routed_scaling_factor
}

// ── Decoder layer: dense for layer < first_k_dense_layers, else MoE ───────────────────────────
//
// Copy `glm4::DecoderLayer` VERBATIM (Attention + the 4-norm sandwich); only swap the FFN field:
//   ffn = if layer_idx < first_k_dense_layers { Ffn::Dense(glm4::Mlp) } else { Ffn::Moe(GlmSparseMoeBlock) }
// The residual flow is unchanged:
//   x = x + post_self_attn_layernorm(attn(input_layernorm(x)))
//   x = x + post_mlp_layernorm(ffn(post_attention_layernorm(x)))

// ── Model + load ─────────────────────────────────────────────────────────────────────────────
//
// Copy `glm4::Model` + `glm4::load_glm4_model` (rename `*_glm4_moe_*`). Build only
// `num_hidden_layers` decoder layers (ignore `num_nextn_predict_layers` = MTP). Keep the AFQ
// `load_weights_remapped` (`.weight`→`.inner.weight`, `.bias`→`.inner.bias` when `.scales` exists);
// experts are AFQ too, so the remap already covers `switch_mlp.*_proj`. Then in rozum:
//   - mlx_native_backend.rs: `LoadedModel::Glm4Moe`, dispatch
//     `"glm4_moe" | "glm4_moe_lite" => glm4_moe::load_glm4_moe_model(dir)`, a `Generate` arm.
//   - bump mlx-rs/mlx-lm rev in crates/rozum-mlx/Cargo.toml.
//
// pub fn load_glm4_moe_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> { todo!() }
