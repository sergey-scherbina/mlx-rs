// Model architectures, split into an always-compiled core and opt-out families.
//
// The features are additive and ON by default (`default = ["all-models"]`), so every existing
// consumer keeps building unchanged; a lean consumer opts out with `default-features = false`
// plus just the families it loads.
//
// The CORE below is deliberately NOT gated. `qwen3` and `qwen3_5` reference each OTHER, and
// cargo features must form a DAG — so they cannot be separated into independent features. The
// core is also what everything else is built on: `qwen3_5` needs `gated_delta` (the shared
// GatedDeltaNet hybrid layer) and `qwen3_5_vision`, and lib.rs itself impls `ModelInput` for
// `qwen3::ModelInput`. Gating any of it would mean cfg-ing that impl too, for no real win —
// the core is present in every configuration rozum ships.
pub mod gated_delta;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_5_vision;

// Opt-out families. Each is a leaf: nothing in this crate references it except the line below,
// so compiling it out costs nothing else — with one edge, noted on `qwen3_5_moe`.
#[cfg(feature = "deepseek-v2")]
pub mod deepseek_v2;
#[cfg(feature = "gemma3")]
pub mod gemma3;
#[cfg(feature = "glm4")]
pub mod glm4;
#[cfg(feature = "glm4-moe-lite")]
pub mod glm4_moe_lite;
#[cfg(feature = "gpt-oss")]
pub mod gpt_oss;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "qwen2")]
pub mod qwen2;
#[cfg(feature = "qwen3-moe")]
pub mod qwen3_moe;
// `qwen3_5_moe` builds on `qwen3_moe`, so its feature enables that one (see Cargo.toml).
#[cfg(feature = "qwen3-5-moe")]
pub mod qwen3_5_moe;
