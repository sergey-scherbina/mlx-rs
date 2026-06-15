//! Standalone Metal kernel sources (`extract-metal-kernels`, L4).
//!
//! Hardware-only MSL — just the kernel math, factored out of the model leaves. The engine
//! *binding* (input/output buffers, grid/threadgroup dispatch, eval control) stays with the
//! caller (e.g. [`crate::models::gated_delta`]). Keeping the `.metal` source here lets a future
//! Metal engine (a candle-metal path, mistralrs-metal) bind the same kernel instead of
//! re-deriving the math.

/// The GatedDeltaNet fused delta-rule scan kernel body — entry `gated_delta_step`, the body of
/// an MLX `fast.metal_kernel`. One GPU dispatch runs the whole T-step recurrence. Requires
/// `Dk % 32 == 0`. See `gated_delta_step.metal` for the input/output contract and
/// [`crate::models::gated_delta`] for the binding.
pub const GATED_DELTA_SOURCE: &str = include_str!("gated_delta_step.metal");
