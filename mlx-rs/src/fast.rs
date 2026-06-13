//! Fast implementations of commonly used multi-op functions.

use std::ffi::CStr;

use crate::error::Result;
use crate::utils::guard::Guarded;
use crate::utils::IntoOption;
use crate::{Array, Stream};
use mlx_internal_macros::{default_device, generate_macro};

/// Optimized implementation of `NN.RoPE`.
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: i32,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset,
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Optimized implementation of `NN.RoPE` with dynamic (array) offset.
///
/// This variant allows specifying the offset as an array, enabling different
/// offsets for different positions in the input.
///
/// # Params
///
/// - `array`: Input array
/// - `dimensions`: The feature dimensions to apply rope to
/// - `traditional`: If true, uses the traditional rope implementation
/// - `base`: The base used to compute angular frequency for each dimension
/// - `scale`: The scale to apply to the positions
/// - `offset`: An array of position offsets
/// - `freqs`: Optional precomputed frequencies
/// - `stream`: Stream to evaluate on
#[allow(clippy::too_many_arguments)]
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rope_dynamic_device<'a>(
    #[named] array: impl AsRef<Array>,
    #[named] dimensions: i32,
    #[named] traditional: bool,
    #[optional] base: impl Into<Option<f32>>,
    #[named] scale: f32,
    #[named] offset: impl AsRef<Array>,
    #[optional] freqs: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let base = base.into();
    let base = mlx_sys::mlx_optional_float {
        value: base.unwrap_or(0.0),
        has_value: base.is_some(),
    };
    let freqs = freqs.into();
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rope_dynamic(
            res,
            array.as_ref().as_ptr(),
            dimensions,
            traditional,
            base,
            scale,
            offset.as_ref().as_ptr(),
            freqs
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            stream.as_ref().as_ptr(),
        )
    })
}

const DEFAULT_MASK_MODE: &CStr = c"";
const CAUSAL_MASK_MODE: &CStr = c"causal";

/// Mask modes for scaled dot product attention.
#[derive(Debug)]
pub enum ScaledDotProductAttentionMask<'a> {
    /// A single mask array
    Array(&'a Array),

    /// Causal masking (no explicit mask array needed)
    Causal,
}

impl<'a> From<&'a Array> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a Array) -> Self {
        ScaledDotProductAttentionMask::Array(mask)
    }
}

impl<'a> IntoOption<ScaledDotProductAttentionMask<'a>> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

impl ScaledDotProductAttentionMask<'_> {
    fn as_mode_and_mask(&self) -> (&'static CStr, mlx_sys::mlx_array) {
        match self {
            ScaledDotProductAttentionMask::Array(mask) => (DEFAULT_MASK_MODE, mask.as_ptr()),
            ScaledDotProductAttentionMask::Causal => {
                (CAUSAL_MASK_MODE, unsafe { mlx_sys::mlx_array_new() })
            }
        }
    }
}

/// A fast implementation of multi-head attention: `O = softmax(Q @ K.T, dim=-1) @ V`
///
/// Supports [Multi-Head Attention](https://arxiv.org/abs/1706.03762), [Grouped Query Attention](https://arxiv.org/abs/2305.13245), and [Multi-Query Attention](https://arxiv.org/abs/1911.02150).
///
/// This function will dispatch to an optimized Metal kernel when the query sequence length is 1. It handles other cases with regular MLX operations.
///
/// > Note: The softmax operation is performed in float32 precision regardless of input precision (float16 or float32).
///
/// > Note: For Grouped Query Attention and Multi-Query Attention, the input arrays for `key` and `value` should not be pre-tiled to match the `query` array.
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn scaled_dot_product_attention_device<'a>(
    queries: impl AsRef<Array>,
    keys: impl AsRef<Array>,
    values: impl AsRef<Array>,
    scale: f32,
    #[optional] mask: impl IntoOption<ScaledDotProductAttentionMask<'a>>,
    #[optional] sinks: impl Into<Option<&'a Array>>,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    // rozum: for "no mask" pass a null-ctx mlx_array, not mlx_array_new() (an
    // empty but non-null array). mlx-c keys off `mask_arr.ctx`: a non-null empty
    // array is treated as a real (degenerate) mask, which the single-query GQA
    // SDPA kernel mishandles (decode garbage). A null ctx -> nullopt -> true
    // no-mask, matching Python's mx.fast.scaled_dot_product_attention(mask=None).
    let (mask_mode, mask_arr) = mask.into_option().map_or_else(
        || {
            (
                DEFAULT_MASK_MODE,
                mlx_sys::mlx_array {
                    ctx: std::ptr::null_mut(),
                },
            )
        },
        |m| m.as_mode_and_mask(),
    );

    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_scaled_dot_product_attention(
            res,
            queries.as_ref().as_ptr(),
            keys.as_ref().as_ptr(),
            values.as_ref().as_ptr(),
            scale,
            mask_mode.as_ptr(),
            mask_arr,
            sinks
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array {
                    ctx: std::ptr::null_mut(),
                }),
            stream.as_ref().as_ptr(),
        )
    })
}

/// Root Mean Square normalization (RMS norm).
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional with the same size as the last axis of `x`.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rms_norm_device(
    x: impl AsRef<Array>,
    weight: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            weight.as_ref().as_ptr(),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

/// `mx.fast.rms_norm(x, None, eps)` — RMS norm with **no** weight. The C API takes a
/// null weight (`weight.ctx == nullptr` -> `std::nullopt`), so this avoids building a
/// per-call ones weight (which shows up as a `Full`+`AsType` in the graph every call).
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn rms_norm_no_weight_device(
    x: impl AsRef<Array>,
    eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    let null_weight = unsafe { mlx_sys::mlx_array_new() };
    let result = Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_rms_norm(
            res,
            x.as_ref().as_ptr(),
            null_weight,
            eps,
            stream.as_ref().as_ptr(),
        )
    });
    unsafe {
        mlx_sys::mlx_array_free(null_weight);
    }
    result
}

/// Layer normalization.
///
/// The normalization is with respect to the last axis of the input `x`.
///
/// # Params
///
/// - x: input array
/// - weight: A multiplicative weight to scale the result by. The `weight` should be one-dimensional
///   with the same size as the last axis of `x`.  If not given no scaling will occur.
/// - bias: An additive offset to be added to the result. The `bias` should be one-dimensional
///   with the same size as the last axis of `x`.  It not given no offset will occur.
/// - eps: A small additive constant for numerical stability
/// - stream: stream or device to evaluate on
#[generate_macro(customize(root = "$crate::fast"))]
#[default_device]
pub fn layer_norm_device<'a>(
    #[named] x: impl AsRef<Array>,
    #[optional] weight: impl Into<Option<&'a Array>>,
    #[optional] bias: impl Into<Option<&'a Array>>,
    #[named] eps: f32,
    #[optional] stream: impl AsRef<Stream>,
) -> Result<Array> {
    Array::try_from_op(|res| unsafe {
        mlx_sys::mlx_fast_layer_norm(
            res,
            x.as_ref().as_ptr(),
            weight
                .into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            bias.into()
                .map(|a| a.as_ptr())
                .unwrap_or(mlx_sys::mlx_array_new()),
            eps,
            stream.as_ref().as_ptr(),
        )
    })
}

// ─── Custom Metal kernels (`mx.fast.metal_kernel`) ───────────────────────────

use crate::error::Exception;
use crate::utils::VectorArray;
use std::ffi::CString;

/// A template argument for a JIT custom Metal kernel.
#[derive(Debug, Clone)]
pub enum TemplateArg<'a> {
    /// A `dtype`-typed template parameter (`name`, value).
    Dtype(&'a str, crate::Dtype),
    /// An `int` template parameter (`name`, value).
    Int(&'a str, i32),
    /// A `bool` template parameter (`name`, value).
    Bool(&'a str, bool),
}

/// A JIT-compiled custom Metal kernel (`mx.fast.metal_kernel`). Build once with
/// [`MetalKernel::new`], then [`apply`](MetalKernel::apply) per call.
pub struct MetalKernel {
    inner: mlx_sys::mlx_fast_metal_kernel,
}

impl std::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MetalKernel")
    }
}

// The compiled kernel is immutable; MLX guards its own state internally.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

fn c_vector_string(items: &[&str]) -> Result<mlx_sys::mlx_vector_string> {
    let cstrs = items
        .iter()
        .map(|s| CString::new(*s).map_err(|_| Exception::custom("metal_kernel: NUL in name")))
        .collect::<Result<Vec<_>>>()?;
    let ptrs: Vec<*const std::os::raw::c_char> = cstrs.iter().map(|c| c.as_ptr()).collect();
    Ok(unsafe {
        mlx_sys::mlx_vector_string_new_data(
            ptrs.as_ptr() as *mut *const std::os::raw::c_char,
            ptrs.len(),
        )
    })
}

impl MetalKernel {
    /// `source` is the kernel body; MLX injects the signature from
    /// `input_names`/`output_names`. Inputs are made row-contiguous; outputs are
    /// non-atomic.
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
    ) -> Result<Self> {
        let name_c =
            CString::new(name).map_err(|_| Exception::custom("metal_kernel: NUL in name"))?;
        let source_c =
            CString::new(source).map_err(|_| Exception::custom("metal_kernel: NUL in source"))?;
        let header_c = CString::new("").unwrap();
        let inputs = c_vector_string(input_names)?;
        let outputs = c_vector_string(output_names)?;
        let inner = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name_c.as_ptr(),
                inputs,
                outputs,
                source_c.as_ptr(),
                header_c.as_ptr(),
                true,
                false,
            )
        };
        unsafe {
            mlx_sys::mlx_vector_string_free(inputs);
            mlx_sys::mlx_vector_string_free(outputs);
        }
        Ok(Self { inner })
    }

    /// Run the kernel. `inputs` order must match `input_names`; one
    /// `output_shapes`/`output_dtypes` entry per `output_names`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        inputs: &[impl AsRef<Array>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        threadgroup: (i32, i32, i32),
        output_shapes: &[&[i32]],
        output_dtypes: &[crate::Dtype],
        stream: impl AsRef<Stream>,
    ) -> Result<Vec<Array>> {
        let cfg = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };
        let build = || -> Result<()> {
            unsafe {
                for (shape, dt) in output_shapes.iter().zip(output_dtypes) {
                    mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                        cfg,
                        shape.as_ptr(),
                        shape.len(),
                        (*dt).into(),
                    );
                }
                mlx_sys::mlx_fast_metal_kernel_config_set_grid(cfg, grid.0, grid.1, grid.2);
                mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                    cfg,
                    threadgroup.0,
                    threadgroup.1,
                    threadgroup.2,
                );
                for t in template {
                    match t {
                        TemplateArg::Dtype(n, d) => {
                            let nc = CString::new(*n)
                                .map_err(|_| Exception::custom("metal_kernel: NUL in template"))?;
                            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                                cfg,
                                nc.as_ptr(),
                                (*d).into(),
                            );
                        }
                        TemplateArg::Int(n, v) => {
                            let nc = CString::new(*n)
                                .map_err(|_| Exception::custom("metal_kernel: NUL in template"))?;
                            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                                cfg,
                                nc.as_ptr(),
                                *v,
                            );
                        }
                        TemplateArg::Bool(n, v) => {
                            let nc = CString::new(*n)
                                .map_err(|_| Exception::custom("metal_kernel: NUL in template"))?;
                            mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                                cfg,
                                nc.as_ptr(),
                                *v,
                            );
                        }
                    }
                }
            }
            Ok(())
        };
        let run = || -> Result<Vec<Array>> {
            build()?;
            let in_vec = VectorArray::try_from_iter(inputs.iter())?;
            let out: VectorArray = VectorArray::try_from_op(|res| unsafe {
                mlx_sys::mlx_fast_metal_kernel_apply(
                    res,
                    self.inner,
                    in_vec.as_ptr(),
                    cfg,
                    stream.as_ref().as_ptr(),
                )
            })?;
            out.try_into_values::<Vec<Array>>()
        };
        let result = run();
        unsafe { mlx_sys::mlx_fast_metal_kernel_config_free(cfg) };
        result
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { mlx_sys::mlx_fast_metal_kernel_free(self.inner) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ops::indexing::{ArrayIndexOp, IndexOp},
        random::normal,
    };
    use float_eq::assert_float_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rope() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let result = rope(a, 8, false, 10000., 1.0, 0, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.456_253_77,
            abs <= 0.009_125_075
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            116.800_964,
            abs <= 2.336_019_3
        );
    }

    // Test adapted from Python test_fast.py/test_rope - the Python test accepts both
    // int offset and array offset, which in C/Rust are separate functions
    #[test]
    fn test_rope_dynamic() {
        crate::random::seed(71).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        // Test with array offset - should produce similar results to int offset of 3
        let offset = crate::Array::from_int(3);
        let result = rope_dynamic(&a, 8, false, 10000., 1.0, &offset, None).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);

        // Compare with regular rope using int offset=3
        let result_int_offset = rope(&a, 8, false, 10000., 1.0, 3, None).unwrap();
        assert_eq!(result_int_offset.shape(), [2, 8, 16]);

        // The results should be close
        let diff = &result - &result_int_offset;
        let max_diff = diff.abs().unwrap().max(None).unwrap().item::<f32>();
        assert!(max_diff < 1e-5, "Max difference was {}", max_diff);
    }

    #[test]
    fn test_rms_norm() {
        crate::random::seed(103).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let result = rms_norm(a, weight, 1e-5).unwrap();
        assert_eq!(result.shape(), [2, 8, 16]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.872_938_75,
            abs <= 0.017_458_774
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            223.472_32,
            abs <= 4.469_446
        );
    }

    #[test]
    pub fn test_layer_norm_affine() {
        crate::random::seed(635).unwrap();
        let a = crate::random::uniform::<_, f32>(0.0, 1.0, &[2, 8, 16], None).unwrap();
        assert_eq!(a.shape(), [2, 8, 16]);
        assert_eq!(a.dtype(), crate::Dtype::Float32);

        let weight = Array::ones::<f32>(&[16]).unwrap();
        let bias = Array::zeros::<f32>(&[16]).unwrap();
        let result = layer_norm(a, &weight, &bias, 1e-5).unwrap();
        let result = result.index((ArrayIndexOp::Ellipsis, 0));
        assert_eq!(result.shape(), [2, 8]);
        assert_eq!(result.dtype(), crate::Dtype::Float32);
        assert_float_eq!(
            result.mean(None).unwrap().item::<f32>(),
            0.290_990_38,
            abs <= 0.005_819_807_8
        );
        assert_float_eq!(
            result.sum(None).unwrap().item::<f32>(),
            4.655_846,
            abs <= 0.093_116_924
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fast_sdpa() {
        // This test just makes sure that `scaled_dot_product_attention` is callable
        // in the various cases, based on the Python test `test_fast_sdpa`.

        let Dk = 64;
        let scale = 1.0 / (Dk as f32).sqrt();
        for seq_len in [63, 129, 400] {
            for dtype in [crate::Dtype::Float32, crate::Dtype::Float16] {
                let B = 2;
                let H = 24;
                let q = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let k = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();
                let v = normal::<f32>(&[B, H, seq_len, Dk], None, None, None)
                    .unwrap()
                    .as_dtype(dtype)
                    .unwrap();

                let result = scaled_dot_product_attention(q, k, v, scale, None, None).unwrap();
                assert_eq!(result.shape(), [B, H, seq_len, Dk]);
                assert_eq!(result.dtype(), dtype);
            }
        }
    }

    // Test adapted from Python test `test_fast_sdpa.py/test_sdpa_attention_sinks`
    #[test]
    fn test_fast_sdpa_with_sinks() {
        let b = 2;
        let n_q = 8;
        let t_q = 128;
        let t_kv = 128;
        let d = 64;

        let q = normal::<f32>(&[b, n_q, t_q, d], None, None, None).unwrap();
        let k = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let v = normal::<f32>(&[b, n_q, t_kv, d], None, None, None).unwrap();
        let scale = (d as f32).powf(-0.5);

        // Test with sinks parameter
        let sinks = normal::<f32>(&[n_q], None, None, None).unwrap() * 10.0;

        let result = scaled_dot_product_attention(&q, &k, &v, scale, None, &sinks).unwrap();
        assert_eq!(result.shape(), &[b, n_q, t_q, d]);
    }

    #[test]
    fn metal_kernel_elementwise_add() {
        use crate::transforms::eval;
        let source = "
            uint i = thread_position_in_grid.x;
            out[i] = inp[i] + static_cast<float>(ADD);
        ";
        let kernel = MetalKernel::new("add_const", &["inp"], &["out"], source).unwrap();
        let inp = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let outs = kernel
            .apply(
                &[&inp],
                &[TemplateArg::Int("ADD", 10)],
                (4, 1, 1),
                (4, 1, 1),
                &[&[4][..]],
                &[crate::Dtype::Float32],
                crate::StreamOrDevice::default(),
            )
            .unwrap();
        let out = &outs[0];
        eval([out]).unwrap();
        assert_eq!(out.as_slice::<f32>(), &[11.0, 12.0, 13.0, 14.0]);
    }
}
