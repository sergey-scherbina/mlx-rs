use std::collections::HashMap;

use mlx_macros::ModuleParameters;
use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::Module,
    nn,
    ops::{arange, maximum, minimum, which},
    Array,
};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq)]
pub enum FloatOrStr<'a> {
    Float(f32),
    Str(&'a str),
}

// TODO: check if additional serde attributes are needed
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum FloatOrString {
    Float(f32),
    String(String),
}

impl FloatOrString {
    pub fn borrowed(&self) -> FloatOrStr<'_> {
        match self {
            FloatOrString::Float(f) => FloatOrStr::Float(*f),
            FloatOrString::String(s) => FloatOrStr::Str(s),
        }
    }
}

/// Get a numeric float value from a scaling config by key.
///
/// Note: str variants in the config are not always floats — values like "default" or "linear"
/// are also valid for non-numeric fields. This function should only be called for keys that
/// are expected to hold numeric values.
fn get_numeric_from_config(
    config: &HashMap<String, FloatOrString>,
    key: &str,
) -> Result<f32, Exception> {
    match config
        .get(key)
        .map(FloatOrString::borrowed)
        .ok_or_else(|| {
            Exception::custom(format!(r#"key "{key}" is not found in scaling config"#))
        })? {
        FloatOrStr::Float(f) => Ok(f),
        FloatOrStr::Str(s) => s
            .parse::<f32>()
            .map_err(|_| Exception::custom(format!(r#"key "{key}" is not a valid number"#))),
    }
}

/// Llama3-style RoPE with frequency scaling.
///
/// Applies piecewise frequency scaling based on wavelength cutoffs derived from
/// `low_freq_factor`, `high_freq_factor`, `factor`, and `original_max_position_embeddings`.
// TODO: support derive ModuleParameters for structs with non-param Array fields
#[derive(Debug, Clone, ModuleParameters)]
pub struct Llama3Rope {
    pub dimensions: i32,
    pub traditional: bool,
    pub scale: f32,
    /// Pre-computed scaled frequencies. Not a module parameter.
    pub freqs: Array,
}

impl Llama3Rope {
    pub fn new(
        dims: i32,
        traditional: bool,
        original_max_position_embeddings: i32,
        base: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
    ) -> Result<Self, Exception> {
        let half_dims = dims / 2;

        // Compute freqs using MLX ops, matching Python:
        //   freqs = base ** (mx.arange(0, dims, 2) / dims)
        // which equals base^(2i/dims) for i in 0..half_dims
        let indices = arange::<_, f32>(None, half_dims, None)?;
        let exponents = indices.multiply(Array::from_f32(2.0 / dims as f32))?;
        let freqs = Array::from_f32(base).power(&exponents)?;

        let old_context_len = original_max_position_embeddings as f32;
        let low_freq_wavelen = old_context_len / low_freq_factor;
        let high_freq_wavelen = old_context_len / high_freq_factor;

        // wavelens = 2 * pi * freqs
        // Apply piecewise scaling matching Python exactly:
        //   freqs = where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        //   is_medium = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        //   smooth_factors = (old_context_len / wavelens - low_freq_factor) / (high - low)
        //   smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        //   freqs = where(is_medium, smooth_freqs, freqs)
        let two_pi = Array::from_f32(2.0 * std::f32::consts::PI);
        let wavelens = freqs.multiply(&two_pi)?;

        // First pass: scale low frequencies (long wavelengths) by factor
        let is_low = wavelens.gt(Array::from_f32(low_freq_wavelen))?;
        let freqs = which(&is_low, &freqs.multiply(Array::from_f32(factor))?, &freqs)?;

        // Second pass: smooth interpolation for medium frequencies
        let is_medium = wavelens
            .gt(Array::from_f32(high_freq_wavelen))?
            .logical_and(&wavelens.lt(Array::from_f32(low_freq_wavelen))?)?;

        let smooth_factors = wavelens
            .reciprocal()?
            .multiply(Array::from_f32(old_context_len))?
            .subtract(Array::from_f32(low_freq_factor))?
            .divide(Array::from_f32(high_freq_factor - low_freq_factor))?;

        // smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        let one_minus_smooth = Array::from_f32(1.0).subtract(&smooth_factors)?;
        let denom = one_minus_smooth
            .divide(Array::from_f32(factor))?
            .add(&smooth_factors)?;
        let smooth_freqs = freqs.divide(&denom)?;

        let freqs = which(&is_medium, &smooth_freqs, &freqs)?;

        Ok(Self {
            dimensions: dims,
            traditional,
            scale: 1.0,
            freqs,
        })
    }
}

impl<'a, Input> Module<Input> for Llama3Rope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        // rozum: apply rope on the 4D [B, n_heads, L, head_dim] directly. The old
        // reshape to 3D [B*n_heads, L, head_dim] tripped an MLX fast-rope bug for
        // the single-position (decode, L=1) case where only the first batch row
        // got rotated (later heads were left un-rotated -> garbage decode).
        mlx_rs::fast::rope(
            x,
            self.dimensions,
            self.traditional,
            None::<f32>,
            self.scale,
            offset,
            &self.freqs,
        )
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// YaRN RoPE: per-dimension frequency interpolation between extrapolation
/// (high-frequency dims, left as-is) and interpolation (low-frequency dims,
/// divided by `factor`) via a linear ramp, plus an attention `mscale` applied to
/// q/k before rotation. A faithful port of Python `mlx_lm`'s `YarnRoPE`. Used by
/// gpt-oss (`rope_type: "yarn"`).
#[derive(Debug, Clone, ModuleParameters)]
pub struct YarnRope {
    pub dimensions: i32,
    pub traditional: bool,
    pub scale: f32,
    pub mscale: f32,
    /// Pre-computed interpolated per-dim periods. Not a module parameter.
    pub freqs: Array,
}

impl YarnRope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: i32,
        traditional: bool,
        base: f32,
        scaling_factor: f32,
        original_max_position_embeddings: i32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<Self, Exception> {
        let half_dims = dims / 2;
        let orig_max = original_max_position_embeddings as f32;

        // freq_extra = base ** (arange(0, dims, 2) / dims) — the un-scaled periods.
        let indices = arange::<_, f32>(None, half_dims, None)?;
        let exponents = indices.multiply(Array::from_f32(2.0 / dims as f32))?;
        let freq_extra = Array::from_f32(base).power(&exponents)?;
        // freq_inter = scaling_factor * freq_extra — the fully-interpolated periods.
        let freq_inter = freq_extra.multiply(Array::from_f32(scaling_factor))?;

        // Correction range: which dims extrapolate (keep) vs interpolate (scale).
        let find_dim = |num_rotations: f32| -> f32 {
            (dims as f32 * (orig_max / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
                / (2.0 * base.ln())
        };
        let low = find_dim(beta_fast).floor().max(0.0);
        let mut high = find_dim(beta_slow).ceil().min((dims - 1) as f32);
        if (high - low).abs() < f32::EPSILON {
            high += 0.001; // prevent singularity
        }

        // freq_mask = 1 - clip((arange(half) - low) / (high - low), 0, 1)
        let ramp = arange::<_, f32>(None, half_dims, None)?
            .subtract(Array::from_f32(low))?
            .divide(Array::from_f32(high - low))?;
        let ramp = minimum(maximum(ramp, Array::from_f32(0.0))?, Array::from_f32(1.0))?;
        let freq_mask = Array::from_f32(1.0).subtract(&ramp)?;

        // _freqs = (freq_inter*freq_extra) / (freq_inter*freq_mask + freq_extra*(1-freq_mask))
        let numer = freq_inter.multiply(&freq_extra)?;
        let one_minus_mask = Array::from_f32(1.0).subtract(&freq_mask)?;
        let denom = freq_inter
            .multiply(&freq_mask)?
            .add(&freq_extra.multiply(&one_minus_mask)?)?;
        let freqs = numer.divide(&denom)?;

        // mscale = get_mscale(scale, 1) / get_mscale(scale, 0); 1.0 when scale <= 1.
        let get_mscale = |scale: f32, m: f32| -> f32 {
            if scale <= 1.0 {
                1.0
            } else {
                0.1 * m * scale.ln() + 1.0
            }
        };
        let mscale = get_mscale(scaling_factor, 1.0) / get_mscale(scaling_factor, 0.0);

        Ok(Self {
            dimensions: dims,
            traditional,
            scale: 1.0,
            mscale,
            freqs,
        })
    }

    /// Python scales `x[..., :dims]` by `mscale` before rope; gpt-oss has
    /// `dims == head_dim` (the whole last axis), so scale the array directly.
    fn pre_scale(&self, x: &Array) -> Result<Array, Exception> {
        if self.mscale != 1.0 {
            x.multiply(Array::from_f32(self.mscale))
        } else {
            Ok(x.clone())
        }
    }
}

impl<'a, Input> Module<Input> for YarnRope
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
        let nn::RopeInput { x, offset } = input.into();
        let x = self.pre_scale(x)?;
        mlx_rs::fast::rope(
            &x,
            self.dimensions,
            self.traditional,
            None::<f32>,
            self.scale,
            offset,
            &self.freqs,
        )
    }

    fn training_mode(&mut self, _mode: bool) {}
}

/// Enum wrapping different RoPE variants so that `initialize_rope` can return
/// a standard, Llama3, or YaRN RoPE.
#[derive(Debug, Clone)]
pub enum RopeVariant {
    Default(nn::Rope),
    Llama3(Llama3Rope),
    Yarn(YarnRope),
}

// TODO: support derive ModuleParameters for enum
impl mlx_rs::module::ModuleParameters for RopeVariant {
    fn num_parameters(&self) -> usize {
        match self {
            RopeVariant::Default(rope) => rope.num_parameters(),
            RopeVariant::Llama3(rope) => rope.num_parameters(),
            RopeVariant::Yarn(rope) => rope.num_parameters(),
        }
    }

    fn freeze_parameters(&mut self, _recursive: bool) {
        match self {
            RopeVariant::Default(rope) => rope.freeze_parameters(_recursive),
            RopeVariant::Llama3(rope) => rope.freeze_parameters(_recursive),
            RopeVariant::Yarn(rope) => rope.freeze_parameters(_recursive),
        }
    }

    fn unfreeze_parameters(&mut self, _recursive: bool) {
        match self {
            RopeVariant::Default(rope) => rope.unfreeze_parameters(_recursive),
            RopeVariant::Llama3(rope) => rope.unfreeze_parameters(_recursive),
            RopeVariant::Yarn(rope) => rope.unfreeze_parameters(_recursive),
        }
    }

    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            RopeVariant::Default(rope) => rope.parameters(),
            RopeVariant::Llama3(rope) => rope.parameters(),
            RopeVariant::Yarn(rope) => rope.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        match self {
            RopeVariant::Default(rope) => rope.parameters_mut(),
            RopeVariant::Llama3(rope) => rope.parameters_mut(),
            RopeVariant::Yarn(rope) => rope.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        match self {
            RopeVariant::Default(rope) => rope.trainable_parameters(),
            RopeVariant::Llama3(rope) => rope.trainable_parameters(),
            RopeVariant::Yarn(rope) => rope.trainable_parameters(),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            RopeVariant::Default(rope) => rope.all_frozen(),
            RopeVariant::Llama3(rope) => rope.all_frozen(),
            RopeVariant::Yarn(rope) => rope.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            RopeVariant::Default(rope) => rope.any_frozen(),
            RopeVariant::Llama3(rope) => rope.any_frozen(),
            RopeVariant::Yarn(rope) => rope.any_frozen(),
        }
    }
}

impl<'a, Input> Module<Input> for RopeVariant
where
    Input: Into<nn::RopeInput<'a>>,
{
    type Error = Exception;
    type Output = Array;

    fn forward(&mut self, input: Input) -> Result<Self::Output, Self::Error> {
        match self {
            RopeVariant::Default(rope) => rope.forward(input),
            RopeVariant::Llama3(rope) => rope.forward(input),
            RopeVariant::Yarn(rope) => rope.forward(input),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            RopeVariant::Default(rope) => {
                <nn::Rope as Module<nn::RopeInput>>::training_mode(rope, mode)
            }
            RopeVariant::Llama3(rope) => {
                <Llama3Rope as Module<nn::RopeInput>>::training_mode(rope, mode)
            }
            RopeVariant::Yarn(rope) => {
                <YarnRope as Module<nn::RopeInput>>::training_mode(rope, mode)
            }
        }
    }
}

impl RopeVariant {
    /// Apply RoPE with a PER-ROW position offset (`offsets` is a `[B]` array of start
    /// positions) instead of one scalar offset — for **ragged batched decode**, where
    /// each batch row is at a different sequence length so each must be rotated at its
    /// own position. Mirrors [`Module::forward`] but via `rope_dynamic` (the offset
    /// broadcasts across the batch). Byte-exact vs the scalar path when all offsets
    /// are equal.
    pub fn forward_dynamic(&self, x: &Array, offsets: &Array) -> Result<Array, Exception> {
        match self {
            RopeVariant::Default(r) => mlx_rs::fast::rope_dynamic(
                x,
                r.dimensions,
                r.traditional,
                Some(r.base),
                r.scale,
                offsets,
                None,
            ),
            RopeVariant::Llama3(r) => mlx_rs::fast::rope_dynamic(
                x,
                r.dimensions,
                r.traditional,
                None::<f32>,
                r.scale,
                offsets,
                Some(&r.freqs),
            ),
            RopeVariant::Yarn(r) => {
                let x = r.pre_scale(x)?;
                mlx_rs::fast::rope_dynamic(
                    &x,
                    r.dimensions,
                    r.traditional,
                    None::<f32>,
                    r.scale,
                    offsets,
                    Some(&r.freqs),
                )
            }
        }
    }
}

pub fn initialize_rope(
    dims: i32,
    base: f32, // rope_theta
    traditional: bool,
    scaling_config: &Option<HashMap<String, FloatOrString>>,
    _max_position_embeddings: i32,
) -> Result<RopeVariant, Exception> {
    let rope_type = scaling_config
        .as_ref()
        .and_then(|config| {
            config
                .get("type")
                .or_else(|| config.get("rope_type"))
                .map(FloatOrString::borrowed)
        })
        .unwrap_or(FloatOrStr::Str("default"));

    if rope_type == FloatOrStr::Str("default") || rope_type == FloatOrStr::Str("linear") {
        let scale = if rope_type == FloatOrStr::Str("linear") {
            let den = get_numeric_from_config(scaling_config.as_ref().unwrap(), "factor")?;
            1.0 / den
        } else {
            1.0
        };

        let rope = nn::RopeBuilder::new(dims)
            .traditional(traditional)
            .base(base)
            .scale(scale)
            .build()
            .expect("Infallible");
        return Ok(RopeVariant::Default(rope));
    } else if rope_type == FloatOrStr::Str("llama3") {
        let config = scaling_config
            .as_ref()
            .ok_or_else(|| Exception::custom("scaling_config is required for llama3 RoPE"))?;

        let factor = get_numeric_from_config(config, "factor")?;
        let low_freq_factor = get_numeric_from_config(config, "low_freq_factor")?;
        let high_freq_factor = get_numeric_from_config(config, "high_freq_factor")?;
        let original_max_position_embeddings =
            get_numeric_from_config(config, "original_max_position_embeddings")? as i32;

        let rope = Llama3Rope::new(
            dims,
            traditional,
            original_max_position_embeddings,
            base,
            factor,
            low_freq_factor,
            high_freq_factor,
        )?;
        return Ok(RopeVariant::Llama3(rope));
    } else if rope_type == FloatOrStr::Str("yarn") {
        let config = scaling_config
            .as_ref()
            .ok_or_else(|| Exception::custom("scaling_config is required for yarn RoPE"))?;

        let factor = get_numeric_from_config(config, "factor")?;
        let original_max_position_embeddings =
            get_numeric_from_config(config, "original_max_position_embeddings")? as i32;
        // beta_fast/beta_slow default to the YaRN paper values when absent.
        let beta_fast = get_numeric_from_config(config, "beta_fast").unwrap_or(32.0);
        let beta_slow = get_numeric_from_config(config, "beta_slow").unwrap_or(1.0);

        let rope = YarnRope::new(
            dims,
            traditional,
            base,
            factor,
            original_max_position_embeddings,
            beta_fast,
            beta_slow,
        )?;
        return Ok(RopeVariant::Yarn(rope));
    } else if rope_type == FloatOrStr::Str("longrope") {
        todo!()
    }

    Err(Exception::custom(format!(
        "Unsupported RoPE type {rope_type:?}"
    )))
}
