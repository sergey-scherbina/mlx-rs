use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use mlx_rs::{
    argmax_axis, array,
    builder::Builder,
    categorical,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, ModuleParametersExt},
    nn,
    ops::indexing::{IndexOp, NewAxis},
    quantization::MaybeQuantized,
    Array,
};
use serde::Deserialize;
use serde_json::Value;
use std::cell::RefCell;
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
};

thread_local! {
    /// Per-row RoPE start offsets for **ragged batched decode** (a `[B]` array). When
    /// set, the dense `Attention` ropes queries/keys at `cache.offset() − pad_offsets`
    /// (per row) via `RopeVariant::forward_dynamic`, so a batch of different-length
    /// sequences (left-padded into one cache) is rotated at each row's true position.
    /// `None` (the default) → the normal single-offset path, so B=1 is unchanged.
    /// The host (worker is single-threaded) sets it before a batched forward + clears
    /// it after, via [`set_batch_pad_offsets`].
    static BATCH_PAD_OFFSETS: RefCell<Option<Array>> = const { RefCell::new(None) };
}

/// Install (or clear with `None`) the per-row RoPE pad offsets for the next forward(s)
/// on this thread — see [`BATCH_PAD_OFFSETS`]. Used by ragged batched decode.
pub fn set_batch_pad_offsets(offsets: Option<Array>) {
    BATCH_PAD_OFFSETS.with(|c| *c.borrow_mut() = offsets);
}

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
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
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
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(false)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(false)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let q_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;
        let k_norm = nn::RmsNormBuilder::new(head_dim)
            .eps(args.rms_norm_eps)
            .build()?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            q_norm,
            k_norm,
            rope,
        })
    }
}

// TODO: check if this input can be generic for other attention modules
pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
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

        if std::env::var("ROZUM_QPROJ_DEBUG").is_ok() {
            let h0 = queries
                .index((0, -1, 0..5))
                .as_dtype(mlx_rs::Dtype::Float32)
                .ok()
                .map(|a| a.as_slice::<f32>().to_vec());
            let h31 = queries
                .index((0, -1, (31 * 128)..(31 * 128 + 5)))
                .as_dtype(mlx_rs::Dtype::Float32)
                .ok()
                .map(|a| a.as_slice::<f32>().to_vec());
            eprintln!(
                "QPROJ xshape={:?} h0_cols={h0:?} h31_cols={h31:?}",
                x.shape()
            );
        }

        let mut queries = self.q_norm.forward(
            &queries
                .reshape(&[B, L, self.n_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut keys = self.k_norm.forward(
            &keys
                .reshape(&[B, L, self.n_kv_heads, -1])?
                .transpose_axes(&[0, 2, 1, 3])?,
        )?;
        let mut values = values
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if std::env::var("ROZUM_QPROJ_DEBUG").is_ok() {
            let h31 = queries
                .index((0, 31, -1, 0..5))
                .as_dtype(mlx_rs::Dtype::Float32)
                .ok()
                .map(|a| a.as_slice::<f32>().to_vec());
            eprintln!("QNORM(pre-rope) qshape={:?} h31={h31:?}", queries.shape());
        }

        if let Some(cache) = cache.as_mut() {
            // Ragged batched decode: rope each row at `cache.offset() − pad_i` (its
            // true position) instead of one shared offset. Else the normal scalar path.
            let per_row = BATCH_PAD_OFFSETS.with(|c| {
                c.borrow().as_ref().map(|pad| {
                    Array::from_int(cache.offset()).subtract(pad)
                })
            });
            match per_row {
                Some(offsets) => {
                    let offsets = offsets?;
                    queries = self.rope.forward_dynamic(&queries, &offsets)?;
                    keys = self.rope.forward_dynamic(&keys, &offsets)?;
                }
                None => {
                    let q_input = nn::RopeInputBuilder::new(&queries)
                        .offset(cache.offset())
                        .build()?;
                    queries = self.rope.forward(q_input)?;
                    let k_input = nn::RopeInputBuilder::new(&keys)
                        .offset(cache.offset())
                        .build()?;
                    keys = self.rope.forward(k_input)?;
                }
            }

            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        if std::env::var("ROZUM_ATTN_DEBUG").is_ok() {
            let l2 = |a: Array| -> f32 {
                a.square()
                    .and_then(|s| s.sum(None))
                    .and_then(|s| s.sqrt())
                    .map(|s| s.item::<f32>())
                    .unwrap_or(f32::NAN)
            };
            let mask_info = match mask {
                Some(m) => {
                    let lastrow = m.index((-1, ..));
                    let sum = lastrow
                        .sum(None)
                        .map(|s| s.item::<f32>())
                        .unwrap_or(f32::NAN);
                    let minv = lastrow
                        .min(None)
                        .map(|s| s.item::<f32>())
                        .unwrap_or(f32::NAN);
                    format!("mask{:?} lastrow_sum={sum:.2} min={minv:.2}", m.shape())
                }
                None => "mask=None".to_string(),
            };
            let vals = queries
                .index((0, 31, -1, 0..5))
                .as_dtype(mlx_rs::Dtype::Float32)
                .ok()
                .map(|a| a.as_slice::<f32>().to_vec());
            eprintln!(
                "ATTN qshape={:?} q_h31={:.4} q_h31_vals={:?} {mask_info}",
                queries.shape(),
                l2(queries.index((0, 31, -1, ..))),
                vals,
            );
        }

        let attn = crate::utils::scaled_dot_product_attention(
            queries, keys, values, cache, self.scale, mask,
        )?;
        if std::env::var("ROZUM_ATTN_DEBUG").is_ok() {
            let l2 = |a: Array| -> f32 {
                a.square()
                    .and_then(|s| s.sum(None))
                    .and_then(|s| s.sqrt())
                    .map(|s| s.item::<f32>())
                    .unwrap_or(f32::NAN)
            };
            eprintln!(
                "ATTNOUT all_heads_last_l2={:.4} h0={:.4} h31={:.4}",
                l2(attn.index((0, .., -1, ..))),
                l2(attn.index((0, 0, -1, ..))),
                l2(attn.index((0, 31, -1, ..))),
            );
        }
        let output = attn.transpose_axes(&[0, 2, 1, 3])?.reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.q_norm.training_mode(mode);
        self.k_norm.training_mode(mode);
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,

    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;

    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let down_proj_input =
            nn::silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&down_proj_input)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    pub self_attn: Attention,

    #[quantizable]
    #[param]
    pub mlp: Mlp,

    #[param]
    pub input_layernorm: nn::RmsNorm,

    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Attention::new(args)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size)?;
        let input_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        let post_attention_layernorm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            num_attention_heads,
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
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
pub struct Qwen3Model {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,

    #[param]
    pub norm: nn::RmsNorm,
}

impl Qwen3Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..num_hidden_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(embed_tokens),
            layers,
            norm,
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Qwen3Model
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
            // Slots must be Some so attention actually writes/reads KV history;
            // a None slot makes attention run cache-less (no context in decode).
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        let layer_debug = std::env::var("ROZUM_LAYER_DEBUG").is_ok();
        for (i, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)?;
            if layer_debug {
                let last = h.index((0, -1, ..));
                let l2 = last
                    .square()
                    .and_then(|s| s.sum(None))
                    .and_then(|s| s.sqrt())
                    .map(|s| s.item::<f32>())
                    .unwrap_or(f32::NAN);
                eprintln!("LAYER {i} last_l2={l2:.4}");
            }
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    pub model: Qwen3Model,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Qwen3Model::new(&args)?;
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
                MaybeQuantized::Original(embed_tokens) => embed_tokens.as_linear(&out),
                MaybeQuantized::Quantized(q_embed_tokens) => q_embed_tokens.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Qwen3Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_qwen3_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_qwen3_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let model_args: ModelArgs = serde_json::from_reader(file)?;

    Ok(model_args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_qwen3_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_qwen3_model_args(model_dir)?;
    let quantization = model_args.quantization.clone();
    let mut model = Model::new(model_args)?;

    // Pre-quantized mlx-community checkpoints ship AFQ weight/scales/biases; build
    // the matching QuantizedLinear structure before loading so the keys line up.
    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    // mlx-rs QuantizedLinear/QuantizedEmbedding nest the packed weight under
    // `<prefix>.inner.weight`, but mlx-community checkpoints store it as
    // `<prefix>.weight`. Remap `<prefix>.weight` -> `<prefix>.inner.weight`
    // whenever a sibling `<prefix>.scales` is present (i.e. a quantized layer);
    // leave norm and other `.weight` keys untouched. load_safetensors is
    // non-strict (skips unmatched keys silently), so without this the packed
    // weights stay at their random init and the model emits garbage.
    fn load_weights_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let has_scales: HashSet<String> = loaded
            .keys()
            .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
            .collect();
        let mut params = model.parameters_mut().flatten();
        let mut matched = 0usize;
        for (key, value) in loaded {
            let mapped = match key.strip_suffix(".weight") {
                Some(prefix) if has_scales.contains(prefix) => format!("{prefix}.inner.weight"),
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
        eprintln!("LOADED {matched} params");
    }
    model.eval()?;

    Ok(model)
}

/// Sampling knobs. `top_k <= 0`, `top_p >= 1.0` and `repeat_penalty == 1.0`
/// disable those filters; `temp == 0.0` is greedy (argmax) once the repetition
/// penalty (if any) has been applied.
#[derive(Debug, Clone, Copy)]
pub struct SamplerOpts {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: i32,
    /// Repetition penalty (HF convention: divide positive logits / multiply
    /// negative ones for recently-seen tokens). `1.0` = off.
    pub repeat_penalty: f32,
}

impl SamplerOpts {
    /// temp only (top_p/top_k/repeat_penalty off) — the historic
    /// `sample(logits, temp)` behavior.
    pub fn with_temp(temp: f32) -> Self {
        Self {
            temp,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
        }
    }
}

/// Greedy@temp0 / temperature-categorical, preserved for existing callers (no
/// repetition penalty — pass `&[]` recent tokens).
pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    sample_with(logits, &SamplerOpts::with_temp(temp), &[])
}

/// Down/up-weight the logits of `recent` token ids in place (HF repetition
/// penalty): positive logits are divided by `penalty`, negative ones multiplied.
/// `logits`: `[B, vocab]` with `B == 1`. Applied once per token regardless of
/// repeat count (a duplicate index just re-writes the same penalized value).
fn apply_repeat_penalty(logits: &Array, recent: &[u32], penalty: f32) -> Result<Array, Exception> {
    let idx_vals: Vec<i32> = recent.iter().map(|&t| t as i32).collect();
    let idx = Array::from_slice(&idx_vals, &[1, idx_vals.len() as i32]);
    let selected = mlx_rs::ops::indexing::take_along_axis(logits, &idx, -1)?; // [1, N]
    let pen = Array::from_f32(penalty);
    let penalized = mlx_rs::ops::r#where(
        &selected.gt(&Array::from_f32(0.0))?,
        &selected.divide(&pen)?,
        &selected.multiply(&pen)?,
    )?;
    mlx_rs::ops::indexing::put_along_axis(logits, &idx, &penalized, -1)
}

/// Sample one token id per row of `logits` (`[B, vocab]`). Applies the repetition
/// penalty over `recent` (if any), then `temp == 0` is argmax (kept byte-exact for
/// the oracle tests when no penalty), else top-k -> top-p (nucleus) -> categorical.
/// Ported from Python `mlx_lm`.
pub fn sample_with(
    logits: &Array,
    opts: &SamplerOpts,
    recent: &[u32],
) -> Result<Array, Exception> {
    let penalized;
    let logits = if opts.repeat_penalty != 1.0 && !recent.is_empty() {
        penalized = apply_repeat_penalty(logits, recent, opts.repeat_penalty)?;
        &penalized
    } else {
        logits
    };

    if opts.temp == 0.0 {
        return argmax_axis!(logits, -1);
    }
    let mut logits = logits.multiply(array!(1.0 / opts.temp))?;
    let vocab = *logits.shape().last().expect("logits rank >= 1");

    // top-k: keep only the k largest logits (mask the rest to -inf).
    if opts.top_k > 0 && opts.top_k < vocab {
        let sorted = mlx_rs::ops::sort_axis(&logits, -1)?; // ascending
        let kth = sorted.index((.., vocab - opts.top_k)).index((.., NewAxis)); // [B,1]
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        logits = mlx_rs::ops::r#where(&logits.lt(&kth)?, &neg_inf, &logits)?;
    }

    // top-p (nucleus): keep the smallest set of highest-prob tokens summing to p.
    if opts.top_p < 1.0 {
        return top_p_sample(&logits, opts.top_p);
    }

    categorical!(&logits)
}

/// Nucleus sampling: sample within the top-p mass, then map the sorted index back
/// to the original vocab id. `logits`: `[B, vocab]` -> token ids `[B]`.
fn top_p_sample(logits: &Array, top_p: f32) -> Result<Array, Exception> {
    use mlx_rs::ops::indexing::take_along_axis;
    let probs = mlx_rs::ops::softmax_axis(logits, -1, true)?;
    let order = mlx_rs::ops::argsort_axis(&probs, -1)?; // ascending by prob
    let sorted_probs = take_along_axis(&probs, &order, -1)?;
    let cum = mlx_rs::ops::cumsum(&sorted_probs, -1, false, true)?; // inclusive
    // Reading ascending, the nucleus is the high end whose cumulative crosses 1-p.
    let keep = cum.gt(&Array::from_f32(1.0 - top_p))?;
    let zero = Array::from_f32(0.0);
    let kept = mlx_rs::ops::r#where(&keep, &sorted_probs, &zero)?;
    let logp = kept.add(Array::from_f32(1e-9))?.log()?;
    let sorted_tok = categorical!(&logp)?; // [B] index into the sorted axis
    let gathered = take_along_axis(&order, &sorted_tok.index((.., NewAxis)), -1)?; // [B,1]
    Ok(gathered.index((.., 0)))
}

/// Per-row batched sampling: sample one token id per row of `[B, vocab]` logits, each row
/// honoring ITS OWN `temp`/`top_k`/`top_p` (`[B]` arrays). A single unified nucleus path
/// serves every config, so one batch can mix greedy and sampling requests:
///   - `temp == 0` → that row's argmax (greedy), via a final per-row `where` override.
///   - `top_k <= 0` → no top-k cutoff (per-row threshold set to `-inf`).
///   - `top_p >= 1.0` → full nucleus (`1 - top_p <= 0` keeps every token), i.e. plain
///     temperature-categorical.
/// Mirrors the scalar [`sample_with`] math (temp → top-k → top-p → categorical) but with
/// per-row `[B,1]` thresholds. The repetition penalty + explicit seeds are NOT handled here
/// (those rows stay on the serial path); this is the argmax-or-sample core for batched decode.
pub fn sample_rows(
    logits: &Array, // [B, vocab]
    temp: &Array,   // [B] f32
    top_k: &Array,  // [B] i32
    top_p: &Array,  // [B] f32
) -> Result<Array, Exception> {
    use mlx_rs::ops::indexing::take_along_axis;
    let shape = logits.shape();
    let (b, vocab) = (shape[0], *shape.last().expect("logits rank >= 1"));
    let temp_c = temp.reshape(&[b, 1])?;
    let top_k_c = top_k.reshape(&[b, 1])?;
    let top_p_c = top_p.reshape(&[b, 1])?;
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let zero_f = Array::from_f32(0.0);

    // Per-row greedy result (used where temp == 0).
    let greedy = argmax_axis!(logits, -1)?; // [B]

    // Temperature scale (guard temp == 0 against div-by-zero; those rows are overridden).
    let safe_temp = mlx_rs::ops::r#where(&temp_c.eq(&zero_f)?, &array!(1.0), &temp_c)?;
    let mut scaled = logits.divide(&safe_temp)?; // [B, vocab]

    // top-k: per-row threshold = the k-th largest logit (sorted ascending → index vocab-k).
    // `top_k <= 0` rows get a `-inf` threshold (keep all).
    let sorted = mlx_rs::ops::sort_axis(&scaled, -1)?; // ascending
    let kpos = array!(vocab).subtract(&top_k_c)?; // [B,1] = vocab - k
    let kpos = mlx_rs::ops::maximum(&kpos, &array!(0))?;
    let kpos = mlx_rs::ops::minimum(&kpos, &array!(vocab - 1))?;
    let kth = take_along_axis(&sorted, &kpos, -1)?; // [B,1]
    let kth_eff = mlx_rs::ops::r#where(&top_k_c.gt(&array!(0))?, &kth, &neg_inf)?;
    scaled = mlx_rs::ops::r#where(&scaled.ge(&kth_eff)?, &scaled, &neg_inf)?;

    // top-p nucleus in sorted-prob space (per-row threshold 1 - top_p; >= 1 keeps all).
    let probs = mlx_rs::ops::softmax_axis(&scaled, -1, true)?;
    let order = mlx_rs::ops::argsort_axis(&probs, -1)?; // ascending by prob
    let sorted_probs = take_along_axis(&probs, &order, -1)?;
    let cum = mlx_rs::ops::cumsum(&sorted_probs, -1, false, true)?; // inclusive
    let thresh = array!(1.0).subtract(&top_p_c)?; // [B,1]
    let kept = mlx_rs::ops::r#where(&cum.gt(&thresh)?, &sorted_probs, &zero_f)?;
    let logp = kept.add(array!(1e-9))?.log()?;
    let sorted_tok = categorical!(&logp)?; // [B] index into the sorted axis
    let sampled = take_along_axis(&order, &sorted_tok.index((.., NewAxis)), -1)? // [B,1]
        .index((.., 0)); // [B]

    // Greedy override per row (temp == 0).
    mlx_rs::ops::r#where(&temp.eq(&zero_f)?, &greedy, &sampled)
}

/// How many most-recent tokens the repetition penalty considers.
pub const REPEAT_CONTEXT: usize = 256;

/// The recent-token window of a generation history (the trailing
/// [`REPEAT_CONTEXT`] ids), for the repetition penalty.
pub fn repeat_window(history: &[u32]) -> &[u32] {
    &history[history.len().saturating_sub(REPEAT_CONTEXT)..]
}

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    sampler: SamplerOpts,
    history: Vec<u32>,
    state: GenerateState<'a>,
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache,
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
            sampler: SamplerOpts::with_temp(temp),
            history: Vec::new(),
            state: GenerateState::Prefill { prompt_token },
        }
    }

    /// Set top-p / top-k / repeat-penalty (temp came from `new`).
    pub fn set_sampler(&mut self, top_p: f32, top_k: i32, repeat_penalty: f32) {
        self.sampler.top_p = top_p;
        self.sampler.top_k = top_k;
        self.sampler.repeat_penalty = repeat_penalty;
    }
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
}

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<'a, C> Iterator for Generate<'a, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        match &self.state {
            GenerateState::Prefill { prompt_token } => {
                let input = ModelInput {
                    inputs: prompt_token,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
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
                if self.sampler.repeat_penalty != 1.0 {
                    tri!(mlx_rs::transforms::eval([&y]));
                    self.history.push(tri!(y.reshape(&[-1])).index(0).item::<u32>());
                }
                self.state = GenerateState::Decode { y: y.clone() };

                Some(Ok(y))
            }
            GenerateState::Decode { y } => {
                let inputs = y.index((.., NewAxis));
                let input = ModelInput {
                    inputs: &inputs,
                    mask: None,
                    cache: self.cache,
                };
                let logits = tri!(self.model.forward(input));
                let recent: &[u32] = if self.sampler.repeat_penalty != 1.0 {
                    repeat_window(&self.history)
                } else {
                    &[]
                };
                let y = tri!(sample_with(&logits, &self.sampler, recent));
                if self.sampler.repeat_penalty != 1.0 {
                    tri!(mlx_rs::transforms::eval([&y]));
                    self.history.push(tri!(y.reshape(&[-1])).index(0).item::<u32>());
                }

                self.state = GenerateState::Decode { y: y.clone() };

                Some(Ok(y))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::{
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::ConcatKeyValueCache,
        models::qwen3::{load_qwen3_model, load_qwen3_tokenizer},
    };

    const CACHED_TEST_MODEL_DIR: &str = "../cache/Qwen3-4B-bf16";

    // Deterministic anchors for the top-k / top-p sampler: both collapse to argmax
    // when only the top token can be selected, regardless of temp. Pins the
    // filtering math without relying on RNG.
    #[test]
    fn sample_with_collapses_to_argmax() {
        use super::{sample_with, SamplerOpts};
        // A clear argmax at index 3.
        let logits = Array::from_slice(&[0.1f32, 0.5, -1.0, 4.0, 0.2, 1.0], &[1, 6]);
        let want = 3u32;

        // top_k = 1 -> only the max survives -> categorical must return it.
        for temp in [0.7f32, 1.0, 2.0] {
            let t = sample_with(
                &logits,
                &SamplerOpts {
                    temp,
                    top_p: 1.0,
                    top_k: 1,
                    repeat_penalty: 1.0,
                },
                &[],
            )
            .unwrap();
            eval([&t]).unwrap();
            assert_eq!(t.index(0).item::<u32>(), want, "top_k=1 temp={temp}");
        }

        // Tiny top_p -> nucleus is just the top token -> argmax.
        let t = sample_with(
            &logits,
            &SamplerOpts {
                temp: 1.0,
                top_p: 1e-4,
                top_k: 0,
                repeat_penalty: 1.0,
            },
            &[],
        )
        .unwrap();
        eval([&t]).unwrap();
        assert_eq!(t.index(0).item::<u32>(), want, "tiny top_p");

        // temp 0 stays argmax (greedy fast path).
        let t = sample_with(&logits, &SamplerOpts::with_temp(0.0), &[]).unwrap();
        eval([&t]).unwrap();
        assert_eq!(t.index(0).item::<u32>(), want, "greedy");

        // Repetition penalty: greedy argmax is index 3 (logit 4.0). Penalizing 3
        // hard (divide by 100) drops it below index 5 (logit 1.0) -> argmax moves.
        let t = sample_with(
            &logits,
            &SamplerOpts {
                temp: 0.0,
                top_p: 1.0,
                top_k: 0,
                repeat_penalty: 100.0,
            },
            &[3],
        )
        .unwrap();
        eval([&t]).unwrap();
        assert_eq!(t.index(0).item::<u32>(), 5, "repeat_penalty pushes argmax off 3");
    }

    // Per-row batched sampler: each row honors its OWN temp/top_k/top_p, and one batch can
    // MIX greedy + sampling. Deterministic anchors — every row's config collapses to that
    // row's argmax (temp=0 greedy override / top_k=1 / tiny top_p), with a dominant logit so
    // the 1e-9 nucleus floor can't flip it. Proves per-row thresholds + the greedy override
    // pick the right token per row independently.
    #[test]
    fn sample_rows_per_row_collapses_to_argmax() {
        use super::sample_rows;
        // Three rows with DISTINCT dominant argmax: row0→2, row1→0, row2→5 (logit 9.0).
        let logits = Array::from_slice(
            &[
                0.1f32, 0.2, 9.0, 0.3, 0.4, 0.5, // row0 argmax = 2
                9.0, 0.1, 0.2, 0.3, 0.4, 0.5, // row1 argmax = 0
                0.1, 0.2, 0.3, 0.4, 0.5, 9.0, // row2 argmax = 5
            ],
            &[3, 6],
        );
        let want = [2u32, 0, 5];
        // Mixed per-row config that each collapses to argmax:
        //   row0: temp 0   (greedy override)
        //   row1: top_k 1  at temp 2 (only the max survives)
        //   row2: tiny top_p at temp 1 (nucleus = just the top token)
        let temp = Array::from_slice(&[0.0f32, 2.0, 1.0], &[3]);
        let top_k = Array::from_slice(&[0i32, 1, 0], &[3]);
        let top_p = Array::from_slice(&[1.0f32, 1.0, 1e-4], &[3]);
        let t = sample_rows(&logits, &temp, &top_k, &top_p).unwrap();
        eval([&t]).unwrap();
        for r in 0..3 {
            assert_eq!(
                t.index(r as i32).item::<u32>(),
                want[r],
                "row {r} must collapse to its own argmax"
            );
        }

        // All-greedy batch (temp 0 everywhere) must equal per-row argmax exactly — this is
        // the path the previously greedy-only batched decode relied on, now via sample_rows.
        let temp0 = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]);
        let tk0 = Array::from_slice(&[0i32, 0, 0], &[3]);
        let tp1 = Array::from_slice(&[1.0f32, 1.0, 1.0], &[3]);
        let g = sample_rows(&logits, &temp0, &tk0, &tp1).unwrap();
        eval([&g]).unwrap();
        for r in 0..3 {
            assert_eq!(g.index(r as i32).item::<u32>(), want[r], "greedy row {r}");
        }
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_qwen3_model() {
        let _model = super::load_qwen3_model(CACHED_TEST_MODEL_DIR).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_qwen3_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_qwen3_with_concat_cache() {
        let tokenizer = load_qwen3_tokenizer(CACHED_TEST_MODEL_DIR).unwrap();

        let mut model = load_qwen3_model(CACHED_TEST_MODEL_DIR).unwrap();

        let encoding = tokenizer.encode("hello", true).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let mut tokens = Vec::new();
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt_tokens,
        );
        for (token, ntoks) in generate.zip(0..10) {
            let token = token.unwrap();
            tokens.push(token.clone());

            if ntoks == 0 {
                eval(&tokens).unwrap();
            }

            if tokens.len() % 20 == 0 {
                eval(&tokens).unwrap();
                let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
                let s = tokenizer.decode(&slice, true).unwrap();
                print!("{s}");
            }
        }

        eval(&tokens).unwrap();
        let slice: Vec<u32> = tokens.drain(..).map(|t| t.item::<u32>()).collect();
        let s = tokenizer.decode(&slice, true).unwrap();
        println!("{s}");

        println!("------");
    }
}
