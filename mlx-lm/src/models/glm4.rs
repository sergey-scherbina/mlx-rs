//! GLM-4 (Zhipu/Z.ai, dense) — `model_type: "glm4"`, `architectures: ["Glm4ForCausalLM"]`.
//!
//! Architecturally Qwen2 (GQA + q/k/v bias) with three GLM specifics, verified against the Python
//! `mlx_lm.models.glm4` reference:
//!   1. **Partial, traditional RoPE** — only `head_dim * partial_rotary_factor` (= 64 of 128) dims
//!      are rotated, in the interleaved/`traditional=true` form; the rest pass through.
//!   2. **Fused `gate_up_proj`** — one `hidden → 2*intermediate` projection, split in half → SwiGLU.
//!   3. **Sandwich norm** — FOUR RMSNorms per layer: a pre-norm before each sublayer AND a post-norm
//!      on each sublayer's output before the residual add (`post_self_attn_layernorm` /
//!      `post_mlp_layernorm`). Qwen2 has only the two pre-norms.
//!
//! Mirrors `qwen2.rs` for everything else (quant-aware load + `.bias`→`.inner.bias` remap, sampler,
//! `Generate`). Field names match the HF checkpoint tensors exactly (the gpt-oss "garbage bug" risk).

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
use tokenizers::Tokenizer;

use crate::{
    cache::KeyValueCache,
    error::Error,
    // Sampler is model-agnostic (operates on logit arrays) — reuse qwen3's, like qwen2 does.
    models::qwen3::{repeat_window, sample_with, QuantizationConfig, SamplerOpts},
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
};

thread_local! {
    /// Per-row RoPE start offsets for ragged batched decode (a `[B]` array) — identical to
    /// qwen2/qwen3. `None` (default) → the scalar path, so B=1 is byte-identical.
    static BATCH_PAD_OFFSETS: std::cell::RefCell<Option<Array>> =
        const { std::cell::RefCell::new(None) };
}

/// Install (or clear with `None`) the per-row RoPE pad offsets for ragged batched decode.
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
    #[serde(default)]
    pub head_dim: i32,
    /// GLM rotates only the first `head_dim * partial_rotary_factor` dims (0.5 → 64 of 128).
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    /// GLM-4 carries q/k/v bias (`attention_bias: true`); o_proj has none.
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Present on pre-quantized (AFQ) mlx-community checkpoints. `None` = full precision.
    pub quantization: Option<QuantizationConfig>,
}

fn default_true() -> bool {
    true
}
fn default_partial_rotary_factor() -> f32 {
    0.5
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
    pub rope: RopeVariant,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        // GLM-4: q/k/v carry bias (`attention_bias`), o_proj does not.
        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(args.attention_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        // Partial + traditional RoPE: rotate only the first `head_dim * partial_rotary_factor` dims.
        let rope_dims = (head_dim as f32 * args.partial_rotary_factor) as i32;
        let rope = initialize_rope(
            rope_dims,
            args.rope_theta,
            true, // GLM uses traditional (interleaved) RoPE
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
            rope,
        })
    }
}

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
            let per_row = BATCH_PAD_OFFSETS.with(|c| {
                c.borrow()
                    .as_ref()
                    .map(|pad| Array::from_int(cache.offset()).subtract(pad))
            });
            match per_row {
                Some(offsets) => {
                    let offsets = offsets?;
                    queries = self.rope.forward_dynamic(&queries, &offsets)?;
                    keys = self.rope.forward_dynamic(&keys, &offsets)?;
                }
                None => {
                    queries = self.rope.forward(
                        nn::RopeInputBuilder::new(&queries).offset(cache.offset()).build()?,
                    )?;
                    keys = self
                        .rope
                        .forward(nn::RopeInputBuilder::new(&keys).offset(cache.offset()).build()?)?;
                }
            }
            (keys, values) = cache.update_and_fetch(keys, values)?;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        let output = crate::utils::scaled_dot_product_attention(
            queries, keys, values, cache, self.scale, mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    /// Fused gate+up: `hidden → 2*intermediate`, split in half for SwiGLU.
    #[quantizable]
    #[param]
    pub gate_up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_up_proj = nn::LinearBuilder::new(dim, 2 * hidden_dim).bias(false).build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim).bias(false).build()?;
        Ok(Self {
            gate_up_proj: MaybeQuantized::Original(gate_up_proj),
            down_proj: MaybeQuantized::Original(down_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gate_up = self.gate_up_proj.forward(input)?;
        let parts = gate_up.split(2, -1)?; // [gate, up] along the last axis
        let h = nn::silu(&parts[0])?.multiply(&parts[1])?;
        self.down_proj.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
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
    pub post_self_attn_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    pub post_mlp_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let self_attn = Attention::new(args)?;
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size)?;
        let mk = || {
            nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()
        };
        Ok(Self {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn,
            mlp,
            input_layernorm: mk()?,
            post_self_attn_layernorm: mk()?,
            post_attention_layernorm: mk()?,
            post_mlp_layernorm: mk()?,
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

        // GLM-4 sandwich norm:
        //   x = x + post_self_attn_layernorm(self_attn(input_layernorm(x)))
        //   x = x + post_mlp_layernorm(mlp(post_attention_layernorm(x)))
        let attn_in = AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        };
        let r = self.self_attn.forward(attn_in)?;
        let h = x.add(&self.post_self_attn_layernorm.forward(&r)?)?;

        let r = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&h)?)?;
        h.add(&self.post_mlp_layernorm.forward(&r)?)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_self_attn_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.post_mlp_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Glm4Model {
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

impl Glm4Model {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());
        let embed_tokens = nn::Embedding::new(args.vocab_size, args.hidden_size)?;
        let layers = (0..args.num_hidden_layers)
            .map(|_| TransformerBlock::new(args))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNormBuilder::new(args.hidden_size)
            .eps(args.rms_norm_eps)
            .build()?;
        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
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

impl<C> Module<ModelInput<'_, C>> for Glm4Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mask, cache } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                _ => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)?;
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
    pub model: Glm4Model,
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Glm4Model::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };
        Ok(Self { args, model, lm_head })
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
                MaybeQuantized::Quantized(e) => e.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Glm4Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_glm4_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    Tokenizer::from_file(model_dir.as_ref().join("tokenizer.json")).map_err(Into::into)
}

pub fn get_glm4_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let file = std::fs::File::open(model_dir.as_ref().join("config.json"))?;
    let mut model_args: ModelArgs = serde_json::from_reader(file)?;
    if model_args.head_dim == 0 {
        model_args.head_dim = model_args.hidden_size / model_args.num_attention_heads;
    }
    Ok(model_args)
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightMap {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

pub fn load_glm4_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_glm4_model_args(model_dir)?;
    let quantization = model_args.quantization.clone();
    let mut model = Model::new(model_args)?;

    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    // Same AFQ remap as qwen2: `<prefix>.weight`→`<prefix>.inner.weight` and (GLM has q/k/v bias)
    // `<prefix>.bias`→`<prefix>.inner.bias` whenever a sibling `<prefix>.scales` marks a quant layer.
    fn load_weights_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let has_scales: HashSet<String> = loaded
            .keys()
            .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
            .collect();
        let mut params = model.parameters_mut().flatten();
        let mut matched = 0usize;
        for (key, value) in loaded {
            let mapped = if let Some(prefix) =
                key.strip_suffix(".weight").filter(|p| has_scales.contains(*p))
            {
                format!("{prefix}.inner.weight")
            } else if let Some(prefix) =
                key.strip_suffix(".bias").filter(|p| has_scales.contains(*p))
            {
                format!("{prefix}.inner.bias")
            } else {
                key
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
        eprintln!("LOADED {matched} params (glm4)");
    }
    model.eval()?;

    Ok(model)
}

pub fn sample(logits: &Array, temp: f32) -> Result<Array, Exception> {
    match temp {
        0.0 => argmax_axis!(logits, -1),
        _ => {
            let logits = logits.multiply(array!(1.0 / temp))?;
            categorical!(logits)
        }
    }
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
                let y = tri!(sample_with(&logits.index((.., -1, ..)), &self.sampler, recent));
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
                let y = tri!(sample_with(&logits.index((.., -1, ..)), &self.sampler, recent));
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
