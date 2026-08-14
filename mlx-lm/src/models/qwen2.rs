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
    // The sampler is model-agnostic (operates on logit arrays); reuse qwen3's so
    // Qwen2 gets top-p/top-k/repeat-penalty parity without duplicating it.
    models::qwen3::{sample_with, QuantizationConfig, SamplerOpts},
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
};

thread_local! {
    /// Per-row RoPE start offsets for ragged batched decode (a `[B]` array) — see
    /// `qwen3`/`llama`'s identical thread-local. When set, `Attention` ropes q/k at
    /// `cache.offset() − pad_i` per row via `forward_dynamic`; `None` (default) → the scalar
    /// path, so B=1 is byte-identical. The key-pad mask is threaded via `AttentionInput.mask`.
    static BATCH_PAD_OFFSETS: std::cell::RefCell<Option<Array>> =
        const { std::cell::RefCell::new(None) };
}

/// Install (or clear with `None`) the per-row RoPE pad offsets for ragged batched decode on the
/// Qwen2 path — see [`BATCH_PAD_OFFSETS`].
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
    // Qwen2 configs often omit `head_dim`; `get_qwen2_model_args` fills it from
    // `hidden_size / num_attention_heads` when it comes through as 0.
    #[serde(default)]
    pub head_dim: i32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Present on pre-quantized (AFQ) mlx-community checkpoints; drives the
    /// `nn::quantize` build before the weights load. `None` = full precision.
    pub quantization: Option<QuantizationConfig>,
}

fn default_true() -> bool {
    true
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

        // Qwen2 attention bias convention: q/k/v carry a bias, o_proj does not
        // (independent of any `attention_bias` config key, which Qwen2 omits).
        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(true)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(true)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
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
            // Ragged batched decode: per-row rope at `cache.offset() − pad_i`; else scalar.
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
    pub fn new(dim: i32, hidden_dim: i32, mlp_bias: bool) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(mlp_bias)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(mlp_bias)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(mlp_bias)
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
        let mlp = Mlp::new(args.hidden_size, args.intermediate_size, args.mlp_bias)?;
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
pub struct LlamaModel {
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

impl LlamaModel {
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

impl<C> Module<ModelInput<'_, C>> for LlamaModel
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

        // Cache-aware causal mask: with prefix reuse the cache holds `offset` prior
        // positions, so the mask must be (T, offset+T), not (T, T). The old
        // `create_additive_causal_mask(T)` ignored the offset and broke multi-turn
        // prefix reuse (a (T,T) vs (1,H,T,offset+T) broadcast mismatch). Matches qwen3.
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
    pub model: LlamaModel,

    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = LlamaModel::new(&args)?;
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
        <LlamaModel as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

pub fn load_qwen2_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_qwen2_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
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

pub fn load_qwen2_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_qwen2_model_args(model_dir)?;
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
            // QuantizedLinear nests BOTH the packed weight and the linear bias
            // under `inner` (`.inner.weight` / `.inner.bias`), while the quant
            // `scales`/`biases` sit at the top level. Qwen2 carries q/k/v biases,
            // so the `.bias` remap matters here (Qwen3/Llama have none). `.biases`
            // (quant zero-points) does not end in `.bias`, so it is left alone.
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
        eprintln!("LOADED {matched} params (qwen2)");
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

    /// Set top-p / top-k / repeat-penalty (temp came from `new`).
    pub fn set_sampler(
        &mut self,
        top_p: f32,
        top_k: i32,
        repeat_penalty: f32,
        frequency_penalty: f32,
        presence_penalty: f32,
    ) {
        self.sampler.top_p = top_p;
        self.sampler.top_k = top_k;
        self.sampler.repeat_penalty = repeat_penalty;
        self.sampler.frequency_penalty = frequency_penalty;
        self.sampler.presence_penalty = presence_penalty;
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
                let recent: &[u32] = if self.sampler.keeps_history() {
                    &self.history
                } else {
                    &[]
                };
                let y = tri!(sample_with(&logits.index((.., -1, ..)), &self.sampler, recent));
                if self.sampler.keeps_history() {
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
                let recent: &[u32] = if self.sampler.keeps_history() {
                    &self.history
                } else {
                    &[]
                };
                let y = tri!(sample_with(&logits.index((.., -1, ..)), &self.sampler, recent));
                if self.sampler.keeps_history() {
                    tri!(mlx_rs::transforms::eval([&y]));
                    self.history.push(tri!(y.reshape(&[-1])).index(0).item::<u32>());
                }

                self.state = GenerateState::Decode { y: y.clone() };

                Some(Ok(y))
            }
        }
    }
}
