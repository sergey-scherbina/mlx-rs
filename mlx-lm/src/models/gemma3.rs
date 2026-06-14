//! Gemma 3 (text) for the native MLX runtime, ported from Python `mlx_lm.models.gemma3_text`.
//!
//! Gemma 3 is a distinct architecture, NOT a Llama alias. The quirks:
//!   - **RMSNorm with the `(1 + weight)` convention** ([`GemmaRmsNorm`]) everywhere, computed in f32.
//!   - **Embedding scaled by `sqrt(hidden_size)`** (cast to the activation dtype) after lookup.
//!   - **Per-head q/k RMSNorm** before RoPE.
//!   - **Four norms per layer** (pre/post around both attention and the MLP).
//!   - **GELU (tanh approx) MLP**, not SiLU.
//!   - **Alternating local/global attention**: every `sliding_window_pattern`-th layer is GLOBAL
//!     (RoPE base `rope_theta`, full attention); the rest are LOCAL (RoPE base `rope_local_base_freq`,
//!     sliding window). The window is approximated by full attention here — exact for contexts within
//!     the window, a bounded divergence beyond it (a windowed-mask follow-up; agent prompts are short).
//!   - **Attention scale** `query_pre_attn_scalar^-0.5` (not `head_dim^-0.5` in general).
//!   - **Tied embeddings** (the lm_head is the embedding read as a linear).

use std::{collections::HashSet, path::Path};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters, ModuleParametersExt, Param},
    nn,
    ops::indexing::{IndexOp, NewAxis},
    quantization::MaybeQuantized,
    Array, Dtype,
};
use serde::Deserialize;

use crate::{
    cache::KeyValueCache,
    error::Error,
    models::qwen3::{repeat_window, sample_with, QuantizationConfig, SamplerOpts},
    utils::rope::{initialize_rope, RopeVariant},
};

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub max_position_embeddings: i32,
    pub rope_theta: f32,
    pub rope_local_base_freq: f32,
    pub sliding_window: i32,
    pub sliding_window_pattern: i32,
    pub query_pre_attn_scalar: f32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub quantization: Option<QuantizationConfig>,
}

/// RMSNorm with Gemma's `output = norm(x) * (1 + weight)` convention, computed in f32.
#[derive(Debug, Clone, ModuleParameters)]
pub struct GemmaRmsNorm {
    pub eps: f32,
    #[param]
    pub weight: Param<Array>,
}

impl GemmaRmsNorm {
    fn new(dims: i32, eps: f32) -> Self {
        Self {
            eps,
            // Stored weight is the delta from 1; the checkpoint overwrites it.
            weight: Param::new(Array::zeros::<f32>(&[dims]).unwrap()),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let xf = x.as_dtype(Dtype::Float32)?;
        let var = xf.square()?.mean_axes(&[-1], true)?;
        let normed = xf.multiply(&var.add(Array::from_f32(self.eps))?.rsqrt()?)?;
        let scaled = normed.multiply(
            &self
                .weight
                .value
                .as_dtype(Dtype::Float32)?
                .add(Array::from_f32(1.0))?,
        )?;
        scaled.as_dtype(x.dtype())
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
#[allow(non_snake_case)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
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
    pub q_norm: GemmaRmsNorm,
    #[param]
    pub k_norm: GemmaRmsNorm,
    #[param]
    pub rope: RopeVariant,
}

impl Attention {
    fn new(args: &ModelArgs, theta: f32) -> Result<Self, Exception> {
        let d = args.hidden_size;
        let nh = args.num_attention_heads;
        let nkv = args.num_key_value_heads;
        let hd = args.head_dim;
        let q_proj = nn::LinearBuilder::new(d, nh * hd).bias(false).build()?;
        let k_proj = nn::LinearBuilder::new(d, nkv * hd).bias(false).build()?;
        let v_proj = nn::LinearBuilder::new(d, nkv * hd).bias(false).build()?;
        let o_proj = nn::LinearBuilder::new(nh * hd, d).bias(false).build()?;
        let q_norm = GemmaRmsNorm::new(hd, args.rms_norm_eps);
        let k_norm = GemmaRmsNorm::new(hd, args.rms_norm_eps);
        let rope = initialize_rope(hd, theta, false, &None, args.max_position_embeddings)?;
        Ok(Self {
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            scale: args.query_pre_attn_scalar.powf(-0.5),
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
        let (B, L) = (shape[0], shape[1]);

        // q/k get a per-head RMSNorm (over head_dim) BEFORE RoPE; v does not.
        let queries = self
            .q_norm
            .forward(&self.q_proj.forward(x)?.reshape(&[B, L, self.n_heads, self.head_dim])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let keys = self
            .k_norm
            .forward(&self.k_proj.forward(x)?.reshape(&[B, L, self.n_kv_heads, self.head_dim])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let values = self
            .v_proj
            .forward(x)?
            .reshape(&[B, L, self.n_kv_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let mut queries = queries;
        let mut keys = keys;
        let (keys, values) = if let Some(cache) = cache.as_mut() {
            queries = self
                .rope
                .forward(nn::RopeInputBuilder::new(&queries).offset(cache.offset()).build()?)?;
            keys = self
                .rope
                .forward(nn::RopeInputBuilder::new(&keys).offset(cache.offset()).build()?)?;
            cache.update_and_fetch(keys, values)?
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
            (keys, values)
        };

        let out =
            crate::utils::scaled_dot_product_attention(queries, keys, values, cache, self.scale, mask)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[B, L, -1])?;
        self.o_proj.forward(&out)
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
    pub up_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    fn new(dim: i32, hidden: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: MaybeQuantized::Original(nn::LinearBuilder::new(dim, hidden).bias(false).build()?),
            up_proj: MaybeQuantized::Original(nn::LinearBuilder::new(dim, hidden).bias(false).build()?),
            down_proj: MaybeQuantized::Original(nn::LinearBuilder::new(hidden, dim).bias(false).build()?),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let gated = nn::gelu_approximate(self.gate_proj.forward(x)?)?;
        self.down_proj.forward(&gated.multiply(self.up_proj.forward(x)?)?)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct DecoderLayer {
    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: GemmaRmsNorm,
    #[param]
    pub post_attention_layernorm: GemmaRmsNorm,
    #[param]
    pub pre_feedforward_layernorm: GemmaRmsNorm,
    #[param]
    pub post_feedforward_layernorm: GemmaRmsNorm,
}

impl DecoderLayer {
    fn new(args: &ModelArgs, layer_idx: i32) -> Result<Self, Exception> {
        // Every `sliding_window_pattern`-th layer is GLOBAL (rope_theta); the rest LOCAL.
        let is_global = (layer_idx + 1) % args.sliding_window_pattern == 0;
        let theta = if is_global { args.rope_theta } else { args.rope_local_base_freq };
        Ok(Self {
            self_attn: Attention::new(args, theta)?,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps),
            post_attention_layernorm: GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps),
            pre_feedforward_layernorm: GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps),
            post_feedforward_layernorm: GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps),
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for DecoderLayer
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;
        let attn = self.self_attn.forward(AttentionInput {
            x: &self.input_layernorm.forward(x)?,
            mask,
            cache,
        })?;
        let h = x.add(&self.post_attention_layernorm.forward(&attn)?)?;
        let ff = self.mlp.forward(&self.pre_feedforward_layernorm.forward(&h)?)?;
        h.add(&self.post_feedforward_layernorm.forward(&ff)?)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        // GemmaRmsNorm carries no training-mode state.
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Gemma3Model {
    pub hidden_size: i32,
    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<DecoderLayer>,
    #[param]
    pub norm: GemmaRmsNorm,
}

impl Gemma3Model {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            hidden_size: args.hidden_size,
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(args.vocab_size, args.hidden_size)?),
            layers,
            norm: GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps),
        })
    }
}

pub struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Gemma3Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput { inputs, mask, cache } = input;
        let mut h = self.embed_tokens.forward(inputs)?;
        // Gemma scales the embeddings by sqrt(hidden) cast to the activation dtype.
        let normalizer = Array::from_f32((self.hidden_size as f32).sqrt()).as_dtype(h.dtype())?;
        h = h.multiply(&normalizer)?;

        let mask = match mask {
            Some(m) => Some(m.clone()),
            None => {
                if h.shape()[1] > 1 {
                    Some(
                        nn::MultiHeadAttention::create_additive_causal_mask::<f32>(h.shape()[1])?
                            .as_dtype(h.dtype())?,
                    )
                } else {
                    None
                }
            }
        };
        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(AttentionInput { x: &h, mask: mask.as_ref(), cache: c.as_mut() })?;
        }
        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <DecoderLayer as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,
    #[quantizable]
    #[param]
    pub model: Gemma3Model,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = Gemma3Model::new(&args)?;
        Ok(Self { args, model })
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
        // Tied embeddings: lm_head is the embedding read as a linear.
        match &mut self.model.embed_tokens {
            MaybeQuantized::Original(e) => e.as_linear(&out),
            MaybeQuantized::Quantized(q) => q.as_linear(&out),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma3Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
    }
}

#[derive(Debug, Clone, Deserialize)]
struct WeightMap {
    weight_map: std::collections::HashMap<String, String>,
}

pub fn load_gemma3_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args: ModelArgs = serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let quantization = args.quantization.clone();
    let mut model = Model::new(args)?;
    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    fn load_remapped(model: &mut Model, file: &Path) -> Result<usize, Error> {
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

    let index = model_dir.join("model.safetensors.index.json");
    let mut matched = 0usize;
    if index.exists() {
        let wm: WeightMap = serde_json::from_str(&std::fs::read_to_string(index)?)?;
        for f in wm.weight_map.values().collect::<HashSet<_>>() {
            matched += load_remapped(&mut model, &model_dir.join(f))?;
        }
    } else {
        matched = load_remapped(&mut model, &model_dir.join("model.safetensors"))?;
    }
    if std::env::var("ROZUM_MLX_DEBUG").is_ok() {
        eprintln!("LOADED {matched} params (gemma3)");
    }
    model.eval()?;
    Ok(model)
}

// ─── Generate (greedy/sampled token iterator; mirrors llama::Generate) ────────

pub struct Generate<'a, C> {
    model: &'a mut Model,
    cache: &'a mut Vec<Option<C>>,
    sampler: SamplerOpts,
    history: Vec<u32>,
    state: GenerateState<'a>,
}

pub enum GenerateState<'a> {
    Prefill { prompt_token: &'a Array },
    Decode { y: Array },
}

impl<'a, C> Generate<'a, C>
where
    C: KeyValueCache,
{
    pub fn new(model: &'a mut Model, cache: &'a mut Vec<Option<C>>, temp: f32, prompt_token: &'a Array) -> Self {
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

macro_rules! tri {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => return Some(Err(e.into())),
        }
    };
}

impl<C> Iterator for Generate<'_, C>
where
    C: KeyValueCache + Default,
{
    type Item = Result<Array, Exception>;

    fn next(&mut self) -> Option<Self::Item> {
        let inputs_owned;
        let inputs: &Array = match &self.state {
            GenerateState::Prefill { prompt_token } => prompt_token,
            GenerateState::Decode { y } => {
                inputs_owned = y.index((.., NewAxis));
                &inputs_owned
            }
        };
        let logits = tri!(self.model.forward(ModelInput { inputs, mask: None, cache: self.cache }));
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
