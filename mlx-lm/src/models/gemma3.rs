//! Gemma 3 (text) for the native MLX runtime, ported from Python `mlx_lm.models.gemma3_text`.
//!
//! Gemma 3 is a distinct architecture, NOT a Llama alias. The quirks:
//!   - **RMSNorm with the `(1 + weight)` convention** ([`GemmaRmsNorm`]) everywhere, computed in f32.
//!   - **Embedding scaled by `sqrt(hidden_size)`** (cast to the activation dtype) after lookup.
//!   - **Per-head q/k RMSNorm** before RoPE.
//!   - **Four norms per layer** (pre/post around both attention and the MLP).
//!   - **GELU (tanh approx) MLP**, not SiLU.
//!   - **Alternating local/global attention**: every `sliding_window_pattern`-th layer is GLOBAL
//!     (RoPE base `rope_theta`, full causal attention); the rest are LOCAL (RoPE base
//!     `rope_local_base_freq`) and additionally mask keys older than `sliding_window` (see
//!     [`build_gemma_masks`]). The two masks coincide when the context fits in the window.
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

thread_local! {
    /// Per-row RoPE start offsets for ragged batched DECODE (a `[B]` array) — see qwen3/llama's
    /// identical thread-local. When set, `Attention` ropes q/k per-row via `forward_dynamic`, and
    /// [`Gemma3Model`] treats the caller's mask as the per-row global (pad) mask and derives the
    /// local mask from it + the sliding window. `None` (default) → the serial path (unchanged).
    static BATCH_PAD_OFFSETS: std::cell::RefCell<Option<Array>> =
        const { std::cell::RefCell::new(None) };
}

/// Install (or clear with `None`) the per-row RoPE pad offsets for ragged batched decode on the
/// Gemma 3 path — see [`BATCH_PAD_OFFSETS`].
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
    /// Local (sliding-window) attention layer, vs a global (full-attention) one.
    pub is_local: bool,
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
            is_local: !is_global,
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
    pub sliding_window: i32,
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
            sliding_window: args.sliding_window,
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(args.vocab_size, args.hidden_size)?),
            layers,
            norm: GemmaRmsNorm::new(args.hidden_size, args.rms_norm_eps),
        })
    }
}

/// Build the `(global, local)` additive attention masks `[L, offset+L]` for Gemma's
/// alternating attention: `global` is causal (`key <= query`); `local` additionally drops keys
/// older than `window` (`key <= query - window`). Built over ABSOLUTE positions (`offset..`) so
/// both are correct at decode; when the whole context fits in the window they're identical.
fn build_gemma_masks(
    offset: i32,
    l: i32,
    window: i32,
    dt: Dtype,
) -> Result<(Array, Array), Exception> {
    let total = offset + l;
    let qpos = mlx_rs::ops::arange::<_, i32>(offset, total, None)?.reshape(&[l, 1])?;
    let kpos = mlx_rs::ops::arange::<_, i32>(0, total, None)?.reshape(&[1, total])?;
    let zero = Array::from_f32(0.0);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let causal = mlx_rs::ops::r#where(&kpos.le(&qpos)?, &zero, &neg_inf)?;
    let global = causal.as_dtype(dt)?;
    let thresh = qpos.subtract(Array::from_int(window))?;
    let window_mask = mlx_rs::ops::r#where(&kpos.gt(&thresh)?, &zero, &neg_inf)?;
    let local = causal.add(&window_mask)?.as_dtype(dt)?;
    Ok((global, local))
}

/// Boolean window keep-mask `[total]` for ragged batched DECODE: keep the last `window` key
/// slots (`j >= total - window`). Rows are right-aligned in the batched cache, so the window is
/// uniform across rows; AND-ed with the per-row pad mask it gives each row's local attention.
fn build_window_keep(total: i32, window: i32) -> Result<Array, Exception> {
    let kpos = mlx_rs::ops::arange::<_, i32>(0, total, None)?;
    kpos.ge(&Array::from_int(total - window))
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

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        // GLOBAL layers attend causally to all keys; LOCAL (sliding-window) layers also drop
        // keys older than `sliding_window`. Build both additive masks `[L, offset+L]` over the
        // absolute positions (so they're correct at decode, where `offset > 0`); local == global
        // whenever the whole context fits in the window. An explicit caller mask (unused by
        // Gemma — it isn't batched) applies to every layer.
        let (global_mask, local_mask) = if let Some(m) = mask {
            // Batched decode: `m` is the per-row pad keep-mask (the global mask). For local
            // layers AND it with the sliding window (keep the last `window` key slots — rows are
            // right-aligned so the window is uniform). `m` is `[B,1,1,total]`; the `[total]`
            // window broadcasts.
            let total = *m.shape().last().expect("mask has a key axis");
            let win = build_window_keep(total, self.sliding_window)?;
            let local = mlx_rs::ops::logical_and(m, &win)?;
            (Some(m.clone()), Some(local))
        } else {
            let l = h.shape()[1];
            let offset = cache.first().and_then(|c| c.as_ref()).map_or(0, |c| c.offset());
            let (g, lo) = build_gemma_masks(offset, l, self.sliding_window, h.dtype())?;
            (Some(g), Some(lo))
        };

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let m = if layer.is_local { local_mask.as_ref() } else { global_mask.as_ref() };
            h = layer.forward(AttentionInput { x: &h, mask: m, cache: c.as_mut() })?;
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
    /// Gemma ties embeddings, but mlx-community 4-bit conversions materialize a SEPARATE
    /// quantized `lm_head` (its quant params differ from the embedding's), so when the
    /// checkpoint ships one we must use it — the tied `embed_tokens.as_linear` would give
    /// wrong logits. `None` = truly tied (use the embedding).
    #[quantizable]
    #[param]
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs, has_lm_head: bool) -> Result<Self, Exception> {
        let model = Gemma3Model::new(&args)?;
        let lm_head = if has_lm_head {
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
            // Truly tied: lm_head is the embedding read as a linear.
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(e) => e.as_linear(&out),
                MaybeQuantized::Quantized(q) => q.as_linear(&out),
            },
        }
    }

    fn training_mode(&mut self, mode: bool) {
        <Gemma3Model as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct WeightMap {
    weight_map: std::collections::HashMap<String, String>,
}

pub fn load_gemma3_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args: ModelArgs =
        serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
    let quantization = args.quantization.clone();

    // Gather every weight tensor up front (so we can detect a materialized `lm_head`).
    let mut loaded: std::collections::HashMap<String, Array> = std::collections::HashMap::new();
    let index = model_dir.join("model.safetensors.index.json");
    let files: Vec<std::path::PathBuf> = if index.exists() {
        let wm: WeightMap = serde_json::from_str(&std::fs::read_to_string(index)?)?;
        wm.weight_map
            .values()
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|f| model_dir.join(f))
            .collect()
    } else {
        vec![model_dir.join("model.safetensors")]
    };
    for f in files {
        for (k, v) in mlx_rs::Array::load_safetensors(&f)? {
            loaded.insert(k, v);
        }
    }

    let has_lm_head = loaded.keys().any(|k| k.starts_with("lm_head."));
    let mut model = Model::new(args, has_lm_head)?;
    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

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
    if std::env::var("ROZUM_MLX_DEBUG").is_ok() {
        eprintln!("LOADED {matched} params (gemma3, lm_head={has_lm_head})");
    }
    drop(params);
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

#[cfg(test)]
mod tests {
    use super::build_gemma_masks;
    use mlx_rs::{transforms::eval, Dtype};

    // The local mask must BAND the causal mask to the last `window` keys; the global mask is
    // plain causal. Deterministic — proves the sliding window is active (no model needed).
    #[test]
    fn sliding_window_mask_bands_local_attention() {
        // L=4 fresh prefill (offset 0), window 2.
        let (g, l) = build_gemma_masks(0, 4, 2, Dtype::Float32).unwrap();
        eval([&g, &l]).unwrap();
        let (gv, lv) = (g.as_slice::<f32>(), l.as_slice::<f32>());
        let neg = f32::NEG_INFINITY;
        for i in 0..4i32 {
            for j in 0..4i32 {
                let idx = (i * 4 + j) as usize;
                let g_keep = j <= i; // causal
                let l_keep = g_keep && j > i - 2; // + within window 2
                assert_eq!(gv[idx], if g_keep { 0.0 } else { neg }, "global[{i}][{j}]");
                assert_eq!(lv[idx], if l_keep { 0.0 } else { neg }, "local[{i}][{j}]");
            }
        }

        // Decode step at offset 5, L=1, window 2: the single query (pos 5) keeps keys 4,5.
        let (_g2, l2) = build_gemma_masks(5, 1, 2, Dtype::Float32).unwrap();
        eval([&l2]).unwrap();
        let lv2 = l2.as_slice::<f32>(); // [1, 6]
        for j in 0..6i32 {
            let keep = j <= 5 && j > 3; // keys 4,5
            assert_eq!(lv2[j as usize], if keep { 0.0 } else { neg }, "decode local[{j}]");
        }
    }
}
