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

thread_local! {
    /// Per-row RoPE start offsets for **ragged batched decode** (a `[B]` array). When set, the
    /// Llama `Attention` ropes q/k at `cache.offset() − pad_offsets` (per row) via
    /// `RopeVariant::forward_dynamic`, so a left-padded batch of different-length sequences is
    /// rotated at each row's true position. `None` (default) → the normal single-offset path, so
    /// B=1 is byte-identical. Set before a batched forward + cleared after; the per-row key-pad
    /// mask is threaded separately via `AttentionInput.mask`. (Independent of `qwen3`'s identical
    /// thread-local — the worker sets both; only the loaded arch's attention reads its own.)
    static BATCH_PAD_OFFSETS: std::cell::RefCell<Option<Array>> =
        const { std::cell::RefCell::new(None) };
}

/// Install (or clear with `None`) the per-row RoPE pad offsets for ragged batched decode on the
/// Llama path — see [`BATCH_PAD_OFFSETS`].
pub fn set_batch_pad_offsets(offsets: Option<Array>) {
    BATCH_PAD_OFFSETS.with(|c| *c.borrow_mut() = offsets);
}

use crate::{
    cache::KeyValueCache,
    error::Error,
    // The sampler is model-agnostic (operates on logit arrays); reuse qwen3's so
    // Llama gets top-p/top-k/repeat-penalty parity without duplicating it.
    models::qwen3::{repeat_window, sample_with, QuantizationConfig, SamplerOpts},
    utils::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
};

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
    /// Per-head dim. Llama configs set it explicitly; Mistral / Mistral-Nemo omit it (it's
    /// `hidden_size / num_attention_heads`), so it's optional with that standard default.
    #[serde(default)]
    pub head_dim: Option<i32>,
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

        let head_dim = args
            .head_dim
            .unwrap_or(args.hidden_size / args.num_attention_heads);
        let scale = (head_dim as f32).sqrt().recip();

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
            .bias(args.attention_bias)
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
            // Ragged batched decode: rope each row at `cache.offset() − pad_i` (its true
            // position); else the normal shared scalar offset.
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

pub fn load_llama_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

pub fn get_llama_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
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

pub fn load_llama_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = get_llama_model_args(model_dir)?;
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
        eprintln!("LOADED {matched} params (llama)");
    }
    model.eval()?;

    Ok(model)
}

/// Split `key` of the form `<prefix>.<name>.<suffix>` into `(prefix, suffix)`, or `None`.
fn split_fused_key<'a>(key: &'a str, name: &str) -> Option<(&'a str, &'a str)> {
    let needle = format!(".{name}.");
    let idx = key.find(&needle)?;
    Some((&key[..idx], &key[idx + needle.len()..]))
}

/// Phi-3 (`model_type: "phi3"`) loads into the Llama [`Model`]: it IS the Llama architecture
/// (RMSNorm, RoPE, SwiGLU, GQA, no qkv-bias) but ships FUSED projections — one `qkv_proj` and
/// one `gate_up_proj` per layer instead of separate `q/k/v_proj` + `gate/up_proj`. We split each
/// fused tensor along the OUTPUT axis into the separate weights the Llama structure expects, then
/// load as usual. The 4-bit AFQ packing is along the INPUT axis, so row-slicing the
/// weight/scales/biases is exact — no unpacking. Reuses every Llama block, the forward, and
/// `Generate`, so Phi-3 needs no new model file or runtime path. (Phi-3-mini-4k: full RoPE, no
/// `rope_scaling`; the 128k `su`/longrope variant would need the scaling threaded — separate.)
pub fn load_phi3_model(model_dir: impl AsRef<Path>) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let args = get_llama_model_args(model_dir)?;
    let quantization = args.quantization.clone();
    let head_dim = args
        .head_dim
        .unwrap_or(args.hidden_size / args.num_attention_heads);
    let q_out = args.num_attention_heads * head_dim;
    let kv_out = args.num_key_value_heads * head_dim;
    let inter = args.intermediate_size;
    let mut model = Model::new(args)?;
    if let Some(q) = quantization {
        model = mlx_rs::nn::quantize(model, q.group_size, q.bits)?;
    }

    // Gather every weight tensor (sharded or single-file).
    let mut loaded: HashMap<String, Array> = HashMap::new();
    let index = model_dir.join("model.safetensors.index.json");
    let files: Vec<std::path::PathBuf> = if index.exists() {
        let json = std::fs::read_to_string(&index)?;
        let wm: WeightMap = serde_json::from_str(&json)?;
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

    // Split the fused projections (q/k/v stacked on the output axis; gate/up stacked).
    let row = |a: &Array, s: i32, e: i32| a.index((s..e, ..));
    let mut split: HashMap<String, Array> = HashMap::new();
    for (k, v) in loaded {
        if let Some((pre, suf)) = split_fused_key(&k, "qkv_proj") {
            split.insert(format!("{pre}.q_proj.{suf}"), row(&v, 0, q_out));
            split.insert(format!("{pre}.k_proj.{suf}"), row(&v, q_out, q_out + kv_out));
            split.insert(
                format!("{pre}.v_proj.{suf}"),
                row(&v, q_out + kv_out, q_out + 2 * kv_out),
            );
        } else if let Some((pre, suf)) = split_fused_key(&k, "gate_up_proj") {
            split.insert(format!("{pre}.gate_proj.{suf}"), row(&v, 0, inter));
            split.insert(format!("{pre}.up_proj.{suf}"), row(&v, inter, 2 * inter));
        } else {
            split.insert(k, v);
        }
    }

    // Remap quantized `<prefix>.weight` -> `<prefix>.inner.weight` (scales sibling), then load.
    let has_scales: HashSet<String> = split
        .keys()
        .filter_map(|k| k.strip_suffix(".scales").map(str::to_string))
        .collect();
    let mut params = model.parameters_mut().flatten();
    let mut matched = 0usize;
    for (key, value) in split {
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
        eprintln!("LOADED {matched} params (phi3, fused-split)");
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

#[cfg(test)]
mod tests {
    use std::{env::home_dir, fs};

    use lazy_static::lazy_static;
    use mlx_rs::{
        ops::indexing::{IndexOp, NewAxis},
        transforms::eval,
        Array,
    };

    use crate::{
        cache::ConcatKeyValueCache,
        models::llama::{load_llama_model, load_llama_tokenizer},
    };

    /// Resolve the HuggingFace cache directory to the actual snapshot path.
    /// The structure is:
    ///   models--<org>--<name>/
    ///     refs/
    ///       main  (contains the commit hash)
    ///     snapshots/
    ///       <commit_hash>/  (actual model files)
    fn resolve_hf_cache_dir(model_cache_dir: &str) -> String {
        let refs_main = std::path::Path::new(model_cache_dir)
            .join("refs")
            .join("main");
        let commit_hash = fs::read_to_string(&refs_main)
            .unwrap_or_default()
            .trim()
            .to_string();
        std::path::Path::new(model_cache_dir)
            .join("snapshots")
            .join(commit_hash)
            .to_string_lossy()
            .into_owned()
    }

    lazy_static! {
        static ref CACHED_TEST_MODEL_DIR: String = {
            let cache_dir = home_dir()
                .map(|p| {
                    p.join(".cache")
                        .join("huggingface")
                        .join("hub")
                        .join("models--meta-llama--Llama-3.2-1B-Instruct")
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default();

            resolve_hf_cache_dir(&cache_dir)
        };
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_llama_model() {
        use mlx_rs::module::ModuleParameters;

        let model_dir = CACHED_TEST_MODEL_DIR.as_str();
        let model_args = super::get_llama_model_args(model_dir).unwrap();
        let model = super::Model::new(model_args).unwrap();

        // Print some model parameter keys
        let params = model.parameters().flatten();
        let mut param_keys: Vec<_> = params.keys().map(|k| k.to_string()).collect();
        param_keys.sort();
        println!("=== Model parameter keys (first 20) ===");
        for key in param_keys.iter().take(20) {
            println!("  {key}");
        }

        // Print some safetensor keys
        let weights_path = std::path::Path::new(model_dir).join("model.safetensors");
        let loaded = mlx_rs::Array::load_safetensors(&weights_path).unwrap();
        let mut weight_keys: Vec<_> = loaded.keys().map(|k| k.to_string()).collect();
        weight_keys.sort();
        println!("=== Safetensor weight keys (first 20) ===");
        for key in weight_keys.iter().take(20) {
            println!("  {key}");
        }

        // Find unmatched keys
        let param_set: std::collections::HashSet<_> = param_keys.iter().collect();
        let weight_set: std::collections::HashSet<_> = weight_keys.iter().collect();
        let unloaded: Vec<_> = weight_set.difference(&param_set).collect();
        let missing: Vec<_> = param_set.difference(&weight_set).collect();
        println!(
            "=== Weight keys NOT in model params ({}) ===",
            unloaded.len()
        );
        for key in unloaded.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "=== Model param keys NOT in weights ({}) ===",
            missing.len()
        );
        for key in missing.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "Total model params: {}, Total weight keys: {}",
            param_keys.len(),
            weight_keys.len()
        );
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();

        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_llama_with_concat_cache() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();
        let mut model = load_llama_model(CACHED_TEST_MODEL_DIR.as_str()).unwrap();

        let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
        let mut cache = Vec::new();

        let eos_token_id = 128001u32;
        let eot_token_id = 128009u32;

        let mut token_ids = Vec::new();
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model,
            &mut cache,
            0.0,
            &prompt_tokens,
        );
        for (token, _ntoks) in generate.zip(0..50) {
            let token = token.unwrap();
            eval([&token]).unwrap();
            let token_id = token.item::<u32>();
            print!("[{}]", token_id);
            if token_id == eos_token_id || token_id == eot_token_id {
                break;
            }
            token_ids.push(token_id);
        }
        println!();

        let output = tokenizer.decode(&token_ids, true).unwrap();
        println!("Response: {output}");
        println!("------");
    }
}
