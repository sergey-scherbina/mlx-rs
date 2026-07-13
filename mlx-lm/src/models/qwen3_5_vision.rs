//! Qwen3.5 vision tower (the ViT half of `Qwen3_5ForConditionalGeneration`).
//!
//! Port of transformers `Qwen3_5VisionModel` (see `docs/ref_qwen3_5.py` +
//! `docs/vu_vision_utils.py`). Standalone: it is NOT nested in the text
//! `qwen3_5::Model` param tree — the rozum multimodal path loads it separately
//! and splices its output onto the image-token positions in the text stream.
//!
//! Architecture (from `config.json` `vision_config`):
//!   - patch_embed: `Conv3d(3, 1024, kernel=stride=[2,16,16])`. Since kernel ==
//!     stride == the full patch, it is exactly a `Linear(3*2*16*16=1536, 1024)`
//!     over pre-flattened patch vectors — so we load the conv weight reshaped.
//!   - pos_embed: learned `Embedding(2304, 1024)`, bilinear-interpolated from the
//!     48x48 grid to the actual (h,w) grid.
//!   - 24 blocks: `LayerNorm -> attn (2-axis h/w rope, full attention) -> LayerNorm
//!     -> gelu-tanh MLP`.
//!   - merger: `LayerNorm(1024) -> reshape to 4x -> Linear(4096,4096) -> GELU ->
//!     Linear(4096, 2560)`. One 2x2 spatial block -> one 2560-dim LLM token.
//!
//! Vision attention is full (per image, no windowing). We currently support a
//! SINGLE image per forward (one full-attention segment); multi-image packing
//! (cu_seqlens block-diagonal masking) is a follow-up.

use std::{collections::HashSet, path::Path};

use mlx_rs::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParameters, ModuleParametersExt},
    nn,
    ops::indexing::IndexOp,
    Array, Dtype,
};
use serde::Deserialize;

use crate::error::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: i32,
    pub hidden_size: i32,
    pub num_heads: i32,
    pub intermediate_size: i32,
    pub patch_size: i32,
    pub spatial_merge_size: i32,
    pub temporal_patch_size: i32,
    pub in_channels: i32,
    pub out_hidden_size: i32,
    pub num_position_embeddings: i32,
}

#[derive(Debug, Clone, Deserialize)]
struct WrappedVisionConfig {
    vision_config: VisionConfig,
}

const VISION_ROPE_THETA: f32 = 10000.0;
const LN_EPS: f32 = 1e-6;

// ---------------------------------------------------------------------------
// Neural modules (param field names mirror the checkpoint key layout under
// `vision_tower.`).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionPatchEmbed {
    #[param]
    pub proj: nn::Linear, // weight [hidden, in_ch*temporal*patch*patch]
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionMlp {
    #[param]
    pub linear_fc1: nn::Linear,
    #[param]
    pub linear_fc2: nn::Linear,
}

impl VisionMlp {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // gelu_pytorch_tanh
        let h = nn::gelu_approximate(self.linear_fc1.forward(x)?)?;
        self.linear_fc2.forward(&h)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionAttention {
    #[param]
    pub qkv: nn::Linear,
    #[param]
    pub proj: nn::Linear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl VisionAttention {
    /// `x`: `[seq, hidden]`. `cos`/`sin`: `[seq, head_dim]` (f32). Full attention.
    fn forward(&mut self, x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
        let seq = x.shape()[0];
        let qkv = self.qkv.forward(x)?; // [seq, 3*hidden]
        let qkv = qkv.reshape(&[seq, 3, self.num_heads, self.head_dim])?;
        // [seq, num_heads, head_dim] each
        let q = qkv.index((.., 0, .., ..));
        let k = qkv.index((.., 1, .., ..));
        let v = qkv.index((.., 2, .., ..));

        // rope in f32 (matches reference `apply_rotary_pos_emb_vision` .float()).
        // cos/sin: [seq, head_dim] -> [seq, 1, head_dim] to broadcast over heads.
        let cos = cos.reshape(&[seq, 1, self.head_dim])?;
        let sin = sin.reshape(&[seq, 1, self.head_dim])?;
        let q = apply_rope(&q.as_dtype(Dtype::Float32)?, &cos, &sin)?;
        let k = apply_rope(&k.as_dtype(Dtype::Float32)?, &cos, &sin)?;
        let v = v.as_dtype(Dtype::Float32)?;

        // [num_heads, seq, head_dim]
        let q = q.transpose_axes(&[1, 0, 2])?;
        let k = k.transpose_axes(&[1, 0, 2])?;
        let v = v.transpose_axes(&[1, 0, 2])?;

        // scores [num_heads, seq, seq]
        let kt = k.transpose_axes(&[0, 2, 1])?;
        let scores = mlx_rs::ops::matmul(&q, &kt)?.multiply(Array::from_f32(self.scale))?;
        let probs = mlx_rs::ops::softmax_axis(&scores, -1, true)?;
        let out = mlx_rs::ops::matmul(&probs, &v)?; // [num_heads, seq, head_dim]
        let out = out
            .transpose_axes(&[1, 0, 2])?
            .reshape(&[seq, self.num_heads * self.head_dim])?
            .as_dtype(x.dtype())?;
        self.proj.forward(&out)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionBlock {
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,
    #[param]
    pub attn: VisionAttention,
    #[param]
    pub mlp: VisionMlp,
}

impl VisionBlock {
    fn forward(&mut self, x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
        let a = self.attn.forward(&self.norm1.forward(x)?, cos, sin)?;
        let x = x.add(&a)?;
        let m = self.mlp.forward(&self.norm2.forward(&x)?)?;
        x.add(&m)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionPatchMerger {
    #[param]
    pub norm: nn::LayerNorm, // over hidden_size
    #[param]
    pub linear_fc1: nn::Linear, // hidden*merge^2 -> hidden*merge^2
    #[param]
    pub linear_fc2: nn::Linear, // hidden*merge^2 -> out_hidden
    merged_dim: i32,
}

impl VisionPatchMerger {
    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let x = self.norm.forward(x)?; // per-patch LN over hidden
        let x = x.reshape(&[-1, self.merged_dim])?; // group 4 consecutive patches
        let x = nn::gelu(self.linear_fc1.forward(&x)?)?; // exact GELU (nn.GELU())
        self.linear_fc2.forward(&x)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct VisionModel {
    #[param]
    pub patch_embed: VisionPatchEmbed,
    #[param]
    pub pos_embed: nn::Embedding,
    #[param]
    pub blocks: Vec<VisionBlock>,
    #[param]
    pub merger: VisionPatchMerger,

    // config-derived scalars (not params)
    spatial_merge_size: i32,
    num_grid_per_side: i32,
    head_dim: i32,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig) -> Result<Self, Exception> {
        let h = cfg.hidden_size;
        let head_dim = h / cfg.num_heads;
        let patch_in = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let merged_dim = h * cfg.spatial_merge_size * cfg.spatial_merge_size;

        let patch_embed = VisionPatchEmbed {
            proj: nn::LinearBuilder::new(patch_in, h).bias(true).build()?,
        };
        let pos_embed = nn::Embedding::new(cfg.num_position_embeddings, h)?;

        let mut blocks = Vec::with_capacity(cfg.depth as usize);
        for _ in 0..cfg.depth {
            blocks.push(VisionBlock {
                norm1: nn::LayerNormBuilder::new(h).eps(LN_EPS).build()?,
                norm2: nn::LayerNormBuilder::new(h).eps(LN_EPS).build()?,
                attn: VisionAttention {
                    qkv: nn::LinearBuilder::new(h, h * 3).bias(true).build()?,
                    proj: nn::LinearBuilder::new(h, h).bias(true).build()?,
                    num_heads: cfg.num_heads,
                    head_dim,
                    scale: (head_dim as f32).powf(-0.5),
                },
                mlp: VisionMlp {
                    linear_fc1: nn::LinearBuilder::new(h, cfg.intermediate_size)
                        .bias(true)
                        .build()?,
                    linear_fc2: nn::LinearBuilder::new(cfg.intermediate_size, h)
                        .bias(true)
                        .build()?,
                },
            });
        }

        let merger = VisionPatchMerger {
            norm: nn::LayerNormBuilder::new(h).eps(LN_EPS).build()?,
            linear_fc1: nn::LinearBuilder::new(merged_dim, merged_dim)
                .bias(true)
                .build()?,
            linear_fc2: nn::LinearBuilder::new(merged_dim, cfg.out_hidden_size)
                .bias(true)
                .build()?,
            merged_dim,
        };

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            spatial_merge_size: cfg.spatial_merge_size,
            num_grid_per_side: (cfg.num_position_embeddings as f64).sqrt() as i32,
            head_dim,
        })
    }

    /// `pixel_values`: `[seq, in_ch*temporal*patch*patch]` (patch vectors in
    /// block-major order, matching the image processor). `grid_thw`: `(t,h,w)`
    /// patch-grid dims for the single image. Returns merged LLM tokens
    /// `[seq/merge^2, out_hidden]`.
    pub fn forward(&mut self, pixel_values: &Array, grid_thw: (i32, i32, i32)) -> Result<Array, Exception> {
        let (t, h, w) = grid_thw;
        let m = self.spatial_merge_size;
        let seq = t * h * w;

        // Match reference `pixel_values.type(self.visual.dtype)` (bf16 checkpoint).
        let pixel_values = pixel_values.as_dtype(Dtype::Bfloat16)?;
        let mut hidden = self.patch_embed.proj.forward(&pixel_values)?; // [seq, hidden]

        // --- learned pos-emb, bilinear-interpolated from the 48x48 grid ---
        let (bidx, bwt) = bilinear_indices_and_weights(t, h, w, self.num_grid_per_side, m);
        let mut pos = None::<Array>;
        for corner in 0..4 {
            let idx = Array::from_slice(&bidx[corner], &[seq]);
            let emb = self.pos_embed.forward(&idx)?.as_dtype(Dtype::Float32)?; // [seq, hidden]
            let wt = Array::from_slice(&bwt[corner], &[seq, 1]);
            let contrib = emb.multiply(&wt)?;
            pos = Some(match pos {
                Some(p) => p.add(&contrib)?,
                None => contrib,
            });
        }
        let pos = pos.unwrap().as_dtype(hidden.dtype())?;
        hidden = hidden.add(&pos)?;

        // --- 2-axis (h,w) vision rope: host-computed cos/sin [seq, head_dim] ---
        let (cos, sin) = vision_rope_cos_sin(t, h, w, m, self.head_dim);
        let cos = Array::from_slice(&cos, &[seq, self.head_dim]);
        let sin = Array::from_slice(&sin, &[seq, self.head_dim]);

        for blk in &mut self.blocks {
            hidden = blk.forward(&hidden, &cos, &sin)?;
        }

        self.merger.forward(&hidden)
    }
}

// ---------------------------------------------------------------------------
// rope application (rotate_half convention)
// ---------------------------------------------------------------------------

fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    // x: [seq, heads, head_dim]; cos/sin: [seq, 1, head_dim]
    let hd = x.shape()[2];
    let half = hd / 2;
    let x1 = x.index((.., .., 0..half));
    let x2 = x.index((.., .., half..hd));
    let rot = mlx_rs::ops::concatenate_axis(&[&x2.multiply(Array::from_f32(-1.0))?, &x1], -1)?;
    x.multiply(cos)?.add(&rot.multiply(sin)?)
}

// ---------------------------------------------------------------------------
// Host-side layout helpers (mirror transformers `vision_utils`).
// ---------------------------------------------------------------------------

/// Block-major (h,w) position ids over 2x2 merge blocks, for a single (t,h,w).
/// Returns `Vec<(hpos, wpos)>` of length `t*h*w` (h/w repeated over t).
fn grid_layout(t: i32, h: i32, w: i32, m: i32) -> Vec<(i32, i32)> {
    let mut out = Vec::with_capacity((t * h * w) as usize);
    let hb = h / m;
    let wb = w / m;
    for _ in 0..t {
        for bh in 0..hb {
            for bw in 0..wb {
                for ih in 0..m {
                    for iw in 0..m {
                        out.push((bh * m + ih, bw * m + iw));
                    }
                }
            }
        }
    }
    out
}

/// cos/sin `[seq, head_dim]` flattened for the 2-axis vision rope. Matches
/// `Qwen3_5VisionRotaryEmbedding(head_dim//2)` + `emb = cat(rope, rope)`.
fn vision_rope_cos_sin(t: i32, h: i32, w: i32, m: i32, head_dim: i32) -> (Vec<f32>, Vec<f32>) {
    let layout = grid_layout(t, h, w, m);
    let rot_dim = head_dim / 2; // = 32; inv_freq over arange(0, rot_dim, 2)/rot_dim
    let n_freq = (rot_dim / 2) as usize; // 16
    let inv_freq: Vec<f32> = (0..n_freq)
        .map(|i| 1.0f32 / VISION_ROPE_THETA.powf((2 * i) as f32 / rot_dim as f32))
        .collect();

    let seq = layout.len();
    let mut cos = vec![0f32; seq * head_dim as usize];
    let mut sin = vec![0f32; seq * head_dim as usize];
    for (p, &(hp, wp)) in layout.iter().enumerate() {
        // freqs = [hp*inv_freq (16), wp*inv_freq (16)] -> 32; emb = [freqs, freqs] -> 64
        let mut emb = vec![0f32; head_dim as usize];
        for (i, &f) in inv_freq.iter().enumerate() {
            let fh = hp as f32 * f;
            let fw = wp as f32 * f;
            emb[i] = fh; // first 16: h freqs
            emb[n_freq + i] = fw; // next 16: w freqs
            emb[rot_dim as usize + i] = fh; // duplicate for the second half
            emb[rot_dim as usize + n_freq + i] = fw;
        }
        let base = p * head_dim as usize;
        for j in 0..head_dim as usize {
            cos[base + j] = emb[j].cos();
            sin[base + j] = emb[j].sin();
        }
    }
    (cos, sin)
}

/// Bilinear interpolation indices/weights into the `side x side` learned pos-emb
/// grid for each block-major output patch. Returns `([4][seq] i32, [4][seq] f32)`.
fn bilinear_indices_and_weights(
    t: i32,
    h: i32,
    w: i32,
    side: i32,
    m: i32,
) -> (Vec<Vec<i32>>, Vec<Vec<f32>>) {
    // linspace(0, side-1, n)
    let lin = |n: i32| -> Vec<f32> {
        if n == 1 {
            return vec![0.0];
        }
        (0..n)
            .map(|i| i as f32 * (side - 1) as f32 / (n - 1) as f32)
            .collect()
    };
    let hg = lin(h);
    let wg = lin(w);

    let floor = |g: &[f32]| -> Vec<i32> { g.iter().map(|&v| v as i32).collect() };
    let h_floor = floor(&hg);
    let w_floor = floor(&wg);
    let ceil = |f: &[i32]| -> Vec<i32> { f.iter().map(|&v| (v + 1).min(side - 1)).collect() };
    let h_ceil = ceil(&h_floor);
    let w_ceil = ceil(&w_floor);
    let frac = |g: &[f32], f: &[i32]| -> Vec<f32> {
        g.iter().zip(f).map(|(&v, &fl)| v - fl as f32).collect()
    };
    let h_frac = frac(&hg, &h_floor);
    let w_frac = frac(&wg, &w_floor);

    let layout = grid_layout(t, h, w, m);
    let seq = layout.len();
    let mut idx = vec![vec![0i32; seq]; 4];
    let mut wt = vec![vec![0f32; seq]; 4];
    for (p, &(hp, wp)) in layout.iter().enumerate() {
        let (hp, wp) = (hp as usize, wp as usize);
        let hf = h_floor[hp];
        let hc = h_ceil[hp];
        let wf = w_floor[wp];
        let wc = w_ceil[wp];
        let hfr = h_frac[hp];
        let wfr = w_frac[wp];
        // corners: (floor,floor) (floor,ceil) (ceil,floor) (ceil,ceil)
        idx[0][p] = hf * side + wf;
        idx[1][p] = hf * side + wc;
        idx[2][p] = hc * side + wf;
        idx[3][p] = hc * side + wc;
        wt[0][p] = (1.0 - hfr) * (1.0 - wfr);
        wt[1][p] = (1.0 - hfr) * wfr;
        wt[2][p] = hfr * (1.0 - wfr);
        wt[3][p] = hfr * wfr;
    }
    (idx, wt)
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load the vision tower from a multimodal Qwen3.5 checkpoint dir. Reads only the
/// `vision_tower.*` weights (reshaping the patch-embed conv weight to the Linear
/// layout). bf16 checkpoint expected (no quantization on the vision tower yet).
pub fn load_vision_tower(model_dir: impl AsRef<Path>) -> Result<VisionModel, Error> {
    let model_dir = model_dir.as_ref();
    let text = std::fs::read_to_string(model_dir.join("config.json"))?;
    let cfg: VisionConfig = serde_json::from_str::<WrappedVisionConfig>(&text)?.vision_config;
    let patch_in =
        cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let mut model = VisionModel::new(&cfg)?;

    fn load_shard(
        model: &mut VisionModel,
        file: &Path,
        patch_in: i32,
    ) -> Result<usize, Error> {
        let loaded = mlx_rs::Array::load_safetensors(file)?;
        let mut params = model.parameters_mut().flatten();
        let param_keys: HashSet<String> = params.keys().map(|k| k.to_string()).collect();
        let mut matched = 0usize;
        for (key, mut value) in loaded {
            let key = match key.strip_prefix("vision_tower.") {
                Some(k) => k.to_string(),
                None => continue, // text / other weights: skip
            };
            // patch-embed conv weight. The mlx-community checkpoint stores it
            // channels-LAST: [out, T, ph, pw, C]. The pixel_values (and the
            // reference conv) are channels-FIRST [C, T, ph, pw], so move C from
            // the last axis to position 1 before flattening to the Linear layout
            // [out, C*T*ph*pw].
            if key == "patch_embed.proj.weight" && value.ndim() == 5 {
                let out = value.shape()[0];
                value = value.move_axis(4, 1)?.reshape(&[out, patch_in])?;
            }
            if !param_keys.contains(&key) {
                continue;
            }
            if let Some(param) = params.get_mut(key.as_str()) {
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
        let map: serde_json::Value = serde_json::from_str(&json)?;
        let files: HashSet<String> = map
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|o| o.values().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        for f in files {
            matched += load_shard(&mut model, &model_dir.join(f), patch_in)?;
        }
    } else {
        let single = model_dir.join("model.safetensors");
        matched += load_shard(&mut model, &single, patch_in)?;
    }
    if std::env::var("ROZUM_MLX_DEBUG").is_ok() {
        eprintln!("LOADED {matched} vision params (qwen3_5_vision)");
    }
    model.eval()?;
    Ok(model)
}
