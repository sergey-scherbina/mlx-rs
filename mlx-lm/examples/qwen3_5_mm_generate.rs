//! End-to-end multimodal generation probe for Qwen3.5-VL.
//!
//! Loads the text stack + vision tower, builds a single-image chat prompt,
//! splices the vision embeddings onto the image-token block, applies 3D
//! interleaved M-RoPE, and greedily decodes an answer.
//!
//! Usage:
//!   cargo run --release --example qwen3_5_mm_generate -- \
//!       <model_dir> <pixel_values.safetensors> <t> <h> <w> [prompt]

use mlx_lm::models::qwen3_5::{self, Model};
use mlx_lm::models::qwen3_5_vision::{
    load_vision_tower, mrope_cos_sin, patchify_normalize, rope_index_single_image, smart_resize,
};
use mlx_rs::Array;
use tokenizers::Tokenizer;

const IMAGE_TOKEN_ID: u32 = 248056;
const IM_END_ID: u32 = 151645; // <|im_end|> (fallback stop; resolved from tokenizer below)
const ROTARY_DIM: i32 = 64; // head_dim(256) * partial_rotary_factor(0.25)
const ROPE_THETA: f32 = 10_000_000.0;
const SPATIAL_MERGE: i32 = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: <model_dir> <pixel_values.safetensors> <t> <h> <w> [prompt]");
        std::process::exit(2);
    }
    let model_dir = &a[1];
    let pv_path = &a[2];
    let (t, h, w): (i32, i32, i32) = (a[3].parse()?, a[4].parse()?, a[5].parse()?);
    let question = a.get(6).cloned().unwrap_or_else(|| "Describe this image.".into());

    // --- load ---
    eprintln!("loading text model...");
    let mut model: Model = qwen3_5::load_qwen3_5_model(model_dir)?;
    eprintln!("loading vision tower...");
    let mut vision = load_vision_tower(model_dir)?;
    let tok = Tokenizer::from_file(format!("{model_dir}/tokenizer.json")).map_err(|e| format!("tokenizer load: {e}"))?;
    let im_end = tok.token_to_id("<|im_end|>").unwrap_or(IM_END_ID);

    // --- pixel_values: either a precomputed .safetensors, or decode+preprocess
    //     a raw image (.jpg/.png) in Rust ---
    let (pv, grid) = if pv_path.ends_with(".safetensors") {
        let loaded = mlx_rs::Array::load_safetensors(pv_path)?;
        let pv = loaded.get("pixel_values").ok_or("missing pixel_values")?.clone();
        (pv, (t, h, w))
    } else {
        let img = image::open(pv_path)?.to_rgb8();
        let (ow, oh) = (img.width() as i32, img.height() as i32);
        let (hbar, wbar) = smart_resize(oh, ow, 16 * 2, 65536, 16_777_216);
        let resized =
            image::imageops::resize(&img, wbar as u32, hbar as u32, image::imageops::FilterType::CatmullRom);
        let rgb = resized.into_raw();
        let (pixels, (gh, gw)) =
            patchify_normalize(&rgb, hbar, wbar, 16, 2, 2, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
        eprintln!("preprocessed {ow}x{oh} -> {wbar}x{hbar}, grid 1x{gh}x{gw}");
        let pv = Array::from_slice(&pixels, &[gh * gw, 1536]);
        (pv, (1, gh, gw))
    };

    // --- vision features ---
    let img_embeds = vision.forward(&pv, grid)?; // [n_img, hidden]
    img_embeds.eval()?;
    let n_img = img_embeds.shape()[0];
    eprintln!("vision embeds: {:?}", img_embeds.shape());

    // --- build prompt tokens: prefix + [image_token]*n_img + suffix ---
    let prefix = "<|im_start|>user\n<|vision_start|>";
    let suffix = format!("<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n");
    let pre_ids = tok.encode(prefix, false).map_err(|e| format!("encode: {e}"))?.get_ids().to_vec();
    let suf_ids = tok.encode(suffix.as_str(), false).map_err(|e| format!("encode: {e}"))?.get_ids().to_vec();
    let img_start = pre_ids.len() as i32;
    let mut tokens: Vec<u32> = pre_ids;
    tokens.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(n_img as usize));
    tokens.extend(suf_ids);
    eprintln!(
        "prompt tokens: {} (img block [{}, {}))",
        tokens.len(),
        img_start,
        img_start + n_img
    );

    // --- 3D M-RoPE positions ---
    let (positions, next_pos) =
        rope_index_single_image(&tokens, IMAGE_TOKEN_ID, grid, SPATIAL_MERGE);
    let (cos, sin) = mrope_cos_sin(&positions, ROTARY_DIM, ROPE_THETA);
    let seq = tokens.len() as i32;
    let cos_a = Array::from_slice(&cos, &[seq, ROTARY_DIM]);
    let sin_a = Array::from_slice(&sin, &[seq, ROTARY_DIM]);

    // --- generate via the mm-aware Generate iterator (the path rozum-mlx uses) ---
    let input_ids: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
    let inp = Array::from_slice(&input_ids, &[1, seq]);
    let mm = qwen3_5::MmContext::new(
        img_embeds.clone(),
        img_start,
        cos_a,
        sin_a,
        next_pos,
        ROTARY_DIM,
        ROPE_THETA,
    );
    let mut g = qwen3_5::Generate::new(&mut model, 0.0, &inp);
    g.set_mm_context(mm);
    print!("\n=== ANSWER ===\n");
    let mut out_ids: Vec<u32> = Vec::new();
    for _ in 0..128 {
        match g.next() {
            Some(Ok(y)) => {
                y.eval()?;
                let id = y.as_slice::<u32>()[0];
                if id == im_end {
                    break;
                }
                out_ids.push(id);
            }
            Some(Err(e)) => return Err(Box::new(e)),
            None => break,
        }
    }
    let text = tok.decode(&out_ids, true).unwrap_or_default();
    println!("{text}");
    println!("\n=== ({} tokens) ===", out_ids.len());
    Ok(())
}
