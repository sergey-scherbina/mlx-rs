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
use mlx_lm::models::qwen3_5_vision::{load_vision_tower, mrope_cos_sin, rope_index_single_image};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use tokenizers::Tokenizer;

const IMAGE_TOKEN_ID: u32 = 248056;
const IM_END_ID: u32 = 151645; // <|im_end|> (fallback stop; resolved from tokenizer below)
const ROTARY_DIM: i32 = 64; // head_dim(256) * partial_rotary_factor(0.25)
const ROPE_THETA: f32 = 10_000_000.0;
const SPATIAL_MERGE: i32 = 2;

fn argmax_last(logits: &Array) -> Result<u32, Box<dyn std::error::Error>> {
    // logits: [1, L, vocab] -> last position -> host argmax
    let last = logits.index((.., -1, ..)).as_dtype(mlx_rs::Dtype::Float32)?;
    last.eval()?;
    let data = last.as_slice::<f32>();
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in data.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    Ok(best as u32)
}

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

    // --- vision features ---
    let loaded = mlx_rs::Array::load_safetensors(pv_path)?;
    let pv = loaded.get("pixel_values").ok_or("missing pixel_values")?;
    let img_embeds = vision.forward(pv, (t, h, w))?; // [n_img, hidden]
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
    let (positions, mut next_pos) =
        rope_index_single_image(&tokens, IMAGE_TOKEN_ID, (t, h, w), SPATIAL_MERGE);
    let (cos, sin) = mrope_cos_sin(&positions, ROTARY_DIM, ROPE_THETA);
    let seq = tokens.len() as i32;
    let cos_a = Array::from_slice(&cos, &[seq, ROTARY_DIM]);
    let sin_a = Array::from_slice(&sin, &[seq, ROTARY_DIM]);

    // --- prefill (single pass) with splice + mrope ---
    let mut cache = model.init_cache();
    let input_ids: Vec<i32> = tokens.iter().map(|&x| x as i32).collect();
    let inp = Array::from_slice(&input_ids, &[1, seq]);
    qwen3_5::set_mm_splice(Some((img_embeds.clone(), img_start)));
    qwen3_5::set_mrope_cossin(Some((cos_a, sin_a)));
    let logits = model.forward(&inp, &mut cache)?;
    qwen3_5::set_mm_splice(None);
    let mut next = argmax_last(&logits)?;
    qwen3_5::set_mrope_cossin(None);

    // --- greedy decode ---
    print!("\n=== ANSWER ===\n");
    let mut out_ids: Vec<u32> = Vec::new();
    for _ in 0..128 {
        if next == im_end {
            break;
        }
        out_ids.push(next);
        // next token position (all axes equal for generated text)
        let (dcos, dsin) = mrope_cos_sin(&[(next_pos, next_pos, next_pos)], ROTARY_DIM, ROPE_THETA);
        next_pos += 1;
        qwen3_5::set_mrope_cossin(Some((
            Array::from_slice(&dcos, &[1, ROTARY_DIM]),
            Array::from_slice(&dsin, &[1, ROTARY_DIM]),
        )));
        let step_in = Array::from_slice(&[next as i32], &[1, 1]);
        let logits = model.forward(&step_in, &mut cache)?;
        qwen3_5::set_mrope_cossin(None);
        next = argmax_last(&logits)?;
    }
    let text = tok.decode(&out_ids, true).unwrap_or_default();
    println!("{text}");
    println!("\n=== ({} tokens) ===", out_ids.len());
    Ok(())
}
