//! Parity probe for the Qwen3.5 vision tower.
//!
//! Loads the vision tower, runs it on a `pixel_values` array produced by the
//! Python `transformers` oracle, and writes the merged image embeddings as a
//! flat little-endian f32 blob for comparison.
//!
//! Usage:
//!   cargo run --release --example qwen3_5_vision_probe -- \
//!       <model_dir> <pixel_values.safetensors> <t> <h> <w> <out.f32>

use mlx_lm::models::qwen3_5_vision::load_vision_tower;
use mlx_rs::Dtype;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!("usage: <model_dir> <pixel_values.safetensors> <t> <h> <w> <out.f32>");
        std::process::exit(2);
    }
    let model_dir = &a[1];
    let pv_path = &a[2];
    let (t, h, w): (i32, i32, i32) = (a[3].parse()?, a[4].parse()?, a[5].parse()?);
    let out_path = &a[6];

    let mut vision = load_vision_tower(model_dir)?;
    let loaded = mlx_rs::Array::load_safetensors(pv_path)?;
    let pv = loaded
        .get("pixel_values")
        .ok_or("missing 'pixel_values' key in safetensors")?;
    eprintln!("pixel_values shape={:?} dtype={:?}", pv.shape(), pv.dtype());

    let out = vision.forward(pv, (t, h, w))?;
    let out = out.as_dtype(Dtype::Float32)?;
    out.eval()?;
    let shape = out.shape().to_vec();
    let data = out.as_slice::<f32>();
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(out_path, &bytes)?;
    println!(
        "VISION_OUT shape={:?} n={} first8={:?}",
        shape,
        data.len(),
        &data[..data.len().min(8)]
    );
    Ok(())
}
