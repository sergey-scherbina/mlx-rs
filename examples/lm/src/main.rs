use std::path::PathBuf;
use std::time::Instant;

use mlx_lm::{cache::ConcatKeyValueCache, models::qwen3::load_qwen3_model};
use mlx_lm_utils::tokenizer::{
    load_model_chat_template_from_file, ApplyChatTemplateArgs, Conversation, Role, Tokenizer,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
    Array,
};

const EOS: u32 = 151645; // Qwen3 <|im_end|>
const MAX_TOKENS: usize = 256;

fn main() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var("ROZUM_MODEL_DIR").expect("set ROZUM_MODEL_DIR"));
    let model_id =
        std::env::var("ROZUM_MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3-4B-4bit".into());
    let prompt_text = std::env::var("ROZUM_PROMPT").unwrap_or_else(|_| {
        "Q: What is the capital of France? Answer in one short sentence. A:".into()
    });

    let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let chat_template = load_model_chat_template_from_file(model_dir.join("tokenizer_config.json"))?
        .expect("chat template not found");

    let conversations = vec![Conversation {
        role: Role::User,
        content: &prompt_text,
    }];
    let args = ApplyChatTemplateArgs {
        conversations: vec![conversations.into()],
        documents: None,
        model_id: &model_id,
        chat_template_id: None,
        add_generation_prompt: Some(true),
        continue_final_message: None,
    };
    let encodings = tokenizer.apply_chat_template_and_encode(chat_template, args)?;
    let prompt: Vec<u32> = encodings.iter().flat_map(|e| e.get_ids()).copied().collect();
    let n_prompt = prompt.len();
    let prompt_tokens = Array::from(&prompt[..]).index(NewAxis);

    let t_load = Instant::now();
    let mut model = load_qwen3_model(&model_dir)?;
    eprintln!(
        "loaded in {:.2}s, prompt {} tokens",
        t_load.elapsed().as_secs_f64(),
        n_prompt
    );

    let mut cache = Vec::new();
    let generate = mlx_lm::models::qwen3::Generate::<ConcatKeyValueCache>::new(
        &mut model,
        &mut cache,
        0.0,
        &prompt_tokens,
    );

    let mut ids = Vec::new();
    let mut t_first = None;
    let t_start = Instant::now();
    for token in generate.take(MAX_TOKENS) {
        let token = token?;
        eval([&token])?; // realistic streaming: force per-token compute
        if t_first.is_none() {
            t_first = Some(t_start.elapsed());
        }
        let id = token.item::<u32>();
        if id == EOS {
            break;
        }
        ids.push(id);
    }
    let total = t_start.elapsed();

    let text = tokenizer
        .decode(&ids, true)
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;
    let ttft = t_first.unwrap_or(total).as_secs_f64();
    let decode_n = ids.len().saturating_sub(1);
    let decode_tps = decode_n as f64 / (total.as_secs_f64() - ttft).max(1e-6);

    println!("=== OUTPUT ===\n{text}\n=== /OUTPUT ===");
    eprintln!(
        "decode: {} tokens, {:.2} T/s (ttft {:.2}s, total {:.2}s)",
        ids.len(),
        decode_tps,
        ttft,
        total.as_secs_f64()
    );
    Ok(())
}
