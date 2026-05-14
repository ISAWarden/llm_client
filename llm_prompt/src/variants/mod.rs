mod api_prompt;
mod local_prompt;

pub use api_prompt::ApiPrompt;
pub use local_prompt::{LocalPrompt, apply_chat_template};
