//! The browser build has no HTTP client and can't keep an API key secret, so the Assistant says so
//! instead of pretending to reach DeepSeek.

use std::sync::atomic::AtomicBool;

use crate::chat::{ChatModel, ChatRequest, ChatResponse, ModelError, ModelEvent};
use crate::config::DeepSeekConfig;

pub struct DeepSeek {
    _config: DeepSeekConfig,
}

impl DeepSeek {
    pub fn new(config: DeepSeekConfig) -> Self {
        Self { _config: config }
    }

    pub fn config(&self) -> &DeepSeekConfig {
        &self._config
    }
}

impl ChatModel for DeepSeek {
    fn complete(&self, _request: &ChatRequest, _on_event: &mut dyn FnMut(ModelEvent), _cancel: &AtomicBool) -> Result<ChatResponse, ModelError> {
        Err(ModelError::Unsupported("The Assistant needs the desktop app or figcraft-cli: the browser build has no DeepSeek client.".into()))
    }
}
