//! The chat conversation types shared by the agent loop and every model transport.

use std::sync::atomic::AtomicBool;

use serde_json::Value;

/// Who wrote a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// One function call the model asked for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// The id the next tool message answers with.
    pub id: String,
    pub name: String,
    /// The arguments as the model wrote them: JSON text, which may be malformed.
    pub arguments: String,
}

impl ToolCall {
    /// The arguments as a JSON object; `{}` when the model sent none.
    pub fn arguments_value(&self) -> Result<Value, String> {
        if self.arguments.trim().is_empty() {
            return Ok(serde_json::json!({}));
        }
        let value: Value = serde_json::from_str(&self.arguments).map_err(|e| format!("invalid arguments for `{}`: {e}", self.name))?;
        if value.is_object() { Ok(value) } else { Err(format!("invalid arguments for `{}`: not a JSON object", self.name)) }
    }
}

/// One message in the conversation. `reasoning` is kept for the UI; it is never sent back to the
/// model (DeepSeek reads the reasoning of a turn in the turn itself).
#[derive(Clone, Debug, PartialEq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub reasoning: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning: None }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into(), tool_calls: Vec::new(), tool_call_id: None, reasoning: None }
    }

    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>, reasoning: Option<String>) -> Self {
        Self { role: Role::Assistant, content: content.into(), tool_calls, tool_call_id: None, reasoning }
    }

    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: Role::Tool, content: content.into(), tool_calls: Vec::new(), tool_call_id: Some(call_id.into()), reasoning: None }
    }
}

/// One request to the model: the whole conversation, the tools it may call, and the mode.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// OpenAI-format functions (see [`crate::tools::openai_tools`]).
    pub tools: Vec<Value>,
    /// DeepSeek's thinking mode; tool calls stay available, `tool_choice: auto`.
    pub thinking: bool,
    /// `none`, `low`, `high` or `max` when thinking is on.
    pub reasoning_effort: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: bool,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>, tools: Vec<Value>) -> Self {
        Self { model: model.into(), messages, tools, thinking: true, reasoning_effort: None, max_tokens: None, temperature: None, stream: true }
    }
}

/// Token counts as the model reports them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// One complete answer: text, the reasoning behind it, and any calls it asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
}

/// A fragment streamed before the answer is whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelEvent {
    Content(String),
    Reasoning(String),
}

/// Why a request didn't produce an answer. Messages never carry the API key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// No key was set (the UI asks for one; the CLI names `DEEPSEEK_API_KEY`).
    MissingKey,
    /// The browser build has no transport.
    Unsupported(String),
    /// A connection, TLS or protocol failure.
    Http(String),
    /// The server refused the request; `message` is its (sanitised, bounded) body.
    Status { code: u16, message: String },
    /// The answer wasn't the JSON we expect.
    Json(String),
    /// The user stopped it.
    Cancelled,
    /// The answer was empty.
    Empty,
}

impl ModelError {
    /// Whether trying again could help: transport failures, 429 and 5xx.
    pub fn retryable(&self) -> bool {
        match self {
            ModelError::Status { code, .. } => matches!(code, 408 | 429 | 500 | 502 | 503 | 504),
            ModelError::Http(_) => true,
            _ => false,
        }
    }

    /// The most tokens the endpoint says it takes, when it refused the request only because the
    /// budget was too big ("the valid range of max_tokens is [1, 393216]"). Asking again with this
    /// budget is the fix, and the panel can say so instead of leaving the user with a 400.
    pub fn max_tokens_limit(&self) -> Option<u32> {
        let ModelError::Status { code: 400, message } = self else { return None };
        if !message.contains("max_tokens") {
            return None;
        }
        // The interval a server names after "max_tokens", in either [1, 393216] or (1, 393216] form.
        let start = message.find("max_tokens")?;
        let rest = &message[start..];
        let open = rest.find(['[', '('])?;
        let close = rest[open..].find([']', ')'])? + open;
        let bound = rest[open + 1..close].rsplit(',').next()?.trim().parse::<u32>().ok()?;
        (bound > 0).then_some(bound)
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::MissingKey => write!(f, "no DeepSeek API key: set {0} or type one in the Assistant", crate::config::API_KEY_ENV),
            ModelError::Unsupported(why) => write!(f, "{why}"),
            ModelError::Http(e) => write!(f, "couldn't reach DeepSeek: {e}"),
            ModelError::Status { code, message } if message.is_empty() => write!(f, "DeepSeek refused the request (HTTP {code})"),
            ModelError::Status { code, message } => write!(f, "DeepSeek refused the request (HTTP {code}): {message}"),
            ModelError::Json(e) => write!(f, "DeepSeek sent an answer this app doesn't understand: {e}"),
            ModelError::Cancelled => write!(f, "stopped"),
            ModelError::Empty => write!(f, "DeepSeek sent an empty answer"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Something that answers a [`ChatRequest`]. [`crate::DeepSeek`] is the real one; tests use a
/// scripted one.
pub trait ChatModel {
    /// Send one request. `on_event` gets streamed fragments as they arrive; `cancel` is set by the
    /// user to stop reading the stream.
    fn complete(&self, request: &ChatRequest, on_event: &mut dyn FnMut(ModelEvent), cancel: &AtomicBool) -> Result<ChatResponse, ModelError>;
}
