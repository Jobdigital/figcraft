//! FigCraft's AI assistant core.
//!
//! The app and the CLI both drive one [`Agent`]: it keeps the conversation, asks the model for a
//! [`ChatRequest`], and turns the answer into either tool calls the caller runs through
//! [`tools::execute_tool`] (the same MCP tools every other agent uses) or a final answer.
//!
//! The model transport is [`DeepSeek`], an OpenAI-compatible chat client with function calling.
//! It is native-only; the browser build gets a stub that reports the limitation. Nothing here
//! touches the UI, the engine or the file system: the caller supplies the [`tools::Backend`].
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod agent;
pub mod chat;
mod config;
#[cfg(not(target_arch = "wasm32"))]
mod deepseek;
#[cfg(target_arch = "wasm32")]
#[path = "deepseek_wasm.rs"]
mod deepseek;
mod mode;
mod outcome;
#[cfg(not(target_arch = "wasm32"))]
pub mod tools;

pub use agent::{Agent, AgentConfig, Step};
pub use chat::{ChatMessage, ChatModel, ChatRequest, ChatResponse, ModelError, ModelEvent, Role, ToolCall, Usage};
pub use config::{API_KEY_ENV, BASE_URL_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL, DeepSeekConfig, MODELS, key_available};
pub use deepseek::DeepSeek;
pub use mode::Mode;
pub use outcome::ToolOutcome;
#[cfg(not(target_arch = "wasm32"))]
pub use tools::{
    READ_ONLY_TOOLS, Scope, execute_tool, openai_tools, system_prompt, system_prompt_for, tool_definitions_for, tool_definitions_for_mode,
};
