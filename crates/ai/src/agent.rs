//! The tool-calling loop: the model asks for tools, the caller runs them, and the model answers
//! with what they returned. Pure state, so the CLI drives it on one thread and the app drives it
//! from a UI pass with the HTTP request on a worker.

use serde_json::Value;

use crate::chat::{ChatMessage, ChatRequest, ChatResponse, Role, ToolCall};
use crate::outcome::ToolOutcome;

/// How far one request may go: the model calls at most `max_steps` times, no single tool result
/// sent back is longer than `max_tool_chars` characters, one answer may use `max_output_tokens`
/// (thinking counts toward it), and an answer that did nothing is nudged at most `max_nudges`
/// times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub max_tool_chars: usize,
    pub max_output_tokens: u32,
    pub max_nudges: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        // The most one answer may use, so the app never cuts a long plan short; thinking counts
        // toward it. An endpoint that takes less says so and `ModelError::max_tokens_limit`
        // lowers it. Kept equal to `figcraft_engine::ASSISTANT_TOKENS_MAX` (this crate sits above
        // the engine, so it cannot name it).
        Self { max_steps: 40, max_tool_chars: 60_000, max_output_tokens: 1_000_000, max_nudges: 2 }
    }
}

/// What the driver does after one answer: run these tools, show this final text, or stop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Run `calls` (in order), answer each with [`Agent::tool_result`], then request again. `text`
    /// is what the model said before calling them, which may be empty.
    Tools { calls: Vec<ToolCall>, text: String },
    /// The model finished: `text` is its answer.
    Done { text: String },
    /// The step budget ran out while the model still wanted tools. `text` is its last words.
    Stopped { text: String },
    /// The answer stopped short: it called no tool and said nothing, or it ran out of output
    /// tokens (`cut_off`). `text` is whatever came back. The driver nudges the model with
    /// [`Agent::nudge`] and asks again, and says so when it has nudged too often.
    Incomplete { text: String, cut_off: bool },
}

/// What a nudge says: the model has to act, not describe what it would do next.
const NUDGE: &str = "You stopped without calling a tool and without finishing, so nothing changed. \
Keep going now: call the tools you need until the task is done, and only then answer with a short \
summary. Do not describe what you are about to do instead of doing it.";

/// One conversation with the model.
#[derive(Clone, Debug)]
pub struct Agent {
    system: String,
    tools: Vec<Value>,
    messages: Vec<ChatMessage>,
    config: AgentConfig,
    steps: usize,
    nudges: usize,
    finished: bool,
}

impl Agent {
    pub fn new(system: impl Into<String>, tools: Vec<Value>, config: AgentConfig) -> Self {
        Self {
            system: system.into(),
            tools,
            messages: Vec::new(),
            config: AgentConfig { max_steps: config.max_steps.max(1), ..config },
            steps: 0,
            nudges: 0,
            finished: false,
        }
    }

    /// Add what the user asked.
    pub fn user(&mut self, text: impl Into<String>) {
        self.nudges = 0;
        self.messages.push(ChatMessage::user(text));
    }

    /// The next request: the system prompt, the conversation so far, and the tools.
    pub fn request(&self, model: impl Into<String>, thinking: bool) -> ChatRequest {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        messages.push(ChatMessage::system(self.system.clone()));
        messages.extend(self.messages.iter().cloned());
        let mut request = ChatRequest::new(model, messages, self.tools.clone());
        request.thinking = thinking;
        request.max_tokens = Some(self.config.max_output_tokens);
        request
    }

    /// Take one answer. With tool calls the driver runs them and calls [`Agent::tool_result`];
    /// otherwise the conversation is over.
    pub fn accept(&mut self, response: ChatResponse) -> Step {
        if self.finished {
            return Step::Done { text: String::new() };
        }
        self.steps += 1;
        let text = response.content.trim().to_string();
        let calls = response.tool_calls;
        // The answer stopped short: it ran out of output tokens, or it neither spoke nor acted, so
        // nothing happened. Not a finished turn either way.
        let cut_off = response.finish_reason.as_deref() == Some("length");
        self.messages.push(ChatMessage::assistant(response.content, calls.clone(), response.reasoning));
        if calls.is_empty() && (cut_off || text.is_empty()) {
            return Step::Incomplete { text, cut_off };
        }
        if calls.is_empty() {
            self.finished = true;
            return Step::Done { text };
        }
        if self.steps >= self.config.max_steps {
            self.finished = true;
            return Step::Stopped { text };
        }
        Step::Tools { calls, text }
    }

    /// Change the most tokens one answer may use (a setting the user picks while a conversation
    /// runs; the next request carries it).
    pub fn set_max_output_tokens(&mut self, tokens: u32) {
        self.config.max_output_tokens = tokens;
    }

    pub fn max_output_tokens(&self) -> u32 {
        self.config.max_output_tokens
    }

    /// Ask the model to carry on after [`Step::Incomplete`]. False once it has been nudged
    /// `max_nudges` times for this prompt, so the driver stops instead of looping.
    pub fn nudge(&mut self) -> bool {
        if self.nudges >= self.config.max_nudges {
            return false;
        }
        self.nudges += 1;
        self.finished = false;
        self.messages.push(ChatMessage::user(NUDGE));
        true
    }

    /// Answer a tool call the model made, so the next request carries what it returned.
    pub fn tool_result(&mut self, call: &ToolCall, outcome: &ToolOutcome) {
        self.messages.push(ChatMessage::tool(call.id.clone(), outcome.for_model(self.config.max_tool_chars)));
    }

    /// The user stopped while the last prompt waited for the model: forget that prompt, so the
    /// conversation doesn't keep an unanswered turn.
    pub fn cancel_pending_user(&mut self) {
        if self.messages.last().is_some_and(|m| m.role == Role::User) {
            self.messages.pop();
        }
    }

    /// Start over (another file, or a cleared conversation).
    pub fn reset(&mut self, system: impl Into<String>, tools: Vec<Value>) {
        self.system = system.into();
        self.tools = tools;
        self.messages.clear();
        self.steps = 0;
        self.nudges = 0;
        self.finished = false;
    }

    /// Change the rules and tools of a conversation in flight (the user picked another mode): the
    /// messages stay, so the thread keeps its history, and the next request carries the new mode.
    pub fn retarget(&mut self, system: impl Into<String>, tools: Vec<Value>) {
        self.system = system.into();
        self.tools = tools;
        self.finished = false;
    }

    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }
}
