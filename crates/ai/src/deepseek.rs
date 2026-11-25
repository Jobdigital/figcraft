//! The DeepSeek client: an OpenAI-compatible chat completions request over HTTPS, streamed as
//! server-sent events. Blocking on purpose: the CLI runs it on its thread and the app on a worker,
//! so no async runtime is pulled in.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::chat::{ChatModel, ChatRequest, ChatResponse, ModelError, ModelEvent, ToolCall, Usage};
use crate::config::DeepSeekConfig;

/// The longest answer body accepted without streaming.
const MAX_BODY: u64 = 32 * 1024 * 1024;
/// How much of a refusal's body is read (and then sanitised and bounded in [`sanitize`]).
const MAX_ERROR_BODY: u64 = 64 * 1024;
/// The DeepSeek chat completions client.
pub struct DeepSeek {
    config: DeepSeekConfig,
    agent: ureq::Agent,
}

impl DeepSeek {
    pub fn new(config: DeepSeekConfig) -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(config.timeout))
            .timeout_connect(Some(Duration::from_secs(15)))
            // A refusal's body says why; read it ourselves instead of ureq turning it into an error.
            .http_status_as_error(false)
            .build()
            .into();
        Self { config, agent }
    }

    pub fn config(&self) -> &DeepSeekConfig {
        &self.config
    }

    fn send_once(
        &self,
        url: &str,
        body: &str,
        stream: bool,
        on_event: &mut dyn FnMut(ModelEvent),
        cancel: &AtomicBool,
    ) -> Result<ChatResponse, ModelError> {
        let mut response = self
            .agent
            .post(url)
            .header("content-type", "application/json")
            .header("accept", if stream { "text/event-stream" } else { "application/json" })
            .header("authorization", &format!("Bearer {}", self.config.api_key))
            .send(body.as_bytes())
            .map_err(|e| ModelError::Http(sanitize(&e.to_string())))?;
        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            let message = read_text(response.body_mut(), MAX_ERROR_BODY);
            return Err(ModelError::Status { code: status, message: sanitize(&message) });
        }
        if stream {
            parse_stream(BufReader::new(response.body_mut().as_reader()), on_event, cancel)
        } else {
            let text = response.body_mut().with_config().limit(MAX_BODY).read_to_string().map_err(|e| ModelError::Http(sanitize(&e.to_string())))?;
            parse_completion(&text)
        }
    }
}

impl ChatModel for DeepSeek {
    fn complete(&self, request: &ChatRequest, on_event: &mut dyn FnMut(ModelEvent), cancel: &AtomicBool) -> Result<ChatResponse, ModelError> {
        if self.config.api_key.trim().is_empty() {
            return Err(ModelError::MissingKey);
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(ModelError::Cancelled);
        }
        let url = self.config.endpoint();
        let body = request_body(request).to_string();
        let mut attempt = 0;
        loop {
            // Only a failure before any text reached the UI may be retried: repeating a stream
            // would show the answer twice.
            let emitted = std::cell::Cell::new(false);
            let result = {
                let mut relay = |event: ModelEvent| {
                    emitted.set(true);
                    on_event(event);
                };
                self.send_once(&url, &body, request.stream, &mut relay, cancel)
            };
            match result {
                Err(e) if !emitted.get() && attempt < self.config.max_retries && e.retryable() => {
                    log::warn!("DeepSeek request failed ({e}); retrying");
                    std::thread::sleep(backoff(attempt));
                    attempt += 1;
                }
                other => return other,
            }
        }
    }
}

/// One request body in DeepSeek's OpenAI-compatible shape. `reasoning_content` is never sent
/// back; the model reads its reasoning in the turn that produced it.
fn request_body(request: &ChatRequest) -> Value {
    let messages: Vec<Value> = request
        .messages
        .iter()
        .map(|m| {
            let mut message = json!({"role": m.role.as_str(), "content": m.content});
            if !m.tool_calls.is_empty() {
                message["tool_calls"] = Value::Array(
                    m.tool_calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments},
                            })
                        })
                        .collect(),
                );
                if m.content.is_empty() {
                    message["content"] = Value::Null;
                }
            }
            if let Some(id) = &m.tool_call_id {
                message["tool_call_id"] = json!(id);
            }
            message
        })
        .collect();
    let mut body = json!({"model": &request.model, "messages": messages, "stream": request.stream});
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.clone());
        body["tool_choice"] = json!("auto");
    }
    body["thinking"] = json!({"type": if request.thinking { "enabled" } else { "disabled" }});
    if request.thinking
        && let Some(effort) = &request.reasoning_effort
    {
        body["reasoning_effort"] = json!(effort);
    }
    if let Some(max) = request.max_tokens {
        body["max_tokens"] = json!(max);
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    body
}

/// Read a body with a cap; an unreadable body reads as empty (the status still explains).
fn read_text(body: &mut ureq::Body, limit: u64) -> String {
    body.with_config().limit(limit).read_to_string().unwrap_or_default()
}

/// Keep a server's message printable and bounded: control characters become spaces and the text
/// is cut at about 2,000 bytes. Never includes the request's headers.
fn sanitize(text: &str) -> String {
    let mut out = String::with_capacity(text.len().min(2000));
    for c in text.chars() {
        if out.len() >= 2000 {
            break;
        }
        out.push(if c.is_control() { ' ' } else { c });
    }
    out.trim().to_string()
}

fn backoff(attempt: u32) -> Duration {
    Duration::from_millis(500u64.saturating_mul(1u64 << attempt.min(4)))
}

#[derive(Deserialize)]
struct Completion {
    #[serde(default)]
    choices: Vec<Choice>,
    usage: Option<UsageJson>,
}

#[derive(Deserialize)]
struct Choice {
    message: MessageJson,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct MessageJson {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCallJson>,
}

#[derive(Deserialize)]
struct ToolCallJson {
    id: Option<String>,
    function: FunctionJson,
}

#[derive(Deserialize)]
struct FunctionJson {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Default)]
struct UsageJson {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

impl From<UsageJson> for Usage {
    fn from(u: UsageJson) -> Self {
        Self { prompt_tokens: u.prompt_tokens, completion_tokens: u.completion_tokens, total_tokens: u.total_tokens }
    }
}

fn parse_completion(text: &str) -> Result<ChatResponse, ModelError> {
    let completion: Completion = serde_json::from_str(text).map_err(|e| ModelError::Json(format!("{e}: {}", sanitize(text))))?;
    let Some(choice) = completion.choices.into_iter().next() else {
        return Err(ModelError::Empty);
    };
    let content = choice.message.content.unwrap_or_default();
    let reasoning = choice.message.reasoning_content.filter(|r| !r.is_empty());
    let tool_calls: Vec<ToolCall> = choice
        .message
        .tool_calls
        .into_iter()
        .enumerate()
        .map(|(i, call)| ToolCall {
            id: call.id.unwrap_or_else(|| format!("call_{i}")),
            name: call.function.name,
            arguments: call.function.arguments,
        })
        .collect();
    if content.trim().is_empty() && reasoning.is_none() && tool_calls.is_empty() {
        return Err(ModelError::Empty);
    }
    Ok(ChatResponse { content, reasoning, tool_calls, finish_reason: choice.finish_reason, usage: completion.usage.map(Usage::from) })
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    usage: Option<UsageJson>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct Delta {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<DeltaToolCall>,
}

#[derive(Deserialize)]
struct DeltaToolCall {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    function: Option<DeltaFunction>,
}

#[derive(Deserialize)]
struct DeltaFunction {
    name: Option<String>,
    arguments: Option<String>,
}

fn parse_stream(mut reader: impl BufRead, on_event: &mut dyn FnMut(ModelEvent), cancel: &AtomicBool) -> Result<ChatResponse, ModelError> {
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: BTreeMap<usize, ToolCall> = BTreeMap::new();
    let mut finish_reason = None;
    let mut usage = None;
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line).map_err(|e| ModelError::Http(sanitize(&e.to_string())))?;
        if read == 0 {
            break;
        }
        if cancel.load(Ordering::Relaxed) {
            return Err(ModelError::Cancelled);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        let Some(data) = trimmed.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }
        let chunk: StreamChunk = serde_json::from_str(data).map_err(|e| ModelError::Json(format!("{e}: {}", sanitize(data))))?;
        if let Some(u) = chunk.usage {
            usage = Some(Usage::from(u));
        }
        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                finish_reason = Some(reason);
            }
            let delta = choice.delta;
            if let Some(text) = delta.content.filter(|t| !t.is_empty()) {
                content.push_str(&text);
                on_event(ModelEvent::Content(text));
            }
            if let Some(text) = delta.reasoning_content.filter(|t| !t.is_empty()) {
                reasoning.push_str(&text);
                on_event(ModelEvent::Reasoning(text));
            }
            for call in delta.tool_calls {
                let entry = calls.entry(call.index).or_insert_with(|| ToolCall { id: String::new(), name: String::new(), arguments: String::new() });
                if let Some(id) = call.id {
                    entry.id = id;
                }
                if let Some(function) = call.function {
                    if let Some(name) = function.name {
                        entry.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        entry.arguments.push_str(&arguments);
                    }
                }
            }
        }
    }
    let tool_calls: Vec<ToolCall> = calls
        .into_iter()
        .filter_map(|(i, mut call)| {
            if call.name.is_empty() {
                return None;
            }
            if call.id.is_empty() {
                call.id = format!("call_{i}");
            }
            Some(call)
        })
        .collect();
    if content.trim().is_empty() && reasoning.is_empty() && tool_calls.is_empty() {
        return Err(ModelError::Empty);
    }
    Ok(ChatResponse { content, reasoning: (!reasoning.is_empty()).then_some(reasoning), tool_calls, finish_reason, usage })
}

#[cfg(test)]
#[path = "deepseek_tests.rs"]
mod tests;
