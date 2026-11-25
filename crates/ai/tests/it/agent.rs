use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use figcraft_ai::tools::Backend;
use figcraft_ai::{Agent, AgentConfig, ChatModel, ChatRequest, ChatResponse, ModelError, ModelEvent, Step, ToolCall, ToolOutcome, execute_tool};
use serde_json::{Value, json};

/// Answers the scripted responses in order and keeps every request it was asked.
struct Scripted {
    responses: Mutex<VecDeque<ChatResponse>>,
    requests: Arc<Mutex<Vec<ChatRequest>>>,
}

impl Scripted {
    fn new(responses: Vec<ChatResponse>) -> (Self, Arc<Mutex<Vec<ChatRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        (Self { responses: Mutex::new(responses.into()), requests: requests.clone() }, requests)
    }
}

impl ChatModel for Scripted {
    fn complete(&self, request: &ChatRequest, _on_event: &mut dyn FnMut(ModelEvent), _cancel: &AtomicBool) -> Result<ChatResponse, ModelError> {
        self.requests.lock().unwrap().push(request.clone());
        self.responses.lock().unwrap().pop_front().ok_or(ModelError::Empty)
    }
}

/// A backend that records the control methods and answers each with `{"ok": true}`.
struct Recording {
    calls: Arc<Mutex<Vec<(String, Value)>>>,
}

impl Backend for Recording {
    fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        self.calls.lock().unwrap().push((method.to_string(), params));
        Ok(json!({"ok": true}))
    }

    fn has_ui(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        "recording".into()
    }
}

fn tool_call(id: &str, command: &str) -> ChatResponse {
    ChatResponse {
        tool_calls: vec![ToolCall { id: id.into(), name: "run_command".into(), arguments: json!({"command": command}).to_string() }],
        ..Default::default()
    }
}

fn answer(text: &str) -> ChatResponse {
    ChatResponse { content: text.into(), ..Default::default() }
}

#[test]
fn a_tool_call_runs_and_the_answer_reads_its_result() {
    let (model, requests) = Scripted::new(vec![tool_call("c1", "view.zoomIn"), answer("Done")]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut backend = Recording { calls: calls.clone() };
    let mut agent = Agent::new("system", vec![], AgentConfig::default());
    agent.user("make it bigger");

    let mut final_text = None;
    for _ in 0..8 {
        let request = agent.request("deepseek-flash", true);
        let response = model.complete(&request, &mut |_| {}, &AtomicBool::new(false)).unwrap();
        match agent.accept(response) {
            Step::Tools { calls: wanting, .. } => {
                for call in wanting {
                    let outcome = execute_tool(&mut backend, &call);
                    assert!(!outcome.is_error, "{:?}", outcome.text);
                    agent.tool_result(&call, &outcome);
                }
            }
            Step::Done { text } => {
                final_text = Some(text);
                break;
            }
            Step::Stopped { .. } | Step::Incomplete { .. } => panic!("the loop stopped early"),
        }
    }
    assert_eq!(final_text.as_deref(), Some("Done"));
    assert_eq!(calls.lock().unwrap()[0].0, "engine.execute");
    assert_eq!(calls.lock().unwrap()[0].1["command"], "view.zoomIn");

    let sent = requests.lock().unwrap();
    assert_eq!(sent.len(), 2);
    let last = sent[1].messages.last().expect("the tool result is the last message");
    assert_eq!(last.role, figcraft_ai::Role::Tool);
    assert_eq!(last.tool_call_id.as_deref(), Some("c1"));
    assert!(last.content.contains("ok"), "{}", last.content);
}

#[test]
fn the_step_cap_stops_a_model_that_keeps_calling_tools() {
    let (model, _) = Scripted::new(vec![tool_call("a", "view.zoomIn"), tool_call("b", "view.zoomIn"), tool_call("c", "view.zoomIn")]);
    let mut backend = Recording { calls: Arc::new(Mutex::new(Vec::new())) };
    let mut agent = Agent::new("system", vec![], AgentConfig { max_steps: 2, max_tool_chars: 1000, ..Default::default() });
    agent.user("loop");
    let mut stopped = None;
    for _ in 0..5 {
        let request = agent.request("deepseek-flash", true);
        let response = model.complete(&request, &mut |_| {}, &AtomicBool::new(false)).unwrap();
        match agent.accept(response) {
            Step::Tools { calls, .. } => {
                for call in calls {
                    let outcome = execute_tool(&mut backend, &call);
                    agent.tool_result(&call, &outcome);
                }
            }
            Step::Stopped { text } => {
                stopped = Some(text);
                break;
            }
            Step::Done { .. } | Step::Incomplete { .. } => panic!("the model never answered"),
        }
    }
    assert!(stopped.is_some(), "the second answer with tool calls hits the cap");
    assert!(agent.is_finished());
}

#[test]
fn malformed_arguments_are_a_readable_tool_error() {
    let call = ToolCall { id: "c".into(), name: "run_command".into(), arguments: "{\"command\":".into() };
    let mut backend = Recording { calls: Arc::new(Mutex::new(Vec::new())) };
    let outcome = execute_tool(&mut backend, &call);
    assert!(outcome.is_error);
    assert!(outcome.text.contains("invalid arguments"), "{}", outcome.text);
}

#[test]
fn cancelling_drops_the_unanswered_prompt() {
    let mut agent = Agent::new("system", vec![], AgentConfig::default());
    agent.user("hi");
    agent.cancel_pending_user();
    assert!(agent.messages().is_empty());
}

#[test]
fn a_result_is_bounded_before_it_goes_back() {
    let outcome = ToolOutcome::text("x".repeat(500));
    assert_eq!(outcome.for_model(10).chars().filter(|c| *c == 'x').count(), 10);
}

#[test]
fn an_answer_that_did_nothing_is_not_the_end() {
    // A reasoning-only reply (what a long plan with no room left for the answer looks like) used
    // to pass as a finished turn, leaving the file untouched and the user waiting.
    let mut agent = Agent::new("system", vec![], AgentConfig::default());
    agent.user("make a card");
    let silent = ChatResponse { content: String::new(), reasoning: Some("I will do it.".into()), ..Default::default() };
    assert_eq!(agent.accept(silent), Step::Incomplete { text: String::new(), cut_off: false });
    assert!(!agent.is_finished());
    assert!(agent.nudge(), "the driver may ask it to go on");
    assert_eq!(agent.accept(answer("Done")), Step::Done { text: "Done".into() });
}

#[test]
fn an_answer_cut_off_for_room_asks_again_a_bounded_number_of_times() {
    let mut agent = Agent::new("system", vec![], AgentConfig { max_nudges: 2, ..Default::default() });
    agent.user("make a tree");
    for _ in 0..2 {
        let cut = ChatResponse { content: "half a".into(), finish_reason: Some("length".into()), ..Default::default() };
        assert_eq!(agent.accept(cut), Step::Incomplete { text: "half a".into(), cut_off: true });
        assert!(agent.nudge(), "a nudge is allowed while the count is under the cap");
    }
    let cut = ChatResponse { content: "half a".into(), finish_reason: Some("length".into()), ..Default::default() };
    assert_eq!(agent.accept(cut), Step::Incomplete { text: "half a".into(), cut_off: true });
    assert!(!agent.nudge(), "the third nudge is refused, so the driver stops instead of looping");
    // A new prompt starts the count over.
    agent.user("again");
    assert!(agent.nudge());
}

#[test]
fn a_request_leaves_room_for_the_answer_after_the_thinking() {
    let agent = Agent::new("system", vec![], AgentConfig::default());
    let request = agent.request("deepseek-flash", true);
    assert!(request.max_tokens.is_some_and(|m| m >= 1_000_000), "the whole budget is asked for: {:?}", request.max_tokens);
}

#[test]
fn a_budget_the_endpoint_refuses_names_the_limit_it_takes() {
    let refused = figcraft_ai::ModelError::Status {
        code: 400,
        message: r#"{"error":{"message":"Invalid max_tokens value, the valid range of max_tokens is [1, 393216]"}}"#.into(),
    };
    assert_eq!(refused.max_tokens_limit(), Some(393216));
    // Other failures say nothing about a budget.
    let other = figcraft_ai::ModelError::Status { code: 400, message: "bad model".into() };
    assert_eq!(other.max_tokens_limit(), None);
    let server = figcraft_ai::ModelError::Status { code: 500, message: "max_tokens is [1, 100]".into() };
    assert_eq!(server.max_tokens_limit(), None, "only a refusal of the request counts");
}
