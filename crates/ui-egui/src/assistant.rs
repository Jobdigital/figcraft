//! The Assistant panel: a conversation with DeepSeek that edits the open file through the same
//! MCP tools every other agent uses. The model runs on a worker thread; tool calls run here on the
//! UI thread, pinned to the file the conversation started on.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};

use figcraft_ai::{Agent, AgentConfig, ChatResponse, DeepSeekConfig, Mode, ModelError, ModelEvent, Step, ToolCall};
#[cfg(not(target_arch = "wasm32"))]
use figcraft_ai::{ChatModel, DeepSeek, Scope, execute_tool, openai_tools, system_prompt_for, tool_definitions_for};
use serde_json::Value;
use serde_json::json;

use crate::i18n::{tr, trf};
use crate::theme::{Tokens, regular};
use crate::{FigApp, a11y, widgets};

/// How wide the sidebar opens for the chat when it is narrower than this: a conversation needs
/// more room than the file tree. The drag handle still sets any width afterwards.
const CHAT_WIDTH: f32 = 320.0;

/// How much of the sidebar's height the composer, its buttons and its hint take, so the thread
/// scrolls in what is left.
const COMPOSER_HEIGHT: f32 = 140.0;

/// The name the Assistant's commands and comments carry ("DeepSeek (agent)").
const AGENT_NAME: &str = "DeepSeek";

/// How many lines a `#reference` may add before it says how many it left out.
const MAX_REFERENCE_LINES: usize = 200;

/// The tools the in-app Assistant may use in Agent mode: design and inspection only. Opening,
/// saving, closing or quitting files, simulated input and waiting for changes belong to
/// `figcraft-cli agent` and the MCP server, not to a panel working on the file in front.
#[cfg(not(target_arch = "wasm32"))]
const APP_TOOLS: &[&str] = &[
    "list_commands",
    "run_command",
    "batch",
    "create_node",
    "set_properties",
    "inspect_document",
    "get_node",
    "design_context",
    "select",
    "export",
    "screenshot",
    "undo",
    "redo",
];

/// The tools Ask and Plan keep: everything above that only looks. Taking the rest away is what
/// makes "change nothing" true, rather than a rule the model could break.
#[cfg(not(target_arch = "wasm32"))]
const APP_TOOLS_READ: &[&str] = &["list_commands", "inspect_document", "get_node", "design_context", "select", "screenshot"];

/// The rules and tools of a mode, for the open file: the system prompt and what it may call.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn rules(file: &str, mode: Mode) -> (String, Vec<Value>) {
    let names = if mode.edits() { APP_TOOLS } else { APP_TOOLS_READ };
    let system = format!("{}\n\nThe user has the file \"{file}\" open.", system_prompt_for(Scope::App, mode));
    (system, openai_tools(&tool_definitions_for(names)))
}

/// A conversation pinned to the open file (called when the file changes or none exists yet).
#[cfg(not(target_arch = "wasm32"))]
fn new_agent(file: &str, mode: Mode, max_tokens: u32) -> Agent {
    let (system, tools) = rules(file, mode);
    let config = AgentConfig { max_output_tokens: max_tokens, ..Default::default() };
    Agent::new(system, tools, config)
}

/// The browser build has no model, so the panel never gets a conversation to run.
#[cfg(target_arch = "wasm32")]
fn new_agent(_file: &str, _mode: Mode, _max_tokens: u32) -> Agent {
    Agent::new("", Vec::new(), AgentConfig::default())
}

/// One line of the conversation shown in the panel.
#[derive(Clone, Debug, PartialEq)]
pub enum Entry {
    User(String),
    Assistant {
        text: String,
        reasoning: String,
    },
    /// A tool call; `ok` is `None` while it runs. `text` is what it was given and what it
    /// answered, bounded, for the copied transcript.
    Tool {
        name: String,
        ok: Option<bool>,
        text: String,
    },
    Note(String),
    Error(String),
}

/// What the Assistant is doing.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Phase {
    #[default]
    Idle,
    Waiting,
    Streaming,
    Tool(String),
    Failed(String),
}

impl Phase {
    /// The name `ui.inspect` publishes: what automation reads to tell a run from a failure.
    pub fn id(&self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Waiting => "waiting",
            Phase::Streaming => "streaming",
            Phase::Tool(_) => "tool",
            Phase::Failed(_) => "failed",
        }
    }
}

/// Something a prompt can name with `#name`, resolved against the open file when the message is
/// sent (so it is always current): a chat client's references, aimed at a design file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reference {
    /// The layers selected on the canvas.
    Selection,
    /// The page shown, with its top-level layers.
    Page,
    /// The file's name and what it holds.
    File,
    /// The local styles.
    Styles,
    /// The local variables, by collection.
    Variables,
    /// The file's main components.
    Components,
}

impl Reference {
    /// Every reference, in picker order.
    pub const ALL: &'static [Reference] =
        &[Reference::Selection, Reference::Page, Reference::File, Reference::Styles, Reference::Variables, Reference::Components];

    /// The word after `#`.
    pub fn id(self) -> &'static str {
        match self {
            Reference::Selection => "selection",
            Reference::Page => "page",
            Reference::File => "file",
            Reference::Styles => "styles",
            Reference::Variables => "variables",
            Reference::Components => "components",
        }
    }

    /// The name the picker and the chips show (a translation key: English).
    pub fn label(self) -> &'static str {
        match self {
            Reference::Selection => "Selected layers",
            Reference::Page => "This page",
            Reference::File => "This file",
            Reference::Styles => "Local styles",
            Reference::Variables => "Local variables",
            Reference::Components => "Components",
        }
    }

    /// One line on what the reference brings, for the picker.
    pub fn hint(self) -> &'static str {
        match self {
            Reference::Selection => "The layers you have selected, with their ids",
            Reference::Page => "The page shown and its top-level layers",
            Reference::File => "The file's name, pages and counts",
            Reference::Styles => "The local styles, by kind",
            Reference::Variables => "The local variables, by collection",
            Reference::Components => "The file's main components",
        }
    }

    /// The reference an id names, in any case.
    pub fn from_id(id: &str) -> Option<Self> {
        Reference::ALL.iter().copied().find(|r| r.id().eq_ignore_ascii_case(id.trim()))
    }

    /// The token the prompt carries: `#selection`.
    pub fn token(self) -> String {
        format!("#{}", self.id())
    }
}

/// A prompt the `/` picker offers: what a chat client's slash commands do for design work. Each
/// one is a head start, not a hidden mode: it fills the composer and attaches what it needs, and
/// the user can edit it before sending.
struct Recipe {
    id: &'static str,
    label: &'static str,
    hint: &'static str,
    prompt: &'static str,
    reference: Reference,
}

/// The recipes Agent mode offers. Ask and Plan change nothing, so they get none: a recipe asks for
/// work to be done.
const RECIPES: &[Recipe] = &[
    Recipe {
        id: "a11y",
        label: "Check accessibility",
        hint: "Contrast, text sizes and colour-only signals on this page",
        prompt: "Check this page for accessibility problems: text contrast against its background, text smaller than 12, and anything signalled by colour alone. List what fails with layer ids, then fix it and say what changed.",
        reference: Reference::Page,
    },
    Recipe {
        id: "responsive",
        label: "Tablet and mobile versions",
        hint: "Copies of the selected frame at 768 and 390 wide, keeping the auto layout",
        prompt: "Make a tablet version (768 wide) and a mobile version (390 wide) of the selected frame, keeping its auto layout and styles, and place them beside it.",
        reference: Reference::Selection,
    },
    Recipe {
        id: "cleanup",
        label: "Tidy the page",
        hint: "Meaningful names, groups where they help, and no empty layers",
        prompt: "Tidy this page: give the layers meaningful names, group the ones that belong together, remove the empty ones, and say what changed.",
        reference: Reference::Page,
    },
];

/// The references a prompt names with `#name`, in the order they appear and without repeats. Only
/// known ids count, so `#FFFFFF` stays a colour.
pub fn references_in(text: &str) -> Vec<Reference> {
    let mut found = Vec::new();
    for word in text.split_whitespace() {
        // Punctuation around the word (`(#page)`, `#page.`) is not part of it; the `#` is.
        let word = word.trim_start_matches(|c: char| !c.is_ascii_alphanumeric() && c != '#');
        let word = word.trim_end_matches(|c: char| !c.is_ascii_alphanumeric());
        let Some(id) = word.strip_prefix('#') else { continue };
        if let Some(r) = Reference::from_id(id)
            && !found.contains(&r)
        {
            found.push(r);
        }
    }
    found
}

/// The `#reference` or `/command` being typed at the end of the text, if any: what the picker
/// offers while the composer has the keyboard.
pub fn trailing_sigil(text: &str) -> Option<(char, String)> {
    let token = text.rsplit(char::is_whitespace).next().unwrap_or("");
    let sigil = token.chars().next()?;
    if !matches!(sigil, '#' | '/') {
        return None;
    }
    Some((sigil, token.chars().skip(1).collect::<String>().to_lowercase()))
}

/// A layer, as a line a reference lists it.
fn layer_line(id: figcraft_doc::NodeId, doc: &figcraft_doc::Document) -> Option<String> {
    let node = doc.node(id)?;
    Some(format!("- {} ({}, id {})", node.name, node.kind.type_name(), id.0))
}

/// What a reference adds to a prompt, from the file as it is now: `None` when there is nothing
/// behind it (no file open, or an empty document part).
fn reference_block(app: &FigApp, reference: Reference) -> Option<String> {
    let ds = app.session.ds().ok()?;
    let doc = &ds.doc;
    let page = doc.node(ds.page)?;
    Some(match reference {
        Reference::Selection => {
            let lines: Vec<String> = ds.selection.iter().filter_map(|id| layer_line(*id, doc)).collect();
            if lines.is_empty() {
                return Some("\n\n#selection: nothing is selected in the file right now.".to_string());
            }
            let mut out = format!(
                "\n\n#selection — {} selected layer(s) (their ids are what get_node, set_properties, node.set and parent take):",
                lines.len()
            );
            out.push_str(&lines_block(lines, 0));
            out
        }
        Reference::Page => {
            let total = page.children.len();
            let lines: Vec<String> = page.children.iter().filter_map(|id| layer_line(*id, doc)).take(MAX_REFERENCE_LINES).collect();
            let mut out = format!("\n\n#page — \"{}\" (id {}), its {total} top-level layer(s):", page.name, page.id.0);
            out.push_str(&lines_block(lines, total.saturating_sub(MAX_REFERENCE_LINES)));
            out
        }
        Reference::File => {
            let layers: usize = doc.pages.iter().map(|p| doc.subtree(*p).len().saturating_sub(1)).sum();
            format!(
                "\n\n#file — \"{}\": {} page(s), {layers} layer(s), {} style(s), {} variable(s), {} image(s). Current page: \"{}\" (id {}).",
                doc.name,
                doc.pages.len(),
                doc.styles.len(),
                doc.variables.variables.len(),
                doc.images.len(),
                page.name,
                page.id.0
            )
        }
        Reference::Styles => {
            if doc.styles.is_empty() {
                return Some("\n\n#styles: this file has no local styles.".to_string());
            }
            let lines: Vec<String> = doc
                .styles
                .iter()
                .take(MAX_REFERENCE_LINES)
                .map(|s| {
                    let kind = match &s.kind {
                        figcraft_doc::StyleKind::Paint { .. } => "paint",
                        figcraft_doc::StyleKind::Text { .. } => "text",
                        figcraft_doc::StyleKind::Effect { .. } => "effect",
                        figcraft_doc::StyleKind::Grid { .. } => "grid",
                    };
                    format!("- {} ({kind})", s.name)
                })
                .collect();
            let mut out = format!("\n\n#styles — {} local style(s):", doc.styles.len());
            out.push_str(&lines_block(lines, doc.styles.len().saturating_sub(MAX_REFERENCE_LINES)));
            out
        }
        Reference::Variables => {
            if doc.variables.collections.is_empty() {
                return Some("\n\n#variables: this file has no local variables.".to_string());
            }
            let mut out =
                format!("\n\n#variables — {} collection(s), {} variable(s):", doc.variables.collections.len(), doc.variables.variables.len());
            for collection in &doc.variables.collections {
                let names: Vec<&str> = doc
                    .variables
                    .variables
                    .iter()
                    .filter(|v| v.collection == collection.id)
                    .map(|v| v.name.as_str())
                    .take(MAX_REFERENCE_LINES)
                    .collect();
                let list = if names.is_empty() { "(none)".to_string() } else { names.join(", ") };
                out.push_str(&format!("\n- {} ({} mode(s)): {list}", collection.name, collection.modes.len()));
            }
            out
        }
        Reference::Components => {
            let mut lines = Vec::new();
            for page_id in &doc.pages {
                for id in doc.subtree(*page_id) {
                    let Some(node) = doc.node(id) else { continue };
                    if matches!(node.kind, figcraft_doc::NodeKind::Component { .. } | figcraft_doc::NodeKind::ComponentSet { .. }) {
                        lines.push(format!("- {} ({}, id {})", node.name, node.kind.type_name(), id.0));
                    }
                }
            }
            if lines.is_empty() {
                return Some("\n\n#components: this file has no main components.".to_string());
            }
            let total = lines.len();
            let mut out = format!("\n\n#components — {total} main component(s):");
            out.push_str(&lines_block(lines.into_iter().take(MAX_REFERENCE_LINES).collect(), total.saturating_sub(MAX_REFERENCE_LINES)));
            out
        }
    })
}

/// A list a reference prints, with a closing line when it was cut.
fn lines_block(lines: Vec<String>, more: usize) -> String {
    let mut out = String::new();
    for line in lines {
        out.push('\n');
        out.push_str(&line);
    }
    if more > 0 {
        out.push_str(&format!("\n- …and {more} more"));
    }
    out
}

/// The Assistant's state: conversation, settings and the request in flight.
pub struct AssistantUi {
    pub open: bool,
    pub entries: Vec<Entry>,
    pub input: String,
    pub model: String,
    pub thinking: bool,
    /// What the Assistant may do with a prompt (Agent, Ask or Plan), the panel's agent picker.
    pub mode: Mode,
    /// The key in use: one saved in the preferences when the user asked to remember it, else one
    /// typed for this session. An empty key means DEEPSEEK_API_KEY.
    pub key: String,
    /// The typed key is kept in FigCraft's settings file (`pref.assistant {remember}`).
    pub remember_key: bool,
    /// The most tokens one answer may use; thinking counts toward it (16 to 1,000,000).
    pub max_tokens: u32,
    /// An OpenAI-compatible endpoint; empty means DEEPSEEK_BASE_URL or DeepSeek.
    pub endpoint: String,
    pub show_settings: bool,
    pub phase: Phase,
    /// The conversation's agent, kept across prompts. `None` until the first prompt or after the
    /// open file changes.
    pub agent: Option<Agent>,
    /// The document the conversation edits (its uid), pinned on every tool call.
    pub document: Option<u64>,
    rx: Option<Receiver<Event>>,
    cancel: Option<Arc<AtomicBool>>,
    focus: bool,
}

impl Default for AssistantUi {
    fn default() -> Self {
        Self {
            open: false,
            entries: Vec::new(),
            input: String::new(),
            model: figcraft_ai::DEFAULT_MODEL.to_string(),
            thinking: true,
            mode: Mode::Agent,
            key: String::new(),
            remember_key: false,
            max_tokens: figcraft_engine::ASSISTANT_TOKENS_DEFAULT,
            endpoint: String::new(),
            show_settings: false,
            phase: Phase::Idle,
            agent: None,
            document: None,
            rx: None,
            cancel: None,
            focus: false,
        }
    }
}

/// A message from the worker thread.
enum Event {
    Delta(ModelEvent),
    Done(Result<ChatResponse, ModelError>),
}

impl AssistantUi {
    pub fn busy(&self) -> bool {
        matches!(self.phase, Phase::Waiting | Phase::Streaming | Phase::Tool(_))
    }

    /// Stop waiting for the answer (the request is dropped; the thread sees the flag and stops
    /// reading the stream).
    pub(crate) fn stop(&mut self) {
        if let Some(cancel) = self.cancel.take() {
            cancel.store(true, Ordering::Relaxed);
        }
        self.rx = None;
        if let Some(agent) = &mut self.agent {
            agent.cancel_pending_user();
        }
        if self.busy() {
            self.entries.push(Entry::Note(tr("Stopped.").to_string()));
        }
        self.phase = Phase::Idle;
    }

    fn config(&self) -> DeepSeekConfig {
        let mut config = DeepSeekConfig::from_env();
        if !self.key.trim().is_empty() {
            config = config.with_key(self.key.trim());
        }
        if !self.endpoint.trim().is_empty() {
            config = config.with_base_url(self.endpoint.trim());
        }
        config
    }

    /// The key this session would use: the environment's, or the one typed for the session.
    pub fn has_key(&self) -> bool {
        figcraft_ai::key_available(&self.key, std::env::var(figcraft_ai::API_KEY_ENV).ok().as_deref())
    }

    /// Take the settings saved in the preferences (called when the app starts).
    pub fn apply_prefs(&mut self, prefs: &figcraft_engine::Prefs) {
        self.model = if prefs.assistant_model.trim().is_empty() { figcraft_ai::DEFAULT_MODEL.to_string() } else { prefs.assistant_model.clone() };
        self.thinking = prefs.assistant_thinking;
        self.mode = Mode::from_id(&prefs.assistant_agent).unwrap_or_default();
        self.endpoint = prefs.assistant_endpoint.clone();
        self.key = prefs.assistant_key.clone();
        self.remember_key = prefs.assistant_remember || !prefs.assistant_key.is_empty();
        self.max_tokens = prefs.assistant_max_tokens.clamp(figcraft_engine::ASSISTANT_TOKENS_MIN, figcraft_engine::ASSISTANT_TOKENS_MAX);
    }

    /// Pick another mode: it is saved, and a conversation already started keeps its messages but
    /// gets the new rules and tools for the next request.
    pub(crate) fn set_mode(&mut self, app: &mut FigApp, mode: Mode) {
        if self.mode == mode {
            return;
        }
        self.apply_mode(app, mode);
        self.save(app, json!({"agent": mode.id()}));
    }

    /// Take a mode without writing the settings (the frame does this when `pref.assistant` set one
    /// from outside, so automation can switch a panel that is already open).
    pub(crate) fn apply_mode(&mut self, app: &FigApp, mode: Mode) {
        if self.mode == mode {
            return;
        }
        self.mode = mode;
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(agent) = &mut self.agent {
            let file = app.session.ds().map(|d| d.doc.name.clone()).unwrap_or_default();
            let (system, tools) = rules(&file, mode);
            agent.retarget(system, tools);
        }
    }

    /// Start a new conversation (the panel's New chat): the thread goes, the settings stay.
    pub(crate) fn new_chat(&mut self) {
        self.stop();
        self.entries.clear();
        self.agent = None;
        self.document = None;
        self.input.clear();
        self.phase = Phase::Idle;
    }

    /// Keep the answer budget, and use it for the next request of a conversation in flight.
    pub(crate) fn save_max_tokens(&mut self, app: &mut FigApp) {
        if let Some(agent) = &mut self.agent {
            agent.set_max_output_tokens(self.max_tokens);
        }
        self.save(app, json!({"maxTokens": self.max_tokens}));
    }

    /// An endpoint that refused the budget says what it takes ("the valid range of max_tokens is
    /// [1, 393216]"): keep that, save it and hand back what to tell the user, so the next request
    /// goes through. None when the error is about anything else.
    pub(crate) fn lower_budget(&mut self, app: &mut FigApp, agent: &mut Agent, error: &ModelError) -> Option<String> {
        let limit = error.max_tokens_limit().filter(|limit| *limit < self.max_tokens)?;
        self.max_tokens = limit;
        agent.set_max_output_tokens(limit);
        self.save(app, json!({"maxTokens": limit}));
        Some(trf("This endpoint takes at most {tokens} tokens per answer: the budget is now that.", &[("tokens", &limit.to_string())]))
    }

    /// Keep the key, or forget it, as the Remember box says. The settings keep the answer too, so
    /// the box comes back the way it was left.
    fn save_remember(&mut self, app: &mut FigApp) {
        let params = if self.remember_key { json!({"key": self.key, "remember": true}) } else { json!({"remember": false}) };
        self.save(app, params);
    }

    /// Add `#reference` to the prompt (the Add selection button, the picker): the text as it is,
    /// then the token, so the prompt itself says what it carries.
    pub(crate) fn add_reference(&mut self, reference: Reference) {
        if !references_in(&self.input).contains(&reference) {
            if !self.input.is_empty() && !self.input.ends_with(char::is_whitespace) {
                self.input.push(' ');
            }
            self.input.push_str(&reference.token());
            self.input.push(' ');
        }
        self.focus = true;
    }

    /// Take a `#reference` out of the prompt (the × on its chip), keeping everything else as typed.
    pub(crate) fn remove_reference(&mut self, reference: Reference) {
        let token = reference.token();
        self.input = self
            .input
            .split('\n')
            .map(|line| line.split(' ').filter(|word| word.trim() != token).collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join("\n");
    }

    /// The references the prompt names, in the order they appear.
    pub fn references(&self) -> Vec<Reference> {
        references_in(&self.input)
    }

    /// What the model reads: the prompt as typed, then each `#reference` with the data behind it,
    /// resolved now, so it is always the file as it is.
    pub fn resolved_prompt(&self, app: &FigApp, text: &str) -> String {
        let mut out = text.to_string();
        for reference in references_in(text) {
            if let Some(block) = reference_block(app, reference) {
                out.push_str(&block);
            }
        }
        out
    }

    /// Keep settings through the settings command (which validates them) and write the preferences.
    fn save(&mut self, app: &mut FigApp, params: Value) {
        if app.run("pref.assistant", params).is_some() {
            app.services.save_prefs(&app.session.prefs);
        }
    }

    /// The whole conversation as text, for pasting into a message or an issue: what was asked,
    /// what the model said (and thought), each tool call with its arguments and answer, and any
    /// note or error.
    pub fn transcript(&self) -> String {
        let mut out = String::from("# FigCraft Assistant\n\n");
        out.push_str(&format!("Mode: {} · Model: {} · Thinking: {}\n", tr(self.mode.label()), self.model, if self.thinking { "on" } else { "off" }));
        if !self.endpoint.trim().is_empty() {
            out.push_str(&format!("Endpoint: {}\n", self.endpoint.trim()));
        }
        for entry in &self.entries {
            match entry {
                Entry::User(text) => out.push_str(&format!("\n## You\n\n{text}\n")),
                Entry::Assistant { text, reasoning } => {
                    out.push_str("\n## Assistant\n");
                    if !text.is_empty() {
                        out.push_str(&format!("\n{text}\n"));
                    }
                    if !reasoning.is_empty() {
                        out.push_str("\nReasoning:\n");
                        for line in reasoning.lines() {
                            out.push_str("> ");
                            out.push_str(line);
                            out.push('\n');
                        }
                    }
                }
                Entry::Tool { name, ok, text } => {
                    let state = match ok {
                        Some(true) => "ok",
                        Some(false) => "failed",
                        None => "running",
                    };
                    out.push_str(&format!("\n## Tool: {name} ({state})\n"));
                    if !text.trim().is_empty() {
                        out.push_str(&format!("\n{text}\n"));
                    }
                }
                Entry::Note(text) => out.push_str(&format!("\n> {text}\n")),
                Entry::Error(text) => out.push_str(&format!("\n## Error\n\n{text}\n")),
            }
        }
        out
    }

    /// Copy the conversation; `false` when there is nothing to copy.
    fn copy_transcript(&mut self, app: &mut FigApp) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        let text = self.transcript();
        app.clipboard_set_text(&text);
        app.toast(tr("Chat copied").to_string());
        true
    }

    /// Send what's typed. The conversation starts over when the open file changed.
    fn start(&mut self, app: &mut FigApp, ctx: &egui::Context) {
        if self.busy() {
            return;
        }
        let prompt = self.input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        let Ok(ds) = app.session.ds() else {
            self.fail(tr("Open a file to use the Assistant.").to_string());
            return;
        };
        let uid = ds.uid;
        let name = ds.doc.name.clone();
        let config = self.config();
        if config.api_key.is_empty() {
            self.fail(trf("Set {variable} or type a key in Settings.", &[("variable", &figcraft_ai::API_KEY_ENV)]));
            return;
        }
        if self.document != Some(uid) {
            if self.agent.is_some() {
                self.entries.clear();
                self.entries.push(Entry::Note(tr("The file changed: the conversation started over.").to_string()));
            }
            self.document = Some(uid);
            self.agent = None;
        }
        if self.agent.is_none() {
            self.agent = Some(new_agent(&name, self.mode, self.max_tokens));
        }
        self.input.clear();
        // What the user typed stays in the thread; the references' data goes to the model only.
        let sent = self.resolved_prompt(app, &prompt);
        self.entries.push(Entry::User(prompt));
        if let Some(agent) = &mut self.agent {
            agent.user(sent);
        }
        self.send(ctx, config);
    }

    /// Ask the model for the next answer on a worker thread.
    #[cfg(not(target_arch = "wasm32"))]
    fn send(&mut self, ctx: &egui::Context, config: DeepSeekConfig) {
        let Some(agent) = &self.agent else { return };
        let mut request = agent.request(&self.model, self.thinking);
        request.stream = true;
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        self.rx = Some(rx);
        self.cancel = Some(cancel.clone());
        self.phase = Phase::Waiting;
        let repaint = ctx.clone();
        let spawned = std::thread::Builder::new().name("figcraft-assistant".into()).spawn(move || {
            let model = DeepSeek::new(config);
            let result = {
                let mut relay = |event: ModelEvent| {
                    let _ = tx.send(Event::Delta(event));
                    repaint.request_repaint();
                };
                model.complete(&request, &mut relay, &cancel)
            };
            let _ = tx.send(Event::Done(result));
            repaint.request_repaint();
        });
        if let Err(e) = spawned {
            self.fail(trf("Couldn't start the Assistant: {error}", &[("error", &e.to_string())]));
        }
    }

    /// The browser has no model: say so instead of pretending to wait.
    #[cfg(target_arch = "wasm32")]
    fn send(&mut self, _ctx: &egui::Context, _config: DeepSeekConfig) {
        self.fail(tr("The Assistant needs the desktop app or figcraft-cli.").to_string());
    }

    fn fail(&mut self, message: String) {
        self.phase = Phase::Failed(message.clone());
        self.entries.push(Entry::Error(message));
        self.rx = None;
        self.cancel = None;
    }

    fn delta(&mut self, event: ModelEvent) {
        self.phase = Phase::Streaming;
        if !matches!(self.entries.last(), Some(Entry::Assistant { .. })) {
            self.entries.push(Entry::Assistant { text: String::new(), reasoning: String::new() });
        }
        if let Some(Entry::Assistant { text, reasoning }) = self.entries.last_mut() {
            match event {
                ModelEvent::Content(t) => text.push_str(&t),
                ModelEvent::Reasoning(t) => reasoning.push_str(&t),
            }
        }
    }

    /// Show the model's whole answer when no fragment arrived (a non-streamed or empty stream).
    fn finish_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        match self.entries.last_mut() {
            Some(Entry::Assistant { text: shown, .. }) if shown.is_empty() => shown.push_str(text),
            Some(Entry::Assistant { .. }) => {}
            _ => self.entries.push(Entry::Assistant { text: text.to_string(), reasoning: String::new() }),
        }
    }

    fn poll(&mut self, app: &mut FigApp, ctx: &egui::Context) {
        let Some(rx) = self.rx.take() else { return };
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(Event::Delta(event)) => self.delta(event),
                Ok(Event::Done(result)) => {
                    self.done(app, ctx, result);
                    finished = true;
                    break;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.fail(tr("The Assistant stopped unexpectedly.").to_string());
                    finished = true;
                    break;
                }
            }
        }
        if !finished {
            self.rx = Some(rx);
        }
    }

    fn done(&mut self, app: &mut FigApp, ctx: &egui::Context, result: Result<ChatResponse, ModelError>) {
        let Some(mut agent) = self.agent.take() else {
            self.fail(tr("The Assistant lost its conversation.").to_string());
            return;
        };
        match result {
            Err(ModelError::Cancelled) => {
                agent.cancel_pending_user();
                self.entries.push(Entry::Note(tr("Stopped.").to_string()));
                self.phase = Phase::Idle;
                self.cancel = None;
                self.agent = Some(agent);
            }
            Err(e) => {
                // The endpoint takes fewer tokens than the budget: lower it to what it says and ask
                // again, saying so, instead of stopping on a 400 the user can't act on.
                if let Some(note) = self.lower_budget(app, &mut agent, &e) {
                    self.entries.push(Entry::Note(note));
                    self.agent = Some(agent);
                    self.cancel = None;
                    let config = self.config();
                    self.send(ctx, config);
                    return;
                }
                self.agent = Some(agent);
                self.fail(e.to_string());
            }
            Ok(response) => match agent.accept(response) {
                Step::Done { text } => {
                    self.finish_text(&text);
                    self.phase = Phase::Idle;
                    self.cancel = None;
                    self.agent = Some(agent);
                }
                Step::Stopped { text } => {
                    self.finish_text(&text);
                    self.entries.push(Entry::Note(
                        tr("The Assistant used all its turns without a final answer. Send “continue” to let it keep going.").to_string(),
                    ));
                    self.phase = Phase::Idle;
                    self.cancel = None;
                    self.agent = Some(agent);
                }
                Step::Incomplete { text, cut_off } => {
                    self.finish_text(&text);
                    self.cancel = None;
                    // A cut-off plan or a silent answer left the file untouched: ask it to go on
                    // rather than leave the user waiting for work that never started.
                    if agent.nudge() {
                        self.agent = Some(agent);
                        let config = self.config();
                        self.send(ctx, config);
                        return;
                    }
                    let note = if cut_off {
                        tr("DeepSeek ran out of room while answering, so it stopped before finishing. Send “continue” to let it carry on.")
                    } else {
                        tr("DeepSeek answered with nothing and changed nothing. Send “continue”, or say what to do next.")
                    };
                    self.entries.push(Entry::Note(note.to_string()));
                    self.phase = Phase::Idle;
                    self.agent = Some(agent);
                }
                Step::Tools { calls, text } => {
                    self.finish_text(&text);
                    for call in calls {
                        self.run_tool(app, ctx, &mut agent, &call);
                    }
                    self.agent = Some(agent);
                    let config = self.config();
                    self.send(ctx, config);
                }
            },
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_tool(&mut self, app: &mut FigApp, ctx: &egui::Context, agent: &mut Agent, call: &ToolCall) {
        self.phase = Phase::Tool(call.name.clone());
        self.entries.push(Entry::Tool { name: call.name.clone(), ok: None, text: String::new() });
        let outcome = {
            let mut backend = UiBackend { app, ctx, document: self.document };
            execute_tool(&mut backend, call)
        };
        if let Some(Entry::Tool { ok, text, .. }) = self.entries.last_mut() {
            *ok = Some(!outcome.is_error);
            *text = format!("arguments: {}\n\nresult:\n{}", bounded(call.arguments.trim(), 2000), bounded(outcome.text.trim(), 4000));
        }
        if outcome.is_error {
            self.entries.push(Entry::Error(format!("{}: {}", call.name, first_line(&outcome.text))));
        }
        agent.tool_result(call, &outcome);
        self.phase = Phase::Waiting;
    }

    /// Draw the Assistant filling the left sidebar, the way a chat view sits docked in an editor's
    /// sidebar: a title row with the way out, the settings, what the model is doing, the thread
    /// scrolling, and the composer at the bottom. The sidebar's drag handle sets its width.
    fn draw_docked(&mut self, app: &mut FigApp, ui: &mut egui::Ui) {
        let t = Tokens::get(ui.ctx());
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tr("Assistant")).color(t.text).font(crate::theme::semibold(13.0)));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::button(ui, tr("Close"), false, true).clicked() {
                    self.open = false;
                }
            });
        });
        ui.add_space(2.0);
        self.header(app, ui);
        if self.show_settings {
            self.settings(app, ui);
        }
        self.status(ui);
        ui.add_space(4.0);
        let max = (ui.available_height() - COMPOSER_HEIGHT).max(0.0);
        egui::ScrollArea::vertical().max_height(max).stick_to_bottom(true).auto_shrink([false, false]).show(ui, |ui| {
            if self.entries.is_empty() {
                widgets::label(ui, tr("Ask the Assistant to design or change something in this file."), t.text_secondary);
            }
            for entry in &self.entries {
                draw_entry(ui, &t, entry);
            }
        });
        ui.separator();
        self.composer(ui, app, &ctx);
    }

    fn header(&mut self, app: &mut FigApp, ui: &mut egui::Ui) {
        let t = Tokens::get(ui.ctx());
        // Two rows, so the pickers fit the sidebar's width: a 240 pt sidebar holds them without
        // pushing the checkbox out of sight.
        ui.horizontal_wrapped(|ui| {
            let resp = widgets::dropdown(ui, tr(self.mode.label()), 88.0, None);
            a11y::combo(&resp, tr("Mode"), tr(self.mode.label()), a11y::menu_open(&resp));
            let mut picked = None;
            crate::right::menu_popup(&resp).show(|ui| {
                for mode in Mode::ALL {
                    let label = tr(mode.label());
                    if crate::right::menu_check(ui, self.mode == *mode, label, "") {
                        picked = Some(*mode);
                        ui.close();
                    }
                }
            });
            if let Some(mode) = picked {
                self.set_mode(app, mode);
            }
            let thinking = widgets::checkbox(ui, self.thinking, tr("Thinking"));
            if thinking {
                self.thinking = !self.thinking;
                self.save(app, json!({"thinking": self.thinking}));
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new(tr("Model")).color(t.text_secondary).font(regular(11.0)));
            let width = (ui.available_width() - 8.0).clamp(90.0, 180.0);
            let resp = widgets::dropdown(ui, &self.model, width, None);
            a11y::combo(&resp, tr("Model"), &self.model, a11y::menu_open(&resp));
            let mut picked = None;
            crate::right::menu_popup(&resp).show(|ui| {
                for model in figcraft_ai::MODELS {
                    if crate::right::menu_check(ui, self.model == *model, model, "") {
                        picked = Some(*model);
                        ui.close();
                    }
                }
            });
            if let Some(model) = picked {
                self.model = model.to_string();
                self.save(app, json!({"model": self.model}));
            }
        });
        // What the picked mode means, where it is picked: a mode that only looks says so.
        if !self.mode.edits() {
            widgets::label(ui, tr(self.mode.hint()), t.text_secondary);
        }
        ui.horizontal_wrapped(|ui| {
            if widgets::button(ui, tr("New chat"), false, !self.entries.is_empty()).clicked() {
                self.new_chat();
            }
            let copy = widgets::button(ui, tr("Copy chat"), false, !self.entries.is_empty());
            if copy.clicked() {
                self.copy_transcript(app);
            }
            if widgets::button(ui, tr("Settings"), false, true).clicked() {
                self.show_settings = !self.show_settings;
            }
        });
    }

    fn settings(&mut self, app: &mut FigApp, ui: &mut egui::Ui) {
        let t = Tokens::get(ui.ctx());
        ui.add_space(4.0);
        if std::env::var(figcraft_ai::API_KEY_ENV).is_ok_and(|k| !k.trim().is_empty()) {
            widgets::label(ui, tr("Using DEEPSEEK_API_KEY from the environment."), t.text_secondary);
        }
        let name = tr("DeepSeek API key");
        let width = ui.available_width();
        if let Some(key) = widgets::secret_field_named(ui, egui::Id::unique("assistant_key"), &self.key, width, name) {
            self.key = key;
            self.save_remember(app);
        }
        if widgets::checkbox(ui, self.remember_key, tr("Remember this key on this computer")) {
            self.remember_key = !self.remember_key;
            self.save_remember(app);
        }
        let note = match (self.remember_key, self.key.trim().is_empty()) {
            (false, _) => tr("Not saved: it is forgotten when the app closes."),
            (true, true) => tr("Kept in FigCraft's settings file on this computer once you type it."),
            (true, false) => tr("Saved in FigCraft's settings file on this computer."),
        };
        widgets::label(ui, note, t.text_secondary);
        ui.horizontal(|ui| {
            ui.label(tr("Endpoint"));
            let name = tr("Endpoint");
            if let Some(url) = widgets::text_field_named(ui, egui::Id::unique("assistant_endpoint"), &self.endpoint, 200.0, name) {
                self.endpoint = url;
                self.save(app, json!({"endpoint": self.endpoint}));
            }
        });
        widgets::label(ui, tr("An OpenAI-compatible endpoint; leave empty for DeepSeek."), t.text_secondary);
        ui.horizontal(|ui| {
            let name = tr("Max answer tokens");
            ui.label(egui::RichText::new(name).color(t.text_secondary).font(regular(11.0)));
            let id = egui::Id::unique("assistant_max_tokens");
            let edit = widgets::number_field(ui, id, crate::widgets::Lead::Named(name), Some(f64::from(self.max_tokens)), "", 96.0, 1024.0);
            if let Some(crate::widgets::FieldEdit::Set(v)) = edit {
                let n = v.round().clamp(f64::from(figcraft_engine::ASSISTANT_TOKENS_MIN), f64::from(figcraft_engine::ASSISTANT_TOKENS_MAX)) as u32;
                if n != self.max_tokens {
                    self.max_tokens = n;
                    self.save_max_tokens(app);
                }
            }
        });
        widgets::label(
            ui,
            tr(
                "The most tokens one answer may use, thinking included: 16 to 1,000,000. An endpoint with a smaller limit lowers it to what it allows.",
            ),
            t.text_secondary,
        );
        widgets::label(ui, tr("Your prompts and the tool results are sent to DeepSeek."), t.text_secondary);
    }

    fn status(&mut self, ui: &mut egui::Ui) {
        let t = Tokens::get(ui.ctx());
        let (text, color) = match &self.phase {
            Phase::Idle => (tr("Ready").to_string(), t.text_secondary),
            Phase::Waiting => (tr("DeepSeek is working…").to_string(), t.text_secondary),
            Phase::Streaming => (tr("Thinking…").to_string(), t.text_secondary),
            Phase::Tool(name) => (trf("Running {tool}…", &[("tool", name)]), t.text_secondary),
            Phase::Failed(e) => (e.clone(), t.danger),
        };
        let resp = ui.label(egui::RichText::new(text.clone()).color(color).font(regular(11.0)));
        a11y::live_status(&resp, tr("Assistant status"), &text);
    }

    fn composer(&mut self, ui: &mut egui::Ui, app: &mut FigApp, ctx: &egui::Context) {
        let t = Tokens::get(ui.ctx());
        let id = egui::Id::unique("assistant_input");
        if self.focus {
            ui.memory_mut(|m| m.request_focus(id));
            self.focus = false;
        }
        let focused = ui.memory(|m| m.has_focus(id));
        let send_key = focused && ui.input_mut(|i| !i.modifiers.shift && i.consume_key(egui::Modifiers::NONE, egui::Key::Enter));
        // The `#` and `/` pickers: what the prompt is naming, and the tasks this mode offers.
        let typing = focused.then(|| trailing_sigil(&self.input)).flatten();
        let mut field: Option<egui::Response> = None;
        egui::Frame::NONE.fill(t.field).corner_radius(egui::CornerRadius::same(6)).inner_margin(egui::Margin::symmetric(8, 6)).show(ui, |ui| {
            let resp = ui.add(
                egui::TextEdit::multiline(&mut self.input)
                    .id(id)
                    .hint_text(tr("Ask the Assistant…"))
                    .font(regular(12.0))
                    .frame(egui::Frame::NONE)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            );
            a11y::text(&resp, tr("Message"), &self.input);
            field = Some(resp);
        });
        // The references the prompt names, as chips: what will be read from the file when it is
        // sent, and a way to drop one.
        let mut drop_reference = None;
        let references = self.references();
        if !references.is_empty() {
            ui.horizontal_wrapped(|ui| {
                for reference in &references {
                    let label = format!("{} ×", tr(reference.label()));
                    if widgets::button(ui, &label, false, true).clicked() {
                        drop_reference = Some(*reference);
                    }
                }
            });
        }
        if let Some(reference) = drop_reference {
            self.remove_reference(reference);
        }
        match typing {
            Some(('#', query)) => {
                let mut chosen = None;
                if let Some(anchor) = &field {
                    crate::right::typing_popup(anchor).show(|ui| {
                        for reference in Reference::ALL.iter().filter(|r| matches_query(r.id(), &query)) {
                            let label = tr(reference.label());
                            if crate::right::menu_check(ui, false, label, &reference.token()) {
                                chosen = Some(*reference);
                                ui.close();
                            }
                            ui.label(egui::RichText::new(tr(reference.hint())).color(t.text_secondary).font(regular(10.0)));
                        }
                    });
                }
                if let Some(reference) = chosen {
                    self.replace_typing(&reference.token());
                }
            }
            Some(('/', query)) => {
                let mut chosen = None;
                if let Some(anchor) = &field {
                    crate::right::typing_popup(anchor).show(|ui| {
                        if !self.mode.edits() {
                            ui.label(egui::RichText::new(tr("Tasks change the file: switch to Agent.")).color(t.text_secondary).font(regular(10.0)));
                        }
                        for recipe in RECIPES.iter().filter(|r| matches_query(r.id, &query) && self.mode.edits()) {
                            if crate::right::menu_check(ui, false, tr(recipe.label), "") {
                                chosen = Some(recipe);
                                ui.close();
                            }
                            ui.label(egui::RichText::new(tr(recipe.hint)).color(t.text_secondary).font(regular(10.0)));
                        }
                    });
                }
                if let Some(recipe) = chosen {
                    self.input = recipe.prompt.to_string();
                    self.add_reference(recipe.reference);
                    self.focus = true;
                }
            }
            _ => {}
        }
        let has_file = app.session.ds().is_ok();
        let desktop = !cfg!(target_arch = "wasm32");
        let ready = desktop && has_file && self.has_key() && !self.input.trim().is_empty() && !self.busy();
        ui.horizontal(|ui| {
            if widgets::button(ui, tr("Send"), true, ready).clicked() || (send_key && ready) {
                self.start(app, ctx);
            }
            if self.busy() && widgets::button(ui, tr("Stop"), false, true).clicked() {
                self.stop();
            }
            let picked = has_file && !app.session.selection().is_empty();
            let add = widgets::button(ui, tr("Add selection"), false, picked);
            if picked {
                add.clone().on_hover_text(tr("Send the layers selected in the file as context"));
            }
            if add.clicked() {
                self.add_reference(Reference::Selection);
            }
        });
        if !desktop {
            widgets::label(ui, tr("The Assistant needs the desktop app or figcraft-cli."), t.text_secondary);
        } else if !has_file {
            widgets::label(ui, tr("Open a file to use the Assistant."), t.text_secondary);
        } else if !self.has_key() {
            let missing = trf("Set {variable} or type a key in Settings.", &[("variable", &figcraft_ai::API_KEY_ENV)]);
            widgets::label(ui, &missing, t.warning);
        } else {
            widgets::label(ui, tr("Enter sends; Shift+Enter adds a line. Type # for a reference, / for a task."), t.text_tertiary);
        }
    }

    /// Replace the `#`/`/` word being typed with what the picker chose, keeping the rest.
    pub(crate) fn replace_typing(&mut self, token: &str) {
        let trimmed = self.input.trim_end();
        let cut = trimmed.rfind(char::is_whitespace).map_or(0, |i| i + 1);
        self.input = format!("{}{token} ", &trimmed[..cut]);
        self.focus = true;
    }
}

/// Whether a picker entry matches what is typed after the sigil.
fn matches_query(id: &str, query: &str) -> bool {
    id.to_lowercase().contains(&query.to_lowercase())
}

fn draw_entry(ui: &mut egui::Ui, t: &Tokens, entry: &Entry) {
    match entry {
        Entry::User(text) => {
            ui.add_space(6.0);
            let resp = ui.add(egui::Label::new(egui::RichText::new(trf("You: {text}", &[("text", text)])).color(t.text).font(regular(12.0))).wrap());
            resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Label, true, text.as_str()));
        }
        Entry::Assistant { text, reasoning } => {
            ui.add_space(6.0);
            if !text.is_empty() {
                let resp = ui.add(egui::Label::new(egui::RichText::new(text).color(t.text).font(regular(12.0))).wrap());
                resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Label, true, text.as_str()));
            }
            if !reasoning.is_empty() {
                egui::CollapsingHeader::new(tr("Reasoning")).id_salt(("assistant_reasoning", text.len(), reasoning.len())).show(ui, |ui| {
                    ui.add(egui::Label::new(egui::RichText::new(reasoning).color(t.text_secondary).font(regular(11.0))).wrap());
                });
            }
        }
        Entry::Tool { name, ok, .. } => {
            let (mark, color) = match ok {
                Some(true) => ("✓", t.success),
                Some(false) => ("!", t.danger),
                None => ("…", t.text_secondary),
            };
            let label = format!("{mark} {name}");
            let resp = ui.label(egui::RichText::new(&label).color(color).font(regular(11.0)));
            a11y::status(&resp, &label, tr("Assistant tool"));
        }
        Entry::Note(text) => widgets::label(ui, text, t.text_secondary),
        Entry::Error(text) => widgets::label(ui, text, t.danger),
    }
}

/// The first line of a tool's error, bounded, for the panel's tool row.
#[cfg(not(target_arch = "wasm32"))]
fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or_default().trim();
    let mut out: String = line.chars().take(160).collect();
    if line.chars().count() > 160 {
        out.push('…');
    }
    out
}

/// `text` cut to `max_chars` characters, with a note when something was left out.
#[cfg(not(target_arch = "wasm32"))]
fn bounded(text: &str, max_chars: usize) -> String {
    let mut out: String = text.chars().take(max_chars).collect();
    if text.chars().count() > max_chars {
        out.push_str("\n… (truncated)");
    }
    out
}

/// The app's session as a tool backend: only the pinned document, only design methods, and every
/// command runs as the Assistant (so comments are signed and its presence shows).
#[cfg(not(target_arch = "wasm32"))]
struct UiBackend<'a> {
    app: &'a mut FigApp,
    ctx: &'a egui::Context,
    document: Option<u64>,
}

#[cfg(not(target_arch = "wasm32"))]
impl figcraft_ai::tools::Backend for UiBackend<'_> {
    fn call(&mut self, method: &str, mut params: Value) -> Result<Value, String> {
        match method {
            "engine.execute" | "document.inspect" | "node.get" | "ui.render" | "engine.commands" => {}
            other => return Err(format!("the in-app Assistant can't call `{other}`")),
        }
        if method == "engine.execute"
            && let Some(reason) = refusal(&params)
        {
            return Err(reason);
        }
        if method != "engine.commands"
            && let Some(fields) = params.as_object_mut()
        {
            if let Some(document) = self.document {
                fields.insert("document".into(), json!(document));
            }
            if method == "engine.execute" {
                fields.insert("agent".into(), json!(AGENT_NAME));
            }
        }
        crate::control::handle(self.app, self.ctx, method, &params)
    }

    fn has_ui(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        "the FigCraft app".into()
    }

    fn screenshot_png(&mut self) -> Result<Vec<u8>, String> {
        self.render_png(&[], 1.0)
    }

    fn window_png(&mut self, _window: u64) -> Result<Vec<u8>, String> {
        self.render_png(&[], 1.0)
    }
}

/// The commands the in-app Assistant may not run: anything that opens, closes or saves files,
/// leaves the app, changes preferences, windows or online state, or starts another Assistant.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn refusal(params: &Value) -> Option<String> {
    let command = params.get("command").and_then(Value::as_str)?;
    if let Some(commands) = params.get("params").and_then(|p| p.get("commands")).and_then(Value::as_array)
        && let Some(id) = commands.iter().find_map(|c| c.get("command").and_then(Value::as_str).filter(|id| denied(id)))
    {
        return Some(refused(id));
    }
    denied(command).then(|| refused(command))
}

#[cfg(not(target_arch = "wasm32"))]
fn denied(id: &str) -> bool {
    if matches!(id, "file.export" | "doc.list") || id.starts_with("comment.") {
        return false;
    }
    id.starts_with("file.")
        || id.starts_with("app.")
        || id.starts_with("window.")
        || id.starts_with("online.")
        || id.starts_with("collab.")
        || id.starts_with("plugin.")
        || id.starts_with("library.")
        || id.starts_with("pref.")
        || id.starts_with("assistant.")
        || id.starts_with("variable.import")
        || id.starts_with("variable.export")
        || matches!(id, "doc.switch" | "doc.nextTab" | "doc.previousTab" | "doc.home")
}

#[cfg(not(target_arch = "wasm32"))]
fn refused(id: &str) -> String {
    trf("The in-app Assistant can't run `{command}`: it only edits the file you have open.", &[("command", &id)])
}

/// Keep the conversation moving every frame, whether or not the view is showing: answers and tool
/// calls arrive here, and a hidden view never loses or stops a run. The drawing happens inside the
/// left sidebar ([`view`]), which the app lays out earlier in the frame.
pub fn show(app: &mut FigApp, ctx: &egui::Context) {
    let mut state = std::mem::take(&mut app.ui.assistant);
    // The settings hold the mode: one set from outside (`pref.assistant {agent}`) reaches a panel
    // that is already open, so automation can do everything the picker does.
    state.apply_mode(app, Mode::from_id(&app.session.prefs.assistant_agent).unwrap_or_default());
    state.poll(app, ctx);
    app.ui.assistant = state;
}

/// Draw the Assistant in the left sidebar, filling it (the panel calls this when the view is the
/// one showing). The state is taken out of `FigApp` so the composer can borrow the app.
pub fn view(app: &mut FigApp, ui: &mut egui::Ui) {
    let mut state = std::mem::take(&mut app.ui.assistant);
    state.draw_docked(app, ui);
    app.ui.assistant = state;
}

/// Open the Assistant (the `assistant` command and the rail's button): the left sidebar shows it,
/// beside the other views, at whatever width the sidebar has — widened to a readable chat width
/// when the file tree's 240 pt would squeeze it.
pub fn open(app: &mut FigApp) {
    app.ui.assistant.open = true;
    app.ui.assistant.focus = true;
    app.ui.left_width = app.ui.left_width.max(CHAT_WIDTH);
    // One view at a time in the sidebar.
    app.ui.versions = Default::default();
    app.ui.open_left_panel();
}
