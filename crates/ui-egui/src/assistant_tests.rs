//! The Assistant panel driven headless: it opens from its command, refuses to send without a key,
//! and stopping a request ends it and drops the prompt the model never answered.

use egui::accesskit::Role;
use figcraft_engine::Session;
use serde_json::json;

use crate::assistant::{AssistantUi, Entry, Phase};
use crate::layers_drag_tests::Ui;
/// Change the panel's state between frames.
fn run(u: &mut Ui, f: impl FnOnce(&mut AssistantUi)) {
    let mut state = std::mem::take(&mut u.app.ui.assistant);
    f(&mut state);
    u.app.ui.assistant = state;
}

#[test]
fn the_command_opens_the_panel_and_it_starts_no_conversation() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    u.app.run("app.assistant", json!({}));
    assert!(u.app.ui.assistant.open);
    assert!(!u.app.ui.left_collapsed, "the sidebar opens to show it");
    u.ctx.enable_accesskit();
    u.frame(vec![]);
    let tree = u.frame_output(vec![]).platform_output.accesskit_update.expect("an accessibility tree");
    let nodes: Vec<&egui::accesskit::Node> = tree.nodes.iter().map(|(_, n)| n).collect();
    // Docked in the sidebar, the way a chat view sits beside an editor's file tree: a named pane
    // inside the sidebar's width, with its controls in it, rather than a floating window.
    let pane = nodes.iter().find(|n| n.role() == Role::Pane && n.label().is_some_and(|l| l == "Assistant")).expect("the sidebar shows the Assistant");
    let b = pane.bounds().expect("the pane has bounds");
    let sidebar_right = crate::panels::layers::RAIL_WIDTH + u.app.ui.left_width;
    assert!(b.x1 as f32 <= sidebar_right + 2.0, "the panel ends at the sidebar's edge ({}, {sidebar_right})", b.x1);
    assert!(nodes.iter().any(|n| n.role() == Role::Button && n.label().is_some_and(|l| l == "New chat")), "its controls are in it");
    assert!(u.app.ui.assistant.agent.is_none(), "opening starts no conversation");
}

#[test]
fn another_sidebar_view_takes_the_panel_and_the_conversation_survives_it() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    run(&mut u, |s| {
        s.open = true;
        s.entries.push(Entry::User("make a card".into()));
    });
    click(&mut u, Role::Tab, "File");
    assert!(!u.app.ui.assistant.open, "the file tree takes the sidebar");
    assert_eq!(u.app.ui.assistant.entries.len(), 1, "the conversation is kept, not thrown away");
    click(&mut u, Role::Button, "Assistant");
    assert!(u.app.ui.assistant.open, "and the rail brings it back as it was");
    assert_eq!(u.app.ui.assistant.entries.len(), 1);
}

#[test]
fn a_blank_key_cannot_send() {
    assert!(!figcraft_ai::key_available("", None));
    assert!(figcraft_ai::key_available("sk-typed", None));
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    run(&mut u, |s| {
        s.open = true;
        s.input = "make a card".into();
        s.key = "  ".into();
    });
    assert!(!u.app.ui.assistant.has_key());
    assert_eq!(u.app.ui.assistant.input, "make a card", "the prompt stays for after the key is set");
    assert_eq!(u.app.ui.assistant.phase, Phase::Idle);
}

#[test]
fn stopping_a_request_ends_it_and_drops_the_unanswered_prompt() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    run(&mut u, |s| {
        s.open = true;
        let mut agent = figcraft_ai::Agent::new("system", vec![], figcraft_ai::AgentConfig::default());
        agent.user("make a card");
        s.agent = Some(agent);
        s.phase = Phase::Waiting;
        s.stop();
    });
    let s = &u.app.ui.assistant;
    assert_eq!(s.phase, Phase::Idle);
    assert!(!s.busy());
    let agent = s.agent.as_ref().unwrap();
    assert!(agent.messages().is_empty(), "the prompt the model never answered is gone");
    assert!(s.entries.iter().any(|e| matches!(e, Entry::Note(t) if t.contains("Stopped"))));
}

#[test]
fn clicking_thinking_turns_it_off_and_it_stays_off() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    run(&mut u, |s| s.open = true);
    assert!(u.app.ui.assistant.thinking, "thinking starts on");
    u.ctx.enable_accesskit();
    u.frame(vec![]);
    let tree = u.frame_output(vec![]).platform_output.accesskit_update.expect("an accessibility tree");
    let node = tree
        .nodes
        .iter()
        .map(|(_, n)| n)
        .find(|n| n.role() == Role::CheckBox && n.label().is_some_and(|l| l == "Thinking"))
        .expect("the Thinking checkbox is in the panel");
    let b = node.bounds().expect("the checkbox has bounds");
    let at = egui::pos2(((b.x0 + b.x1) / 2.0) as f32, ((b.y0 + b.y1) / 2.0) as f32);
    let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    u.frame(vec![egui::Event::PointerMoved(at), button(true), button(false)]);
    u.frame(vec![]);
    assert!(!u.app.ui.assistant.thinking, "the click turns Thinking off");
    u.frame(vec![]);
    assert!(!u.app.ui.assistant.thinking, "and it stays off on the next frame");
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn the_panel_takes_its_saved_settings_from_the_preferences() {
    let mut s = Session::empty();
    s.prefs.assistant_model = "deepseek-v4-pro".into();
    s.prefs.assistant_thinking = false;
    s.prefs.assistant_endpoint = "http://127.0.0.1:11434/v1".into();
    s.prefs.assistant_key = "sk-saved".into();
    let u = Ui::new(s);
    let a = &u.app.ui.assistant;
    assert_eq!((a.model.as_str(), a.thinking), ("deepseek-v4-pro", false));
    assert_eq!(a.endpoint, "http://127.0.0.1:11434/v1");
    assert_eq!(a.key, "sk-saved");
    assert!(a.remember_key, "a saved key shows as remembered");
}

#[test]
fn the_transcript_holds_the_whole_conversation() {
    let mut u = Ui::new(Session::empty());
    run(&mut u, |s| {
        s.model = "deepseek-v4-pro".into();
        s.thinking = false;
        s.entries.push(Entry::User("make a card".into()));
        s.entries.push(Entry::Assistant { text: "On it.".into(), reasoning: "a frame first".into() });
        s.entries.push(Entry::Tool {
            name: "create_node".into(),
            ok: Some(true),
            text: "arguments: {\"type\":\"frame\"}\n\nresult:\n{\"id\":2}".into(),
        });
        s.entries.push(Entry::Tool { name: "node.set".into(), ok: Some(false), text: "arguments: {}\n\nresult:\nError: no such node 9".into() });
        s.entries.push(Entry::Note("The file changed: the conversation started over.".into()));
        s.entries.push(Entry::Error("batch: invalid arguments".into()));
    });
    let text = u.app.ui.assistant.transcript();
    let order = [
        "## You\n\nmake a card",
        "## Assistant\n\nOn it.",
        "> a frame first",
        "## Tool: create_node (ok)",
        "## Tool: node.set (failed)",
        "> The file changed",
        "## Error\n\nbatch",
    ];
    let mut at = 0;
    for part in order {
        let found = text[at..].find(part).unwrap_or_else(|| panic!("missing or out of order: {part:?} in\n{text}"));
        at += found;
    }
    assert!(text.contains("Model: deepseek-v4-pro · Thinking: off"), "{text}");
    assert!(text.contains("\"type\":\"frame\""), "the tool arguments are kept: {text}");
    assert!(text.contains("no such node 9"), "the tool result is kept: {text}");
}

#[derive(Clone, Default)]
struct Clipboard(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

impl crate::Services for Clipboard {
    fn clipboard_set_text(&mut self, s: &str) {
        self.0.lock().unwrap().push(s.to_string());
    }
}

#[test]
fn the_copy_chat_button_writes_the_conversation_to_the_clipboard() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    let clipboard = Clipboard::default();
    u.app.services = Box::new(clipboard.clone());
    run(&mut u, |s| {
        s.open = true;
        s.entries.push(Entry::User("make a card".into()));
        s.entries.push(Entry::Assistant { text: "On it.".into(), reasoning: String::new() });
    });
    u.ctx.enable_accesskit();
    u.frame(vec![]);
    let tree = u.frame_output(vec![]).platform_output.accesskit_update.expect("an accessibility tree");
    let labels: Vec<&str> = tree.nodes.iter().filter_map(|(_, n)| n.label()).collect();
    let node = tree
        .nodes
        .iter()
        .map(|(_, n)| n)
        .find(|n| n.role() == Role::Button && n.label().is_some_and(|l| l == "Copy chat"))
        .unwrap_or_else(|| panic!("the Copy chat button is in the panel: {labels:?}"));
    let b = node.bounds().expect("the button has bounds");
    let at = egui::pos2(((b.x0 + b.x1) / 2.0) as f32, ((b.y0 + b.y1) / 2.0) as f32);
    let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    u.frame(vec![egui::Event::PointerMoved(at), button(true), button(false)]);
    let copied = clipboard.0.lock().unwrap().clone();
    assert_eq!(copied.len(), 1, "one copy: {copied:?}");
    assert!(copied[0].contains("## You\n\nmake a card"), "{copied:?}");
    assert!(copied[0].contains("## Assistant\n\nOn it."), "{copied:?}");
    assert!(u.app.ui.last_error.is_none(), "{:?}", u.app.ui.last_error);
}

#[test]
fn the_in_app_assistant_edits_the_open_file_and_nothing_that_leaves_it() {
    use crate::assistant::refusal;
    assert!(refusal(&json!({"command": "node.create"})).is_none());
    assert!(refusal(&json!({"command": "file.export", "params": {"format": "png"}})).is_none());
    for denied in ["file.open", "file.save", "file.close", "app.preferences", "window.new", "doc.switch", "online.publish", "pref.theme"] {
        let reason = refusal(&json!({"command": denied})).unwrap_or_else(|| panic!("{denied} is refused"));
        assert!(reason.contains(denied), "{reason}");
    }
    let batched = refusal(&json!({"command": "command.batch", "params": {"commands": [{"command": "node.create"}, {"command": "file.save"}]}}));
    assert!(batched.is_some_and(|r| r.contains("file.save")), "a batch is checked step by step");
}

/// Click the panel control whose accessibility label is `label`.
fn click(u: &mut Ui, role: Role, label: &str) {
    u.ctx.enable_accesskit();
    u.frame(vec![]);
    let tree = u.frame_output(vec![]).platform_output.accesskit_update.expect("an accessibility tree");
    let node = tree
        .nodes
        .iter()
        .map(|(_, n)| n)
        .find(|n| n.role() == role && n.label().is_some_and(|l| l == label))
        .unwrap_or_else(|| panic!("{label} is in the panel"));
    let b = node.bounds().expect("the control has bounds");
    let at = egui::pos2(((b.x0 + b.x1) / 2.0) as f32, ((b.y0 + b.y1) / 2.0) as f32);
    let button = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    u.frame(vec![egui::Event::PointerMoved(at), button(true), button(false)]);
    u.frame(vec![]);
}

#[test]
fn clicking_remember_keeps_the_answer_it_was_left_at() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    run(&mut u, |s| {
        s.open = true;
        s.show_settings = true;
        s.key = "sk-typed".into();
    });
    assert!(!u.app.ui.assistant.remember_key, "the box starts empty");
    click(&mut u, Role::CheckBox, "Remember this key on this computer");
    assert!(u.app.ui.assistant.remember_key, "a click ticks it");
    assert_eq!(u.app.session.prefs.assistant_key, "sk-typed", "the tick saves the key");
    assert!(u.app.session.prefs.assistant_remember);
    click(&mut u, Role::CheckBox, "Remember this key on this computer");
    assert!(!u.app.ui.assistant.remember_key, "a second click clears it");
    assert_eq!(u.app.session.prefs.assistant_key, "", "clearing forgets the key");
    assert!(!u.app.session.prefs.assistant_remember, "and the answer is kept as no");
}

#[test]
fn the_answer_survives_a_restart_before_a_key_is_typed() {
    let mut s = Session::empty();
    s.prefs.assistant_remember = true;
    let mut a = crate::assistant::AssistantUi::default();
    a.apply_prefs(&s.prefs);
    assert!(a.remember_key, "the box comes back ticked with no key saved yet");
    assert!(a.key.is_empty());
}

#[test]
fn the_selection_is_a_reference_in_the_prompt_and_resolves_to_the_layers() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    let made = u.app.session.execute("node.create", &json!({"type": "rectangle", "name": "Card"})).unwrap();
    let id = made["id"].as_u64().expect("the layer's id");
    u.app.session.set_selection(vec![figcraft_doc::NodeId(id)]);
    run(&mut u, |s| s.open = true);
    click(&mut u, Role::Button, "Add selection");
    let a = &u.app.ui.assistant;
    assert_eq!(a.input, "#selection ", "the prompt itself says what it carries");
    assert_eq!(a.references(), vec![crate::assistant::Reference::Selection]);
    let resolved = a.resolved_prompt(&u.app, "make it blue #selection");
    assert!(resolved.starts_with("make it blue"), "{resolved}");
    assert!(resolved.contains("Card") && resolved.contains(&format!("id {id}")), "{resolved}");
    // The chip's × takes the token out and leaves the rest of the prompt alone.
    run(&mut u, |s| s.input = "make #selection it blue".to_string());
    run(&mut u, |s| s.remove_reference(crate::assistant::Reference::Selection));
    assert_eq!(u.app.ui.assistant.input, "make it blue", "only the token goes");
    assert!(u.app.ui.assistant.resolved_prompt(&u.app, "make it blue").ends_with("blue"), "nothing attached, nothing added");
}

#[test]
fn only_known_tokens_are_references_and_the_picker_appears_while_typing_one() {
    use crate::assistant::{Reference, references_in, trailing_sigil};
    assert_eq!(references_in("#page and #selection, then #styles."), vec![Reference::Page, Reference::Selection, Reference::Styles]);
    assert!(references_in("#FFFFFF is a colour, #nope is nothing").is_empty(), "colours and unknown words stay text");
    assert_eq!(references_in("#page #page"), vec![Reference::Page], "no repeats");
    assert_eq!(trailing_sigil("check the #pa"), Some(('#', "pa".into())));
    assert_eq!(trailing_sigil("/a11y"), Some(('/', "a11y".into())));
    assert_eq!(trailing_sigil("done #page more"), None, "only the word being typed");
    assert_eq!(trailing_sigil("#FFFFFF"), Some(('#', "ffffff".into())), "typing a colour does not match a reference");
}

#[test]
fn the_references_read_the_file_as_it_is_now() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    u.app.session.execute("node.create", &json!({"type": "frame", "name": "Login"})).unwrap();
    let a = crate::assistant::AssistantUi::default();
    let page = a.resolved_prompt(&u.app, "#page");
    assert!(page.contains("Login") && page.contains("id "), "the page lists its top-level layers: {page}");
    let file = a.resolved_prompt(&u.app, "#file");
    assert!(file.contains("1 page") && file.contains("layer"), "{file}");
    // The frame just made is selected, so #selection lists it; with nothing selected it says so.
    let selected = a.resolved_prompt(&u.app, "#selection");
    assert!(selected.contains("Login"), "{selected}");
    u.app.session.set_selection(vec![]);
    let empty = a.resolved_prompt(&u.app, "#selection");
    assert!(empty.contains("nothing is selected"), "{empty}");
}

#[test]
fn ask_and_plan_are_offered_without_a_tool_that_can_change_the_file() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    run(&mut u, |s| s.open = true);
    let mut state = std::mem::take(&mut u.app.ui.assistant);
    state.set_mode(&mut u.app, figcraft_ai::Mode::Ask);
    u.app.ui.assistant = state;
    assert_eq!(u.app.ui.assistant.mode, figcraft_ai::Mode::Ask);
    assert_eq!(u.app.session.prefs.assistant_agent, "ask", "the mode is saved");
    // A conversation gets the rules and the tools of its mode.
    let mut state = std::mem::take(&mut u.app.ui.assistant);
    let (system, tools) = crate::assistant::rules("Login", figcraft_ai::Mode::Ask);
    state.agent = Some(figcraft_ai::Agent::new(system, tools, figcraft_ai::AgentConfig::default()));
    u.app.ui.assistant = state;
    let request = u.app.ui.assistant.agent.as_ref().expect("a conversation").request("deepseek-flash", true);
    let names: Vec<&str> = request.tools.iter().filter_map(|t| t["function"]["name"].as_str()).collect();
    assert!(names.contains(&"get_node"), "{names:?}");
    for editing in ["run_command", "create_node", "set_properties", "batch", "undo"] {
        assert!(!names.contains(&editing), "Ask must not offer {editing}: {names:?}");
    }
    assert!(request.messages[0].content.contains("Ask mode"), "and the rules say so");
    // Switching back gives the editing tools again, keeping the thread.
    let mut state = std::mem::take(&mut u.app.ui.assistant);
    state.set_mode(&mut u.app, figcraft_ai::Mode::Agent);
    u.app.ui.assistant = state;
    let request = u.app.ui.assistant.agent.as_ref().expect("a conversation").request("deepseek-flash", true);
    let names: Vec<&str> = request.tools.iter().filter_map(|t| t["function"]["name"].as_str()).collect();
    assert!(names.contains(&"run_command"), "{names:?}");
}

#[test]
fn a_new_chat_forgets_the_thread_and_keeps_the_settings() {
    let mut u = Ui::new(Session::empty());
    u.app.session.execute("file.new", &json!({})).unwrap();
    run(&mut u, |s| {
        s.open = true;
        s.model = "deepseek-v4-pro".into();
        s.thinking = false;
        s.input = "half a prompt".into();
        s.entries.push(Entry::User("make a card".into()));
        s.agent = Some(figcraft_ai::Agent::new("system", vec![], figcraft_ai::AgentConfig::default()));
    });
    click(&mut u, Role::Button, "New chat");
    let a = &u.app.ui.assistant;
    assert!(a.entries.is_empty() && a.input.is_empty() && a.agent.is_none(), "the thread and the draft go");
    assert_eq!((a.model.as_str(), a.thinking), ("deepseek-v4-pro", false), "the settings stay");
}
