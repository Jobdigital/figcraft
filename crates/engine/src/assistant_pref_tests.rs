//! The Assistant's saved settings (`pref.assistant`): the endpoint and model are ordinary
//! preferences, the API key is kept only when asked for and never comes back in a result or a log.

use serde_json::json;

use crate::{Prefs, Session};

#[test]
fn the_endpoint_model_and_thinking_are_kept() {
    let mut s = Session::new();
    let r = s.execute("pref.assistant", &json!({"endpoint": "http://127.0.0.1:11434/v1", "model": "deepseek-v4-pro", "thinking": false})).unwrap();
    assert_eq!(r["endpoint"], "http://127.0.0.1:11434/v1");
    assert_eq!(r["model"], "deepseek-v4-pro");
    assert_eq!(r["thinking"], false);
    assert_eq!(s.prefs.assistant_endpoint, "http://127.0.0.1:11434/v1");
    assert_eq!(s.prefs.assistant_model, "deepseek-v4-pro");
    assert!(!s.prefs.assistant_thinking);
    // An empty endpoint goes back to DeepSeek's.
    s.execute("pref.assistant", &json!({"endpoint": ""})).unwrap();
    assert!(s.prefs.assistant_endpoint.is_empty());
}

#[test]
fn the_key_is_saved_only_when_asked_and_never_read_back() {
    let mut s = Session::new();
    let r = s.execute("pref.assistant", &json!({"key": "sk-secret", "remember": true})).unwrap();
    assert_eq!(r["keySaved"], true);
    assert!(!r.to_string().contains("sk-secret"), "a result never carries the key: {r}");
    assert_eq!(s.prefs.assistant_key, "sk-secret");
    assert!(s.prefs_unsaved, "the settings are written at once, not only when the app quits");
    // Asking to forget it clears it.
    let r = s.execute("pref.assistant", &json!({"remember": false})).unwrap();
    assert_eq!(r["keySaved"], false);
    assert!(s.prefs.assistant_key.is_empty());
    // An empty key with `remember` also forgets it.
    s.execute("pref.assistant", &json!({"key": "sk-secret", "remember": true})).unwrap();
    s.execute("pref.assistant", &json!({"key": "", "remember": true})).unwrap();
    assert!(s.prefs.assistant_key.is_empty());
    // Asking to keep a key before typing one keeps the answer, so the box comes back ticked.
    let kept = s.execute("pref.assistant", &json!({"remember": true})).unwrap();
    assert_eq!(kept["remember"], true);
    assert_eq!(kept["keySaved"], false, "the answer is kept without a key");
    assert!(s.prefs.assistant_remember, "the answer is kept even with no key yet");
    assert!(s.execute("pref.assistant", &json!({"remember": false})).unwrap()["remember"] == false);
    assert!(!s.prefs.assistant_remember, "forgetting the key forgets the answer too");
    // A key without `remember` is refused, and leaves the saved one alone.
    s.execute("pref.assistant", &json!({"key": "sk-kept", "remember": true})).unwrap();
    let e = s.execute("pref.assistant", &json!({"key": "sk-other"})).unwrap_err().to_string();
    assert!(e.contains("remember: true"), "{e}");
    assert_eq!(s.prefs.assistant_key, "sk-kept");
    assert!(!e.contains("sk-other"), "the error never carries the key: {e}");
}

#[test]
fn the_answer_budget_is_kept_within_its_range() {
    let mut s = Session::new();
    assert_eq!(s.prefs.assistant_max_tokens, crate::ASSISTANT_TOKENS_DEFAULT);
    let r = s.execute("pref.assistant", &json!({"maxTokens": 128000})).unwrap();
    assert_eq!(r["maxTokens"], 128000);
    assert_eq!(s.prefs.assistant_max_tokens, 128000);
    // The whole range is taken, and nothing outside it changes anything.
    s.execute("pref.assistant", &json!({"maxTokens": crate::ASSISTANT_TOKENS_MAX})).unwrap();
    assert_eq!(s.prefs.assistant_max_tokens, crate::ASSISTANT_TOKENS_MAX);
    for bad in
        [json!({"maxTokens": 0}), json!({"maxTokens": 15}), json!({"maxTokens": 1_000_001}), json!({"maxTokens": 12.5}), json!({"maxTokens": "many"})]
    {
        assert!(s.execute("pref.assistant", &bad).is_err(), "{bad}");
        assert_eq!(s.prefs.assistant_max_tokens, crate::ASSISTANT_TOKENS_MAX, "a refused budget changes nothing");
    }
    let prefs = Prefs { assistant_max_tokens: 65536, ..Default::default() };
    let back: Prefs = serde_json::from_str(&serde_json::to_string(&prefs).unwrap()).unwrap();
    assert_eq!(back.assistant_max_tokens, 65536);
}

#[test]
fn the_agent_mode_is_kept_and_old_settings_default_to_agent() {
    let mut s = Session::new();
    let r = s.execute("pref.assistant", &json!({"agent": "Ask"})).unwrap();
    assert_eq!(r["agent"], "ask", "the id is stored in its canonical form");
    assert_eq!(s.prefs.assistant_agent, "ask");
    assert_eq!(s.execute("pref.assistant", &json!({"agent": "plan"})).unwrap()["agent"], "plan");
    let e = s.execute("pref.assistant", &json!({"agent": "everything"})).unwrap_err().to_string();
    assert!(e.contains("agent, ask, plan"), "{e}");
    assert_eq!(s.prefs.assistant_agent, "plan", "a refused mode changes nothing");
    // No mode saved reads as the default in the result, and an old file still opens.
    let mut fresh = Session::new();
    assert_eq!(fresh.execute("pref.assistant", &json!({})).unwrap()["agent"], "agent");
    let old: Prefs = serde_json::from_str("{}").unwrap();
    assert!(old.assistant_agent.is_empty(), "an old file has no mode: the app picks Agent");
    let plain = serde_json::to_value(Prefs::default()).unwrap();
    assert!(plain.get("assistant_agent").is_none(), "an unset mode stays out of the file: {plain}");
}

#[test]
fn a_bad_endpoint_or_model_is_refused() {
    let mut s = Session::new();
    for bad in [
        json!({"endpoint": "ftp://example"}),
        json!({"endpoint": "api.deepseek.com"}),
        json!({"endpoint": "https://a b.example"}),
        json!({"model": "two words"}),
        json!({"model": ""}),
        json!({"model": "x".repeat(101)}),
    ] {
        assert!(s.execute("pref.assistant", &bad).is_err(), "{bad}");
    }
    assert!(s.prefs.assistant_endpoint.is_empty() && s.prefs.assistant_model.is_empty(), "a refused setting changes nothing");
}

#[test]
fn the_settings_survive_the_settings_file_and_old_files_keep_their_defaults() {
    let prefs = Prefs {
        assistant_endpoint: "https://proxy.example/v1".into(),
        assistant_model: "deepseek-v4-pro".into(),
        assistant_thinking: false,
        assistant_key: "sk-secret".into(),
        assistant_remember: true,
        ..Default::default()
    };
    let text = serde_json::to_string(&prefs).unwrap();
    let back: Prefs = serde_json::from_str(&text).unwrap();
    assert_eq!(back.assistant_endpoint, prefs.assistant_endpoint);
    assert_eq!(back.assistant_model, prefs.assistant_model);
    assert!(!back.assistant_thinking);
    assert_eq!(back.assistant_key, "sk-secret");
    assert!(back.assistant_remember, "the Remember answer survives the file");
    // Unset settings are left out of the file, and a file without them reads the defaults.
    let plain = serde_json::to_value(Prefs::default()).unwrap();
    assert!(plain.get("assistant_key").is_none() && plain.get("assistant_endpoint").is_none() && plain.get("assistant_model").is_none(), "{plain}");
    let old: Prefs = serde_json::from_str("{}").unwrap();
    assert!(old.assistant_key.is_empty() && old.assistant_endpoint.is_empty() && old.assistant_model.is_empty() && old.assistant_thinking);
    assert!(!old.assistant_remember, "an old file starts with the box empty");
}
