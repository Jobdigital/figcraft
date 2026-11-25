//! Where the DeepSeek API is reached, which model answers, and the key that authorises it.

use std::time::Duration;

/// The DeepSeek API's origin: an OpenAI-compatible `/chat/completions` endpoint.
pub const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

/// The model the Assistant starts with.
pub const DEFAULT_MODEL: &str = "deepseek-flash";

/// The chat models the Assistant offers (DeepSeek's line-up, November 2026).
pub const MODELS: &[&str] = &["deepseek-flash", "deepseek-v4-pro"];

/// The environment variable the API key is read from.
pub const API_KEY_ENV: &str = "DEEPSEEK_API_KEY";

/// An environment variable naming another OpenAI-compatible endpoint (a local server, a proxy).
pub const BASE_URL_ENV: &str = "DEEPSEEK_BASE_URL";

/// How to reach the model. The key is a secret: it is never logged, shown in errors, or kept in
/// preferences (the desktop app reads [`API_KEY_ENV`] or a key typed for the session).
#[derive(Clone, PartialEq, Eq)]
pub struct DeepSeekConfig {
    pub api_key: String,
    pub base_url: String,
    /// The whole exchange, response body included; a long thinking answer takes a while.
    pub timeout: Duration,
    /// Retries after a connection failure, a 429 or a 5xx before any text was streamed.
    pub max_retries: u32,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self { api_key: String::new(), base_url: DEFAULT_BASE_URL.to_string(), timeout: Duration::from_secs(300), max_retries: 2 }
    }
}

impl DeepSeekConfig {
    /// The key and endpoint from the environment, or the defaults.
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Ok(key) = std::env::var(API_KEY_ENV) {
            config.api_key = key.trim().to_string();
        }
        if let Ok(url) = std::env::var(BASE_URL_ENV) {
            config.base_url = url.trim().to_string();
        }
        config
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = key.into().trim().to_string();
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim().to_string();
        self
    }

    pub fn with_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// The chat completions endpoint: a base with `/chat/completions` already is used as is.
    pub fn endpoint(&self) -> String {
        let base = self.base_url.trim().trim_end_matches('/');
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else if base.is_empty() {
            format!("{DEFAULT_BASE_URL}/chat/completions")
        } else {
            format!("{base}/chat/completions")
        }
    }
}

/// Whether a key is available: the one typed for the session, else the environment's. Whitespace
/// alone is no key.
pub fn key_available(typed: &str, from_env: Option<&str>) -> bool {
    let set = |k: &str| !k.trim().is_empty();
    set(typed) || from_env.is_some_and(set)
}

/// The key is printed as `<set>` or `<unset>` so a log line never holds it.
impl std::fmt::Debug for DeepSeekConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekConfig")
            .field("api_key", &if self.api_key.is_empty() { "<unset>" } else { "<set>" })
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_endpoint_is_built_once() {
        for (base, want) in [
            ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
            ("https://api.deepseek.com/", "https://api.deepseek.com/chat/completions"),
            ("http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions"),
            ("https://proxy.example/chat/completions/", "https://proxy.example/chat/completions"),
        ] {
            let config = DeepSeekConfig::default().with_base_url(base);
            assert_eq!(config.endpoint(), want);
        }
    }

    #[test]
    fn the_debug_form_never_prints_the_key() {
        let config = DeepSeekConfig::default().with_key("sk-secret");
        let shown = format!("{config:?}");
        assert!(!shown.contains("sk-secret"), "{shown}");
        assert!(shown.contains("<set>"));
    }

    #[test]
    fn a_key_is_the_typed_one_or_the_environments_and_blanks_dont_count() {
        assert!(key_available("sk-typed", None));
        assert!(key_available("", Some("sk-env")));
        assert!(key_available("  ", Some("sk-env")));
        assert!(!key_available("", None));
        assert!(!key_available("   ", Some("  ")));
    }
}
