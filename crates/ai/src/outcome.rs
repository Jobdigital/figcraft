//! What one tool call produced, and how it reads back to the model.

/// A tool call's outcome. `text` is what the model reads; an image block (a screenshot) is
/// described rather than carried, because the chat models read text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolOutcome {
    pub text: String,
    pub is_error: bool,
}

impl ToolOutcome {
    pub fn text(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: false }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: true }
    }

    /// The text to send back to the model: failures are prefixed, and long output is cut at a
    /// character boundary with a note saying how much was left out.
    pub fn for_model(&self, max_chars: usize) -> String {
        let max = max_chars.max(1);
        let mut text =
            if self.is_error && !self.text.trim_start().starts_with("Error") { format!("Error: {}", self.text) } else { self.text.clone() };
        let total = text.chars().count();
        if total > max {
            let mut cut = 0;
            for (n, (i, _)) in text.char_indices().enumerate() {
                if n == max {
                    cut = i;
                    break;
                }
            }
            if cut > 0 {
                text.truncate(cut);
                text.push_str(&format!("\n… (truncated {} characters)", total - max));
            }
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_output_is_cut_on_a_character_boundary() {
        let outcome = ToolOutcome::text("á".repeat(100));
        let text = outcome.for_model(10);
        assert!(text.starts_with(&"á".repeat(10)), "{text}");
        assert!(text.contains("truncated 90 characters"), "{text}");
    }

    #[test]
    fn a_failure_is_prefixed_once() {
        assert_eq!(ToolOutcome::error("no such node 7").for_model(100), "Error: no such node 7");
        assert_eq!(ToolOutcome::error("Error: already").for_model(100), "Error: already");
    }
}
