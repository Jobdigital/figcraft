//! How the Assistant answers: doing the work, only looking, or planning it. The panel's agent
//! picker (and `figcraft-cli agent --mode`) chooses one; the mode decides the rules the model is
//! given and whether it may call a tool that changes anything.

/// What the Assistant may do with one prompt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Do the work: the design tools, `run_command` and `batch`.
    #[default]
    Agent,
    /// Answer questions and inspect, changing nothing: a tool that edits is not even offered.
    Ask,
    /// Write a numbered plan naming the commands that would do the work, changing nothing.
    Plan,
}

impl Mode {
    /// Every mode, in picker order.
    pub const ALL: &'static [Mode] = &[Mode::Agent, Mode::Ask, Mode::Plan];

    /// The id the preference, the command line and `ui.inspect` use.
    pub fn id(self) -> &'static str {
        match self {
            Mode::Agent => "agent",
            Mode::Ask => "ask",
            Mode::Plan => "plan",
        }
    }

    /// The name the picker shows: a translation key (English).
    pub fn label(self) -> &'static str {
        match self {
            Mode::Agent => "Agent",
            Mode::Ask => "Ask",
            Mode::Plan => "Plan",
        }
    }

    /// The one-line explanation the picker shows under the name.
    pub fn hint(self) -> &'static str {
        match self {
            Mode::Agent => "Designs and changes the file",
            Mode::Ask => "Answers questions, changing nothing",
            Mode::Plan => "Plans the work, changing nothing",
        }
    }

    /// The mode for an id, in any case; `None` for a name this build doesn't know.
    pub fn from_id(id: &str) -> Option<Self> {
        Mode::ALL.iter().copied().find(|m| m.id().eq_ignore_ascii_case(id.trim()))
    }

    /// Whether its tools may change the file.
    pub fn edits(self) -> bool {
        matches!(self, Mode::Agent)
    }

    /// What the model is told about this mode, as lines for the system prompt.
    pub fn rules(self) -> &'static str {
        match self {
            Mode::Agent => "\n- Agent mode: do the work with the tools and check the result; the final answer says what changed.",
            Mode::Ask => {
                "\n- Ask mode: you may only look. Never change the file and never claim you did; no tool that edits is \
                 offered to you. Answer with what you find, naming layers by id. If the user wants a change, say to \
                 switch to Agent mode."
            }
            Mode::Plan => {
                "\n- Plan mode: you may only look. Change nothing (no tool that edits is offered to you) and answer \
                 with a short numbered plan: for each step the exact command to run with its params, and what to \
                 check after it."
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_id_round_trips_in_any_case_and_unknown_names_are_refused() {
        for mode in Mode::ALL {
            assert_eq!(Mode::from_id(mode.id()), Some(*mode));
            assert_eq!(Mode::from_id(&mode.id().to_uppercase()), Some(*mode));
            assert_eq!(Mode::from_id(&format!(" {} ", mode.id())), Some(*mode));
        }
        assert_eq!(Mode::from_id("everything"), None);
        assert_eq!(Mode::from_id(""), None);
    }

    #[test]
    fn only_agent_may_edit_and_each_mode_says_so() {
        assert!(Mode::Agent.edits());
        assert!(!Mode::Ask.edits() && !Mode::Plan.edits());
        assert!(Mode::Ask.rules().contains("Never change"));
        assert!(Mode::Plan.rules().contains("numbered plan"));
        assert!(Mode::Agent.rules().contains("do the work"));
        // The ids are stable: they are stored in the settings file and published by ui.inspect.
        assert_eq!(Mode::ALL.iter().map(|m| m.id()).collect::<Vec<_>>(), ["agent", "ask", "plan"]);
    }
}
