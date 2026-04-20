"""System prompts for each debate role.

Each constant below defines the persona and output expectations for one role.
The scaffolder leaves these as minimal placeholders; the author writes the real
voice. Iterate on prompts here — no other file needs to change when tuning voice.
"""

# TODO(you): Write the moderator's OPENING prompt.
# Called once at the start of the debate, before any transcript exists.
# Goal: frame a sharp, specific focus question for round 1 that forces the
# proposer and critic to engage substantively.
# Tradeoffs:
#   - Too vague ("what are the merits?") → shallow debate
#   - Too leading ("explain why remote work is bad") → biased, unfair
#   - Too long → distracts proposer from the actual topic
# Aim for ~2-3 sentences.
MODERATOR_OPEN_SYSTEM = """TODO: write me."""

# TODO(you): Write the moderator's DISTILLATION prompt.
# Called after each critic turn. Receives the full transcript and must:
#   1. Briefly summarise where the two sides agree and disagree.
#   2. Emit a sharpened follow-up question for the next round.
# The output becomes the next round's focus_question, so phrasing matters.
# Keep it concise — this is a bridge, not a new debate turn.
MODERATOR_DISTILL_SYSTEM = """TODO: write me."""

# TODO(you): Write the proposer's prompt.
# Argues FOR the proposition. Must address the current focus_question and
# engage with the critic's previous rebuttal (if any).
# Style decisions:
#   - Rigorous vs rhetorical?
#   - First-principles vs evidence-citing?
#   - Conceding weak points vs refusing to yield?
# You choose.
PROPOSER_SYSTEM = """TODO: write me."""

# TODO(you): Write the critic's prompt.
# Argues AGAINST the proposition. Should attack the strongest form of the
# proposer's argument (steelmanning), not strawman it.
# Typically symmetric in style to the proposer prompt, but with opposite stance.
CRITIC_SYSTEM = """TODO: write me."""

# TODO(you): Write the judge's prompt.
# Decides whether the debate should continue or end.
# Return a short response that either:
#   - Starts with "CONTINUE:" followed by a reason → debate loops
#   - Starts with "VERDICT:" followed by the final ruling → debate ends
# The judge_node parses this prefix to decide termination.
# Consider: when SHOULD a debate end? Consensus? Convergence? Clear winner?
# One side conceding? Just max rounds? Your choice shapes the agent's feel.
JUDGE_SYSTEM = """TODO: write me."""
