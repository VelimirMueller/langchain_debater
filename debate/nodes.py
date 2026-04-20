"""Node functions for the debate graph.

Each node reads DebateState (+ RunnableConfig for tracing) and returns a state
delta dict. The model factory centralises per-role temperature choices.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from debate.state import DebateState, Turn

MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS_PER_TURN = 500

# Per-role temperature: argumentative roles get more creativity,
# adjudicating roles get more consistency.
_ROLE_TEMPERATURES: dict[str, float] = {
    "proposer": 0.7,
    "critic": 0.7,
    "moderator": 0.3,
    "judge": 0.3,
}


def _model_for(role: str) -> ChatAnthropic:
    """Construct a ChatAnthropic client with role-appropriate temperature."""
    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=_ROLE_TEMPERATURES[role],
        max_tokens=MAX_TOKENS_PER_TURN,
    )


def setup_node(state: DebateState) -> dict:
    """Initialise runtime state. No LLM call; runs once at graph entry."""
    return {
        "transcript": [],
        "focus_question": "",
        "round_num": 0,
        "verdict": None,
    }
