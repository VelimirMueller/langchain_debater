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


def moderator_node(state: DebateState, config: RunnableConfig) -> dict:
    """Moderator: opens the debate on first run, distils rounds afterward.

    Registered at two graph positions (moderator_open, moderator_distill).
    Branches internally based on whether the transcript is empty.
    """
    from debate.prompts import MODERATOR_DISTILL_SYSTEM, MODERATOR_OPEN_SYSTEM

    round_n = state["round_num"]
    is_opening = len(state["transcript"]) == 0

    # 👤 TODO(you): Write the branching logic.
    # You need to set two variables based on is_opening:
    #   - system_prompt: which of the two moderator prompts to use
    #   - user_prompt:   the concrete ask, built from state
    #
    # For the OPENING branch (is_opening == True):
    #   Build a prompt that gives the model the topic and asks it to frame
    #   the initial focus question. The transcript is empty, so don't reference it.
    #
    # For the DISTILLATION branch (is_opening == False):
    #   Include the transcript (or at least the latest proposer + critic turns)
    #   and ask the model to distil and pose the next focus question.
    #
    # The user_prompt string you build will go into a HumanMessage below.
    # Keep it under ~15 lines total.

    system_prompt: str  # set this
    user_prompt: str    # set this

    # END USER TODO ------------------------------------------------------------

    llm_config = {**config, "run_name": f"moderator-{'open' if is_opening else 'distill'}-r{round_n}"}
    response = _model_for("moderator").invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        config=llm_config,
    )
    new_focus = response.content.strip()

    return {
        "focus_question": new_focus,
        "transcript": [Turn(role="moderator", content=new_focus, round_num=round_n)],
    }
