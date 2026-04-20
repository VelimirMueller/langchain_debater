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


def _format_transcript(transcript: list[Turn]) -> str:
    """Render transcript as a readable dialog for inclusion in prompts."""
    lines = []
    for turn in transcript:
        lines.append(f"[{turn['role']}, round {turn['round_num']}]")
        lines.append(turn["content"])
        lines.append("")
    return "\n".join(lines)


def proposer_node(state: DebateState, config: RunnableConfig) -> dict:
    """Argues FOR the proposition."""
    from debate.prompts import PROPOSER_SYSTEM

    round_n = state["round_num"] + 1
    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Current focus question: {state['focus_question']}\n\n"
        f"Debate so far:\n{_format_transcript(state['transcript'])}\n\n"
        f"Make your case for the proposition. Round {round_n}."
    )

    llm_config = {**config, "run_name": f"proposer-argument-r{round_n}"}
    response = _model_for("proposer").invoke(
        [SystemMessage(content=PROPOSER_SYSTEM), HumanMessage(content=user_prompt)],
        config=llm_config,
    )

    return {
        "transcript": [Turn(role="proposer", content=response.content, round_num=round_n)]
    }


def critic_node(state: DebateState, config: RunnableConfig) -> dict:
    """Argues AGAINST the proposition, engaging with proposer's latest turn."""
    from debate.prompts import CRITIC_SYSTEM

    round_n = state["round_num"] + 1
    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Current focus question: {state['focus_question']}\n\n"
        f"Debate so far:\n{_format_transcript(state['transcript'])}\n\n"
        f"Rebut the proposer's latest argument. Round {round_n}."
    )

    llm_config = {**config, "run_name": f"critic-rebuttal-r{round_n}"}
    response = _model_for("critic").invoke(
        [SystemMessage(content=CRITIC_SYSTEM), HumanMessage(content=user_prompt)],
        config=llm_config,
    )

    return {
        "transcript": [Turn(role="critic", content=response.content, round_num=round_n)]
    }


def judge_node(state: DebateState, config: RunnableConfig) -> dict:
    """Decide whether to continue the debate or end it with a verdict."""
    from debate.prompts import JUDGE_SYSTEM

    round_n = state["round_num"] + 1
    at_cap = round_n >= state["max_rounds"]

    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Debate so far:\n{_format_transcript(state['transcript'])}\n\n"
        f"This was round {round_n} of at most {state['max_rounds']}.\n\n"
        "Decide: has the debate reached a natural end, or should it continue?\n"
        "Respond with either 'CONTINUE: <reason>' or 'VERDICT: <your ruling>'."
    )

    llm_config = {**config, "run_name": f"judge-decision-r{round_n}"}
    response = _model_for("judge").invoke(
        [SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user_prompt)],
        config=llm_config,
    )
    judge_text = response.content.strip()

    # 👤 TODO(you): Decide the termination rule.
    # You have three inputs:
    #   - judge_text: what the judge wrote (starts with "CONTINUE:" or "VERDICT:")
    #   - at_cap: True if round_n >= max_rounds (safety cap reached)
    #   - round_n: the round number just completed
    #
    # Set `verdict` to None if debate should continue, or to a string if it ends.
    # The simplest policy:
    #   - If at_cap → force end, extract the verdict text
    #   - Else, honor the judge's choice: CONTINUE → None, VERDICT → ending string
    # But you can make it stricter (e.g., always run all max_rounds),
    # or looser (e.g., end early on any "VERDICT" even in round 1).
    # Think about: does the debate benefit from early termination on consensus,
    # or does forcing more rounds produce richer arguments?

    verdict: str | None  # set this

    # END USER TODO ------------------------------------------------------------

    return {
        "round_num": round_n,
        "verdict": verdict,
        "transcript": [Turn(role="judge", content=judge_text, round_num=round_n)],
    }


def judge_router(state: DebateState) -> str:
    """Conditional edge function: route to 'continue' or 'end' based on verdict."""
    return "end" if state["verdict"] is not None else "continue"
