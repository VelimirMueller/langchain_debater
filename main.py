"""Entry point for the debate agent.

IMPORTANT: load_dotenv() must run BEFORE any `from langchain...` or
`from langgraph...` or `from debate...` import. LangSmith's auto-attach
reads env vars at import time; wrong order silently disables tracing.
"""

import sys

from dotenv import load_dotenv

load_dotenv()  # do not move below the langchain imports

from langfuse.callback import CallbackHandler as LangfuseHandler  # noqa: E402

from debate.graph import build_graph  # noqa: E402
from debate.state import DebateState  # noqa: E402


def build_run_config(topic: str, max_rounds: int) -> dict:
    """Build the RunnableConfig passed to graph.invoke().

    This is where run_name, tags, and metadata are set — the three axes you'll
    filter traces on in LangSmith and Langfuse. Getting these right makes your
    traces searchable.
    """
    langfuse_handler = LangfuseHandler()

    # 👤 TODO(you): Populate run_name, tags, and metadata.
    # Guidance:
    #   - run_name: a short, human-readable title. Goes in the trace list UI.
    #     Something like f"debate:{topic[:40]}" works; make it yours.
    #   - tags: list of short strings for filtering. At minimum ["debate"].
    #     Consider adding "learning", a topic category, a model identifier, etc.
    #   - metadata: dict of anything you want searchable. Good candidates:
    #     max_rounds, the full topic, model name, timestamp, experiment name.

    run_name: str  # set this
    tags: list[str]  # set this
    metadata: dict  # set this

    # END USER TODO ------------------------------------------------------------

    return {
        "callbacks": [langfuse_handler],
        "run_name": run_name,
        "tags": tags,
        "metadata": metadata,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<debate topic>\"", file=sys.stderr)
        sys.exit(2)

    topic = sys.argv[1]
    max_rounds = 3

    initial_state: DebateState = {
        "topic": topic,
        "transcript": [],
        "focus_question": "",
        "round_num": 0,
        "max_rounds": max_rounds,
        "verdict": None,
    }

    graph = build_graph()
    config = build_run_config(topic=topic, max_rounds=max_rounds)

    final_state = graph.invoke(initial_state, config=config)

    print("=" * 70)
    print(f"TOPIC: {topic}")
    print("=" * 70)
    for turn in final_state["transcript"]:
        print(f"\n[{turn['role'].upper()}, round {turn['round_num']}]")
        print(turn["content"])
    print("\n" + "=" * 70)
    print(f"VERDICT: {final_state['verdict']}")
    print("=" * 70)
    print("\nTraces:")
    print("  LangSmith: https://smith.langchain.com/  (open your project)")
    print("  Langfuse:  https://cloud.langfuse.com/  (open your project)")


if __name__ == "__main__":
    main()
