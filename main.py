"""Entry point for the debate agent.

IMPORTANT: load_dotenv() must run BEFORE any `from langchain...` or
`from langgraph...` or `from debate...` import. LangSmith's auto-attach
reads env vars at import time; wrong order silently disables tracing.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # do not move below the langchain imports

from langfuse.langchain import CallbackHandler as LangfuseHandler  # noqa: E402

from debate.graph import build_graph  # noqa: E402
from debate.nodes import MAX_TOOL_CALLS  # noqa: E402
from debate.state import DebateState  # noqa: E402


def build_run_config(topic: str, max_rounds: int) -> dict:
    """Build the RunnableConfig passed to graph.invoke().

    This is where run_name, tags, and metadata are set — the three axes you'll
    filter traces on in LangSmith and Langfuse. Getting these right makes your
    traces searchable.
    """
    langfuse_handler = LangfuseHandler()

    run_name = f"debate:{topic[:40]}"
    tags = ["debate", "learning", "experiment:v1", "tools:v1"]
    # Langfuse v4 requires metadata values to be strings; coerce.
    metadata = {
        "topic": topic,
        "max_rounds": str(max_rounds),
        "max_tool_calls": str(MAX_TOOL_CALLS),
        "model": "claude-sonnet-4-6",
        "prompts_version": "v2",
        "search_provider": "tavily",
    }

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

    if not os.getenv("TAVILY_API_KEY"):
        print(
            "TAVILY_API_KEY not set. Sign up at https://tavily.com and add "
            "the key to your .env file.",
            file=sys.stderr,
        )
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
