"""Regression tests for the two edge paths in debate/nodes.py.

No network: the judge runs against a stub model, and the forced close is
checked on the request payload langchain-anthropic would send.
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import debate.nodes as nodes


@pytest.fixture
def stub_judge(monkeypatch):
    """Make every model call return `reply`; record the last prompt sent."""
    seen = {}

    def install(reply: str):
        class Stub:
            def invoke(self, messages, config=None):
                seen["prompt"] = messages[-1].content
                return SimpleNamespace(content=reply)

        monkeypatch.setattr(nodes, "_model_for", lambda role: Stub())
        return seen

    return install


def _state(round_num: int) -> dict:
    return {"topic": "t", "transcript": [], "round_num": round_num, "max_rounds": 3}


@pytest.mark.parametrize(
    ("round_num", "reply", "verdict", "route"),
    [
        (0, "CONTINUE: more ground", None, "continue"),
        (0, "VERDICT: critic wins", "critic wins", "end"),
        (0, "Verdict: draw", "draw", "end"),
        (2, "VERDICT: proposer wins", "proposer wins", "end"),
        (2, "CONTINUE: still open", "", "end"),
    ],
)
def test_judge_paths(stub_judge, round_num, reply, verdict, route):
    stub_judge(reply)
    out = nodes.judge_node(_state(round_num), {})
    assert out["verdict"] == verdict
    assert nodes.judge_router(out) == route


def test_judge_final_round_asks_for_verdict_only(stub_judge):
    seen = stub_judge("VERDICT: x")
    nodes.judge_node(_state(2), {})
    assert "must rule now" in seen["prompt"]
    assert "CONTINUE" not in seen["prompt"]


def test_forced_close_keeps_tools_bound(monkeypatch):
    """Anthropic rejects tool_use history unless tools are defined."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    @tool
    def tavily_search(query: str) -> str:
        """Search the web."""
        return ""

    calls = iter(range(10))

    class LoopStub:
        def invoke(self, messages, config=None):
            i = next(calls)
            return AIMessage(
                content="",
                tool_calls=[{"name": "tavily_search", "args": {"query": "q"}, "id": f"toolu_{i}"}],
            )

    captured = {}
    real_model_for = nodes._model_for

    def model_for(role):
        model = real_model_for(role)

        class Bindable:
            def bind_tools(self, tools, **kwargs):
                if kwargs.get("tool_choice") == {"type": "none"}:
                    bound = model.bind_tools(tools, **kwargs)

                    class Closer:
                        def invoke(self, messages, config=None):
                            captured["payload"] = model._get_request_payload(messages, **bound.kwargs)
                            return SimpleNamespace(content="final argument")

                    return Closer()
                return LoopStub()

        return Bindable()

    monkeypatch.setattr(nodes, "_model_for", model_for)
    out = nodes._run_with_tools("proposer", "sys", "user", [tavily_search], {}, "p")

    assert out == "final argument"
    payload = captured["payload"]
    assert payload["tool_choice"] == {"type": "none"}
    assert [t["name"] for t in payload["tools"]] == ["tavily_search"]
