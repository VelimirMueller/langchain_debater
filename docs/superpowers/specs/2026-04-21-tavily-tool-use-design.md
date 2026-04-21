# Tavily Tool Use for Debaters — Design

**Date:** 2026-04-21
**Status:** Approved (brainstorming phase complete)
**Scope:** Approach 1 (narrow) — proposer and critic get search/extract tools. Moderator and judge stay pure-LLM.

---

## Goal

Add ReAct-style tool-calling to the debate agent so the proposer and critic can ground their arguments in current web evidence. The agent shifts from "closed-loop LLM + state" (explicitly a non-goal in the README) to a tool-using agent — the single biggest architectural leap available on this learning trajectory.

Success is measured by observability: a research-hungry topic should show many tool spans in LangSmith and Langfuse; a pure-principles topic should show few or none. The LLM learning *when not to search* is as important as searching well.

## Non-goals

- Tools for the moderator or judge (Approach 2 territory; may follow later).
- Cross-turn source deduplication or state-level tracking of fetched URLs.
- Caching Tavily responses across runs.
- Custom tools beyond Tavily's `tavily_search` and `tavily_extract`.
- Retry / repair logic beyond Tavily's built-in one-shot retry.
- Streaming output, UI, persistence — unchanged from existing README non-goals.

---

## Design decisions and their alternatives

Each decision below was made during brainstorming. Alternatives are recorded so future-you knows what was considered and why rejected.

### 1. Who uses tools: proposer + critic only
**Alternatives considered:** (a) researcher node running once before debate; (b) per-round researcher node; (c) also judge (fact-checks claims).
**Chosen:** per-role tool calling on proposer + critic. **Why:** teaches the ReAct pattern (the actual learning goal) without muddying adjudicator roles. Judge-with-tools is a natural follow-up.

### 2. Tool capability: separate `search` and `fetch`
**Alternatives considered:** (a) search only; (b) scrape only; (d) combined `search_and_summarize`.
**Chosen:** two tools, `tavily_search` and `tavily_extract`. **Why:** most realistic agent pattern; forces the LLM to make a decision (snippet sufficient? or do I need the full article?) that shows up cleanly in traces.

### 3. Provider: Tavily only
**Alternatives considered:** Firecrawl only; Tavily search + Firecrawl fetch; DuckDuckGo + Trafilatura (zero keys).
**Chosen:** Tavily only. **Why:** one vendor, one key, LangChain-canonical integration (`langchain-tavily`). Maximum transferable learning — you'll see this exact combo in most modern agent tutorials.

### 4. Loop implementation: hand-rolled ReAct loop
**Alternatives considered:** `create_react_agent` (prebuilt); sub-graph per role.
**Chosen:** hand-rolled loop inside a shared `_run_with_tools()` helper. **Why:** the loop is the actual learning target. ~10 lines of code; any abstraction over it defeats the purpose. Helper keeps the target factored rather than duplicated.

### 5. State shape: unchanged
**Alternatives considered:** new `sources_seen` field tracking URLs; per-turn `sources` list on `Turn`.
**Chosen:** no state changes. Sources live inline in argument markdown as `[claim](https://url)`. **Why:** YAGNI. If cross-turn source awareness becomes interesting, add it in Approach 2.

---

## Architecture

The graph topology is unchanged. `proposer` and `critic` remain single nodes. What changes is what happens inside those nodes: each becomes a mini-ReAct agent with its own tool-call loop.

```
START ─▶ setup ─▶ moderator_open ─▶ proposer ─▶ critic ─▶ moderator_distill ─▶ judge ─▶ (loop or END)
                                     │            │
                                     │            └── ReAct loop:
                                     │                LLM → tool_call(s) → ToolMessage → LLM → ... → argument
                                     └── ReAct loop (same shape)
```

### File changes

| File | Change | Responsibility |
|---|---|---|
| `debate/tools.py` | **new** | Tavily config; returns `[search, extract]` as LangChain `Tool` instances. Single responsibility: isolate tool config from orchestration. |
| `debate/nodes.py` | modified | Add `_run_with_tools()` helper + `MAX_TOOL_CALLS`. Rewrite `proposer_node` and `critic_node` to call it. Factor shared prompt-building into `_build_debater_prompt()`. |
| `debate/prompts.py` | modified | Append "Research strategy" block to `PROPOSER_SYSTEM` and `CRITIC_SYSTEM`. |
| `main.py` | modified | TAVILY_API_KEY presence check. Bump `prompts_version` to `"v2"`. Add `max_tool_calls` and `search_provider` metadata. Add `"tools:v1"` tag. |
| `requirements.txt` | modified | `langchain-tavily>=0.1` |
| `.env.example` | modified | `TAVILY_API_KEY=` line with signup URL comment |
| `debate/state.py` | unchanged | No new state fields. |
| `debate/graph.py` | unchanged | Topology preserved. |

### `debate/tools.py` (new module)

```python
"""Tavily search + extract, wrapped as LangChain tools for debater use.

Exposed as LangChain Tool instances so they trace cleanly in LangSmith
and Langfuse under their canonical names (tavily_search, tavily_extract).
"""
from langchain_tavily import TavilyExtract, TavilySearch

def get_research_tools():
    """Return [search, extract] with debate-appropriate defaults."""
    search = TavilySearch(
        max_results=4,
        search_depth="basic",      # "advanced" ≈ 2x cost, rarely worth it
        include_answer=False,      # we want raw sources, not Tavily's synthesis
    )
    extract = TavilyExtract(extract_depth="basic")
    return [search, extract]
```

### `debate/nodes.py` changes

Add constant and imports:

```python
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)

MAX_TOOL_CALLS = 4
```

Add the loop helper. The marked block is the user's contribution — ~10 lines of actual ReAct loop:

```python
def _run_with_tools(
    role: str,
    system_prompt: str,
    user_prompt: str,
    tools: list,
    config: RunnableConfig,
    run_name_prefix: str,
) -> str:
    """Run a role with a ReAct tool-call loop. Returns final argument text.

    Protocol: LLM may emit tool_calls; we execute them and feed results back
    as ToolMessages until it produces prose (no tool_calls). Hard-capped at
    MAX_TOOL_CALLS iterations as defence-in-depth.
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    llm = _model_for(role).bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    # ===== USER-WRITTEN LOOP (~10 lines) =====
    # For i in range(MAX_TOOL_CALLS):
    #   1. response = llm.invoke(messages, config={**config, "run_name": f"{run_name_prefix}-iter{i}"})
    #   2. messages.append(response)  # CRITICAL: Claude needs to see its own tool_calls in history
    #   3. if not response.tool_calls: return response.content
    #   4. for tc in response.tool_calls:
    #        try: result = tool_by_name[tc["name"]].invoke(tc["args"])
    #        except Exception as e: result = f"Tool error: {e}"
    #        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    # =========================================

    # Budget exhausted without prose: force a tool-free close.
    forced = _model_for(role).invoke(
        messages + [HumanMessage(content=(
            "You've used your research budget. Produce your final argument now, "
            "using only the evidence gathered above."
        ))],
        config={**config, "run_name": f"{run_name_prefix}-forced-close"},
    )
    return forced.content
```

Refactored debater nodes (symmetric for critic):

```python
def proposer_node(state: DebateState, config: RunnableConfig) -> dict:
    from debate.prompts import PROPOSER_SYSTEM
    from debate.tools import get_research_tools

    round_n = state["round_num"] + 1
    user_prompt = _build_debater_prompt(state, round_n, side="for")
    content = _run_with_tools(
        role="proposer",
        system_prompt=PROPOSER_SYSTEM,
        user_prompt=user_prompt,
        tools=get_research_tools(),
        config=config,
        run_name_prefix=f"proposer-argument-r{round_n}",
    )
    return {"transcript": [Turn(role="proposer", content=content, round_num=round_n)]}
```

`_build_debater_prompt(state, round_n, side)` extracts the prompt-body construction currently duplicated between `proposer_node` and `critic_node`. Factor now while we're in the file.

### Prompt additions

Appended to both `PROPOSER_SYSTEM` and `CRITIC_SYSTEM`:

```
You have access to two research tools:
- tavily_search: find 3-5 relevant sources for a specific query. Use this first;
  the returned snippets are often enough for citation.
- tavily_extract: fetch a specific URL's full text when a snippet is too thin
  for a proper quotation.

Research strategy:
- Open your turn with one well-crafted tavily_search query to ground the argument
  in current evidence. One focused query beats five scattered ones.
- Only tavily_extract when the snippet is genuinely insufficient.
- You have a hard budget of 4 tool calls per turn. Using fewer is often correct.
- Cite inline as markdown: [claim](https://url). Do not invent URLs.
- If search returns nothing useful, argue from principle — do not apologize for
  the absence of sources or hedge your claim. A strong rhetorical turn that
  acknowledges the limit of available evidence is better than a weak apology.
```

The closing sentence is load-bearing: without it, Claude tends to weaken its own argument when search fails.

### Metadata and tracing

`main.py` metadata:

```python
metadata = {
    "topic": topic,
    "max_rounds": str(max_rounds),
    "max_tool_calls": str(MAX_TOOL_CALLS),   # NEW
    "model": "claude-sonnet-4-6",
    "prompts_version": "v2",                  # bumped from v1
    "search_provider": "tavily",              # NEW
}
tags = ["debate", "learning", "experiment:v1", "tools:v1"]  # tools:v1 added
```

New spans (auto-created, no code):
- `tavily_search` — input: `{query: ...}`, output: result JSON
- `tavily_extract` — input: `{url: ...}`, output: full markdown

New run names (for trace filtering):
- `proposer-argument-r1-iter0`, `-iter1`, ... — the iterations of the ReAct loop
- `proposer-argument-r1-forced-close` — appears only when budget exhausted (signal of a pathological run)

---

## Guardrails

| Constant | Value | Rationale |
|---|---|---|
| `MAX_TOOL_CALLS` | 4 | Covers `search → maybe extract → maybe re-search → argue`. 5+ loops unproductively. |
| `MAX_TOKENS_PER_TURN` | 500 (unchanged) | Output ceiling; tool calls don't affect this. |
| Tavily `max_results` | 4 | Keeps each search response ≈1.5k input tokens. |

**Worst-case cost per debate:** 2 debaters × 4 tool calls × 3 rounds = 24 Tavily calls + 24 extra LLM invocations. Input tokens climb from ~4k to ~40-60k per research-hungry debate. Estimated $0.15–0.30 on Sonnet 4.6 — acceptable for learning.

## Error handling

Philosophy: surface failures in traces, never swallow silently. You want to *see* a Tavily 429 in Langfuse the day it happens.

- **Tavily API error inside tool call:** caught in `_run_with_tools` loop; appended as `ToolMessage(content=f"Tool error: {e}")`. Prompt instructs the LLM to recover by arguing from principle.
- **Rate limit (429):** treated like any other Tavily error — caught, surfaced as a `ToolMessage`, LLM recovers via the prompted fallback. Any built-in retry behaviour in `langchain-tavily` is a bonus, not assumed.
- **Missing TAVILY_API_KEY:** `main.py` startup check prints signup URL, exits code 2. Same pattern as existing Anthropic/LangSmith/Langfuse key checks.
- **Malformed tool call from LLM:** same as API error path — error message fed back as ToolMessage, LLM retries.
- No global try/except swallowing — failures bubble up visibly.

## Validation plan

Manual, via traces — matches the project's philosophy.

1. **Research-hungry topic:** `"Should EU AI Act general-purpose AI rules be delayed past August 2026?"` — expect heavy tool use (3-4 searches, maybe 1-2 extracts, inline citations in transcript).
2. **Pure-principles topic:** `"Is lying ever morally required?"` — expect minimal or zero tool use. This verifies the "when *not* to search" learning.
3. **Rate-limit/error handling:** repeat research-hungry topic 5× rapidly. Expect either graceful degradation or visible 429 spans.
4. **v1 vs v2 A/B:** same topic, filter LangSmith/Langfuse by `prompts_version: v1` vs `v2`. Inspect argument quality, length, citation density, cost.

## Open questions

None blocking. The following are deliberate deferrals, not unknowns:

- Whether to extend to judge (Approach 2) — decide after running Approach 1 and inspecting traces.
- Whether to add cross-turn source dedup — same trigger.
- Whether Tavily's `include_answer=True` would improve or hurt debate quality — worth an experiment post-launch.

---

## Summary

- Graph topology unchanged; debater nodes internally become mini-ReAct agents.
- One new module (`debate/tools.py`), 3 modified files (`nodes.py`, `prompts.py`, `main.py`), small config changes (`requirements.txt`, `.env.example`).
- ~10 lines of user-written ReAct loop is the core learning artefact.
- Observability is the payoff: every tool call, every reasoning iteration visible in LangSmith and Langfuse.
- YAGNI-bounded: state unchanged, moderator/judge untouched, no caching or dedup.
