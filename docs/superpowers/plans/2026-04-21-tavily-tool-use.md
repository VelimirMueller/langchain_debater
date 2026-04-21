# Tavily Tool Use for Debaters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project testing philosophy:** This project has no automated tests by design (README: "No unit tests (manual validation via traces)"). Verification steps below use **run-and-inspect** commands instead of pytest. Each task still has an explicit gate — don't skip it.

**Goal:** Give the `proposer` and `critic` debate nodes ReAct-style tool use against Tavily's search and extract APIs, so each turn can be grounded in current web evidence.

**Architecture:** The LangGraph topology is unchanged. `proposer_node` and `critic_node` remain single nodes in the main graph — but internally each one runs a hand-rolled ReAct loop (`_run_with_tools()`) that can call `tavily_search` / `tavily_extract` up to `MAX_TOOL_CALLS=4` times before producing its final argument. Tool-use is absent from moderator, judge, setup, and state (Approach 1 / narrow).

**Tech Stack:** Python 3.13, LangGraph, LangChain (`langchain-core`, `langchain-anthropic`), `langchain-tavily` (new), Tavily (new vendor), LangSmith + Langfuse (unchanged).

**Spec:** `docs/superpowers/specs/2026-04-21-tavily-tool-use-design.md` (commit `05b6096`).

**⚠️ User-written section:** Task 5 contains ~10 lines the project owner (in learning mode) will write by hand. Agents executing this plan should **pause at Task 5 and hand back** rather than write the loop body themselves.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `debate/tools.py` | NEW | Tavily search/extract wrapped as LangChain `Tool` instances. Isolates tool config from orchestration. |
| `debate/nodes.py` | MODIFIED | Adds `_run_with_tools()` ReAct loop helper, `MAX_TOOL_CALLS` constant, and `_build_debater_prompt()` refactor. Rewrites `proposer_node` / `critic_node` around the helper. |
| `debate/prompts.py` | MODIFIED | Appends "Research strategy" block to `PROPOSER_SYSTEM` and `CRITIC_SYSTEM`. |
| `main.py` | MODIFIED | Adds TAVILY_API_KEY startup check. Bumps `prompts_version` metadata to `"v2"`. Adds `max_tool_calls` + `search_provider` metadata. Adds `"tools:v1"` tag. |
| `requirements.txt` | MODIFIED | Adds `langchain-tavily>=0.1`. |
| `.env.example` | MODIFIED | Adds `TAVILY_API_KEY=` line with signup comment. |
| `debate/state.py` | unchanged | No state shape changes. |
| `debate/graph.py` | unchanged | Topology preserved. |

---

## Task 1: Add Tavily dependency, env scaffolding, startup check

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `main.py`

- [ ] **Step 1: Add langchain-tavily to requirements.txt**

Open `requirements.txt` and append one line (after the existing `langsmith>=0.1`):

```
langchain-tavily>=0.1
```

Final file should read:
```
langgraph>=0.2
langchain>=0.3
langchain-core>=0.3
langchain-anthropic>=0.3
langfuse>=4.0
langsmith>=0.1
langchain-tavily>=0.1
python-dotenv>=1.0
```

- [ ] **Step 2: Install the new dependency**

Run:
```bash
pip install -r requirements.txt
```

Expected: a line like `Successfully installed langchain-tavily-x.y.z tavily-python-x.y.z`. If the install fails (e.g., SSL error), resolve before continuing — nothing else will work without this package.

- [ ] **Step 3: Verify the package imports cleanly**

Run:
```bash
python -c "from langchain_tavily import TavilySearch, TavilyExtract; print('ok')"
```

Expected output: `ok`. If you see `ImportError`, the package name or symbols may have changed in a newer release — check `pip show langchain-tavily` and the package's README on PyPI.

- [ ] **Step 4: Add TAVILY_API_KEY to .env.example**

Open `.env.example`. Add at the bottom:

```
# Tavily — web search + page extract. Sign up at https://tavily.com.
# Free tier: 1,000 credits/month. One credit per basic search or extract.
TAVILY_API_KEY=
```

- [ ] **Step 5: Sign up for Tavily and add the key to .env**

Go to `https://tavily.com`, create an account, copy your API key.

In your local `.env` (not `.env.example`), add:
```
TAVILY_API_KEY=tvly-...
```

- [ ] **Step 6: Add startup check to main.py**

In `main.py`, find the top of `main()`:

```python
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py \"<debate topic>\"", file=sys.stderr)
        sys.exit(2)
```

Add an `os` import at the top of the file (alongside `import sys`):

```python
import os
import sys
```

Then insert the key check immediately after the argv check:

```python
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
```

- [ ] **Step 7: Verify the startup check fails loudly when the key is missing**

Temporarily unset the key in the current shell:
```bash
TAVILY_API_KEY= python main.py "Is the sky blue?"
```

Expected: the command exits with code 2 and prints the `TAVILY_API_KEY not set...` message. This is the failure-mode check for this task.

- [ ] **Step 8: Verify the startup check passes when the key is set**

Run (with `.env` containing a real key):
```bash
python main.py "Is the sky blue?"
```

Expected: the debate starts running normally (we haven't wired tools in yet, so it will use only the LLM — that's fine for now). Let it finish or Ctrl-C after you see the moderator's opening question — the rest of the task will be validated in Task 11.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .env.example main.py
git commit -m "$(cat <<'EOF'
chore: add langchain-tavily dep and TAVILY_API_KEY startup check

Prepares main.py to fail fast when the Tavily key is missing and
documents the new env var in .env.example. No behavioural change
to the debate yet — tools are wired in subsequent tasks.
EOF
)"
```

---

## Task 2: Create `debate/tools.py`

**Files:**
- Create: `debate/tools.py`

- [ ] **Step 1: Create the new module**

Create `debate/tools.py` with this exact content:

```python
"""Tavily search + extract, wrapped as LangChain tools for debater use.

Exposed as LangChain Tool instances so they trace cleanly in LangSmith
and Langfuse under their canonical names (tavily_search, tavily_extract).

This module is the one place to tune retrieval knobs (max_results,
search_depth, extract_depth) without touching orchestration code.
"""

from langchain_tavily import TavilyExtract, TavilySearch


def get_research_tools() -> list:
    """Return [search, extract] with debate-appropriate defaults.

    max_results=4 keeps each search response to roughly 1.5k input tokens.
    search_depth='basic' is 1 credit per call; 'advanced' is 2 credits
    and rarely worth it for debate-scale queries.
    include_answer=False — we want raw sources the LLM cites, not
    Tavily's pre-synthesised one-liner.
    """
    search = TavilySearch(
        max_results=4,
        search_depth="basic",
        include_answer=False,
    )
    extract = TavilyExtract(extract_depth="basic")
    return [search, extract]
```

- [ ] **Step 2: Verify the tools construct without error**

Run:
```bash
python -c "from debate.tools import get_research_tools; tools = get_research_tools(); print([t.name for t in tools])"
```

Expected output: `['tavily_search', 'tavily_extract']` (or similar — the exact `.name` attribute comes from langchain-tavily; confirm both tool names appear).

If the names differ, **note them down** — you'll need to know them in Task 5 when the user-written loop dispatches by tool name.

- [ ] **Step 3: Sanity-check a live search call**

This consumes one Tavily credit. Run:
```bash
python -c "
from debate.tools import get_research_tools
import json
search, _ = get_research_tools()
result = search.invoke({'query': 'What is LangGraph?'})
print(json.dumps(result, indent=2)[:500])
"
```

Expected: a JSON dict with a `results` key containing a list of `{title, url, content, ...}` entries. If you see a `401` or `unauthorized`, your `TAVILY_API_KEY` is wrong.

- [ ] **Step 4: Commit**

```bash
git add debate/tools.py
git commit -m "$(cat <<'EOF'
feat: add debate/tools.py with Tavily search + extract

Single-responsibility module that exposes get_research_tools()
returning [TavilySearch, TavilyExtract] configured for debate-scale
queries (max_results=4, basic depth, no pre-synthesised answer).
Nothing is wired into the graph yet.
EOF
)"
```

---

## Task 3: Extract `_build_debater_prompt()` helper

This is a pure refactor. No behavioural change. We factor out the prompt body currently duplicated between `proposer_node` and `critic_node` so Task 6 can wire both to `_run_with_tools()` without repeating ourselves.

**Files:**
- Modify: `debate/nodes.py`

- [ ] **Step 1: Add the helper function**

Open `debate/nodes.py`. Find `_format_transcript` (line 80-ish). **Immediately after** `_format_transcript`, add:

```python
def _build_debater_prompt(state: DebateState, round_n: int, side: str) -> str:
    """Build the user-message body shared by proposer and critic.

    `side` is either 'for' (proposer) or 'against' (critic); it controls
    only the final instruction sentence.
    """
    task_sentence = {
        "for": "Make your case for the proposition.",
        "against": "Rebut the proposer's latest argument.",
    }[side]

    return (
        f"Topic: {state['topic']}\n\n"
        f"Current focus question: {state['focus_question']}\n\n"
        f"Debate so far:\n{_format_transcript(state['transcript'])}\n\n"
        f"{task_sentence} Round {round_n}."
    )
```

- [ ] **Step 2: Use the helper in `proposer_node`**

Replace the existing `user_prompt = (...)` block in `proposer_node` with a call to the helper. The function should now look like:

```python
def proposer_node(state: DebateState, config: RunnableConfig) -> dict:
    """Argues FOR the proposition."""
    from debate.prompts import PROPOSER_SYSTEM

    round_n = state["round_num"] + 1
    user_prompt = _build_debater_prompt(state, round_n, side="for")

    llm_config = {**config, "run_name": f"proposer-argument-r{round_n}"}
    response = _model_for("proposer").invoke(
        [SystemMessage(content=PROPOSER_SYSTEM), HumanMessage(content=user_prompt)],
        config=llm_config,
    )

    return {
        "transcript": [Turn(role="proposer", content=response.content, round_num=round_n)]
    }
```

- [ ] **Step 3: Use the helper in `critic_node`**

Same pattern for the critic:

```python
def critic_node(state: DebateState, config: RunnableConfig) -> dict:
    """Argues AGAINST the proposition, engaging with proposer's latest turn."""
    from debate.prompts import CRITIC_SYSTEM

    round_n = state["round_num"] + 1
    user_prompt = _build_debater_prompt(state, round_n, side="against")

    llm_config = {**config, "run_name": f"critic-rebuttal-r{round_n}"}
    response = _model_for("critic").invoke(
        [SystemMessage(content=CRITIC_SYSTEM), HumanMessage(content=user_prompt)],
        config=llm_config,
    )

    return {
        "transcript": [Turn(role="critic", content=response.content, round_num=round_n)]
    }
```

- [ ] **Step 4: Verify no behaviour change**

Run a short debate end-to-end (any cheap topic will do):

```bash
python main.py "Is Python a better first language than JavaScript?"
```

Expected: the debate runs to completion with a VERDICT. Don't worry about output *quality* here — we're only checking that the refactor didn't break anything. If you see `NameError` or `KeyError: 'for'`/`'against'`, you've mis-typed the side value somewhere.

- [ ] **Step 5: Commit**

```bash
git add debate/nodes.py
git commit -m "$(cat <<'EOF'
refactor: extract _build_debater_prompt helper in debate/nodes.py

Factors the user-prompt body shared between proposer_node and
critic_node into one helper. Pure refactor — behaviourally identical.
Sets up the shared call site for the ReAct-loop wiring in later tasks.
EOF
)"
```

---

## Task 4: Scaffold `_run_with_tools()` (everything except the loop body)

**Files:**
- Modify: `debate/nodes.py`

The helper is added with everything in place EXCEPT the ReAct loop itself — that's Task 5 (user-written). After this task, `_run_with_tools` exists and is importable, but calling it would always fall through to the forced-close path.

- [ ] **Step 1: Update imports at the top of `debate/nodes.py`**

Currently:
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from debate.state import DebateState, Turn
```

Change to:
```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from debate.state import DebateState, Turn
```

(`AIMessage` is not listed because we don't explicitly construct it — `llm.invoke()` returns one and we just append it as-is.)

- [ ] **Step 2: Add `MAX_TOOL_CALLS` constant**

Find the existing `MAX_TOKENS_PER_TURN = 500` line near the top. Immediately below it, add:

```python
MAX_TOOL_CALLS = 4
```

- [ ] **Step 3: Add the `_run_with_tools` helper**

Add this function anywhere after `_model_for` and before `proposer_node`. The `...` inside the loop-marker block is intentional — that's Task 5's work.

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

    Protocol: the LLM may emit tool_calls on each invocation. We execute
    them and feed the results back as ToolMessages until the LLM produces
    prose (no tool_calls). Hard-capped at MAX_TOOL_CALLS iterations as
    defence-in-depth; beyond that we force one more tool-free invocation
    to produce a final answer from whatever evidence was gathered.
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    llm = _model_for(role).bind_tools(tools)
    tool_by_name = {t.name: t for t in tools}

    # ===== TASK 5: USER-WRITTEN REACT LOOP GOES HERE =====
    # This block implements the ReAct protocol. See plan Task 5 for full hint.
    # Your loop must either:
    #   - return response.content when the LLM emits no tool_calls, OR
    #   - fall through to the forced-close path below when MAX_TOOL_CALLS hit.
    # =====================================================

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

- [ ] **Step 4: Verify the file imports without error**

Run:
```bash
python -c "from debate.nodes import _run_with_tools, MAX_TOOL_CALLS; print(MAX_TOOL_CALLS)"
```

Expected output: `4`. If you see an `ImportError` or `SyntaxError`, re-check the imports and the function body.

- [ ] **Step 5: Commit**

```bash
git add debate/nodes.py
git commit -m "$(cat <<'EOF'
feat: scaffold _run_with_tools helper (loop body pending)

Adds the ReAct-loop helper signature, messages setup, LLM binding,
tool-name lookup, and forced-close fallback. The loop body itself
is a marked TODO block (Task 5) — currently calls always fall through
to the forced-close path, so the helper is not yet wired into
proposer/critic nodes.
EOF
)"
```

---

## Task 5: **[USER-WRITTEN]** Implement the ReAct loop body

> 🛑 **AGENTS: STOP HERE.** This task is reserved for the project owner (learning-mode contribution). Do not write the loop body in their place. Hand back to the user.

**Files:**
- Modify: `debate/nodes.py` (the marked block inside `_run_with_tools`)

### Context (read before writing)

You're writing the core of a ReAct agent. The protocol:

1. Ask the LLM to continue the conversation.
2. If the LLM replies with text (no tool calls), it's done — return that text.
3. If the LLM replies with tool calls, execute them, feed the results back, and loop.

### Rules that matter

- **Append the LLM's response BEFORE appending tool results.** Claude's conversation history requires: `Human → [Assistant-with-tool_calls → ToolMessage → ToolMessage] → Assistant-with-final-prose`. If you skip the assistant turn, Claude sees a conversation that jumped and will either repeat the tool call or refuse.
- **Use `tool_by_name[tc["name"]].invoke(tc["args"])`** to execute a tool call. The `tc` dict is from `response.tool_calls` (LangChain's normalized shape, already a dict).
- **Wrap each tool call in `try/except Exception as e`** and use `f"Tool error: {e}"` as the content. Your prompt tells the LLM to recover from this; don't propagate.
- **Construct the `ToolMessage` with `tool_call_id=tc["id"]`** — Claude requires this ID to match its own call.
- **Tag each iteration** with `run_name=f"{run_name_prefix}-iter{i}"` in the config so it shows up as its own span in LangSmith/Langfuse.
- **If you exit the `for` loop normally** (budget exhausted without prose), do nothing — the forced-close code after the block handles that case.

### Shape of the code (you write the details)

```
for i in range(MAX_TOOL_CALLS):
    <invoke the LLM on messages, with run_name="{prefix}-iter{i}">
    <append the response to messages>
    <if no tool_calls in the response: return response.content>
    <for each tool_call in response.tool_calls:>
        <try: invoke the right tool; except: wrap the error message>
        <append a ToolMessage with the result and the tool_call_id>
```

About ~10 lines of actual code. All the imports (`ToolMessage`, `BaseMessage`) are already there from Task 4.

### After you've written it

- [ ] **Step 1: Write the loop**

Replace the marker block (between `# ===== TASK 5 ... =====` and `# =====...=====`) with your implementation.

- [ ] **Step 2: Smoke-test the loop with a short debate**

```bash
python main.py "Is strong coffee better than weak coffee?"
```

Expected: the debate runs to completion. But note — you haven't wired `_run_with_tools` into the nodes yet (that's Task 6), so tools won't actually fire. The point of this run is to verify the file still imports and runs without `SyntaxError` / `NameError`.

- [ ] **Step 3: Commit**

```bash
git add debate/nodes.py
git commit -m "$(cat <<'EOF'
feat: implement ReAct loop body in _run_with_tools

Completes the hand-rolled ReAct protocol: the LLM can emit tool_calls
which are executed and fed back as ToolMessages until prose emerges
or MAX_TOOL_CALLS is hit. Tool errors are caught and surfaced as
ToolMessages so the prompted fallback ("argue from principle if
search fails") can take over.
EOF
)"
```

---

## Task 6: Wire `_run_with_tools` into `proposer_node` and `critic_node`

**Files:**
- Modify: `debate/nodes.py`

- [ ] **Step 1: Rewrite `proposer_node`**

Replace the entire current `proposer_node` function with:

```python
def proposer_node(state: DebateState, config: RunnableConfig) -> dict:
    """Argues FOR the proposition, with access to research tools."""
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

    return {
        "transcript": [Turn(role="proposer", content=content, round_num=round_n)]
    }
```

- [ ] **Step 2: Rewrite `critic_node`**

Symmetric. Replace the entire current `critic_node` function with:

```python
def critic_node(state: DebateState, config: RunnableConfig) -> dict:
    """Argues AGAINST the proposition, with access to research tools."""
    from debate.prompts import CRITIC_SYSTEM
    from debate.tools import get_research_tools

    round_n = state["round_num"] + 1
    user_prompt = _build_debater_prompt(state, round_n, side="against")

    content = _run_with_tools(
        role="critic",
        system_prompt=CRITIC_SYSTEM,
        user_prompt=user_prompt,
        tools=get_research_tools(),
        config=config,
        run_name_prefix=f"critic-rebuttal-r{round_n}",
    )

    return {
        "transcript": [Turn(role="critic", content=content, round_num=round_n)]
    }
```

- [ ] **Step 3: Run a debate and verify tool spans appear**

```bash
python main.py "Should all algorithmic decisions affecting citizens be legally required to be auditable?"
```

This topic is policy-adjacent and should trigger some research. The debate may take 1–3 minutes.

- [ ] **Step 4: Open the LangSmith and Langfuse traces**

Scroll the terminal output to the two trace URLs printed at the end. Open both.

In LangSmith: expand the run tree. Under `proposer` and/or `critic` nodes you should see at least one child span labeled `tavily_search` (and possibly `tavily_extract`). You should also see the per-iteration run_names like `proposer-argument-r1-iter0`, `-iter1`.

In Langfuse: the Observations table should show the tool calls as their own rows.

If **no tool spans appear at all**, the prompts haven't been updated yet (Task 7) — that's expected for now; the LLM doesn't know the tools exist. Running the full end-to-end validation is Task 9–11.

- [ ] **Step 5: Commit**

```bash
git add debate/nodes.py
git commit -m "$(cat <<'EOF'
feat: wire _run_with_tools into proposer and critic nodes

Both debater nodes now run the ReAct loop with tavily_search and
tavily_extract. Moderator, judge, and setup remain unchanged
(pure-LLM, as decided in Approach 1). The model still doesn't know
the tools exist until prompts are updated in the next task.
EOF
)"
```

---

## Task 7: Update prompts with the "Research strategy" block

**Files:**
- Modify: `debate/prompts.py`

- [ ] **Step 1: Append the research strategy block to `PROPOSER_SYSTEM`**

Open `debate/prompts.py`. Find `PROPOSER_SYSTEM`. Change from:

```python
PROPOSER_SYSTEM = """You are arguing FOR the proposition in a formal Oxford Union debate. \
Present your strongest case: make a clear claim, support it with reasoning, and acknowledge \
(without conceding) the most serious counterargument. Engage directly with any prior critic turn \
if one exists — ignoring it looks weak. Maintain a rigorous register: precise language, tight \
reasoning, no padding. Do not hedge unnecessarily, but do not overstate. \
Length: three to five short paragraphs."""
```

to:

```python
PROPOSER_SYSTEM = """You are arguing FOR the proposition in a formal Oxford Union debate. \
Present your strongest case: make a clear claim, support it with reasoning, and acknowledge \
(without conceding) the most serious counterargument. Engage directly with any prior critic turn \
if one exists — ignoring it looks weak. Maintain a rigorous register: precise language, tight \
reasoning, no padding. Do not hedge unnecessarily, but do not overstate. \
Length: three to five short paragraphs.

You have access to two research tools:
- tavily_search: find 3-5 relevant sources for a specific query. Use this first; \
the returned snippets are often enough for citation.
- tavily_extract: fetch a specific URL's full text when a snippet is too thin \
for a proper quotation.

Research strategy:
- Open your turn with one well-crafted tavily_search query to ground the argument \
in current evidence. One focused query beats five scattered ones.
- Only tavily_extract when the snippet is genuinely insufficient.
- You have a hard budget of 4 tool calls per turn. Using fewer is often correct.
- Cite inline as markdown: [claim](https://url). Do not invent URLs.
- If search returns nothing useful, argue from principle — do not apologize for \
the absence of sources or hedge your claim. A strong rhetorical turn that \
acknowledges the limit of available evidence is better than a weak apology."""
```

- [ ] **Step 2: Append the research strategy block to `CRITIC_SYSTEM`**

Find `CRITIC_SYSTEM`. Change from:

```python
CRITIC_SYSTEM = """You are arguing AGAINST the proposition in a formal Oxford Union debate. \
Steelman the proposer's position first, then attack its strongest form — not a strawman. Make a \
clear counter-claim, support it with reasoning, and acknowledge the best argument for the other \
side without conceding. Rigorous register: precise language, tight reasoning, no padding. Direct \
engagement with the proposer's latest argument is expected. \
Length: three to five short paragraphs."""
```

to:

```python
CRITIC_SYSTEM = """You are arguing AGAINST the proposition in a formal Oxford Union debate. \
Steelman the proposer's position first, then attack its strongest form — not a strawman. Make a \
clear counter-claim, support it with reasoning, and acknowledge the best argument for the other \
side without conceding. Rigorous register: precise language, tight reasoning, no padding. Direct \
engagement with the proposer's latest argument is expected. \
Length: three to five short paragraphs.

You have access to two research tools:
- tavily_search: find 3-5 relevant sources for a specific query. Use this first; \
the returned snippets are often enough for citation.
- tavily_extract: fetch a specific URL's full text when a snippet is too thin \
for a proper quotation.

Research strategy:
- Open your turn with one well-crafted tavily_search query to ground the argument \
in current evidence. One focused query beats five scattered ones.
- Only tavily_extract when the snippet is genuinely insufficient.
- You have a hard budget of 4 tool calls per turn. Using fewer is often correct.
- Cite inline as markdown: [claim](https://url). Do not invent URLs.
- If search returns nothing useful, argue from principle — do not apologize for \
the absence of sources or hedge your claim. A strong rhetorical turn that \
acknowledges the limit of available evidence is better than a weak apology."""
```

(The block is identical for both — intentional; both roles use the same research discipline.)

- [ ] **Step 3: Verify prompts still import cleanly**

```bash
python -c "from debate.prompts import PROPOSER_SYSTEM, CRITIC_SYSTEM; print(len(PROPOSER_SYSTEM), len(CRITIC_SYSTEM))"
```

Expected: two integers in the ~1200–1500 range (up from ~500 before the block was added). If you see `SyntaxError`, check for missing line continuations (`\`) on long lines.

- [ ] **Step 4: Commit**

```bash
git add debate/prompts.py
git commit -m "$(cat <<'EOF'
feat: teach proposer and critic how to use the research tools

Both system prompts gain a Research Strategy block telling the LLM
when to tavily_search, when to tavily_extract, the 4-call budget,
citation format, and — critically — how to argue gracefully when
search yields nothing. Last instruction prevents the common weak
"I was unable to find sources, so..." opener.
EOF
)"
```

---

## Task 8: Update `main.py` metadata and bump `prompts_version`

**Files:**
- Modify: `main.py`

This closes the loop on observability. After this task, LangSmith/Langfuse traces are filterable between v1 (no tools) and v2 (tools) runs, which is how you'll A/B-compare argument quality.

- [ ] **Step 1: Import `MAX_TOOL_CALLS` from `debate.nodes`**

In `main.py`, the `from debate.graph import build_graph` import currently reads:
```python
from debate.graph import build_graph  # noqa: E402
from debate.state import DebateState  # noqa: E402
```

Add a new import for `MAX_TOOL_CALLS` (same noqa treatment because `load_dotenv()` must run first):

```python
from debate.graph import build_graph  # noqa: E402
from debate.nodes import MAX_TOOL_CALLS  # noqa: E402
from debate.state import DebateState  # noqa: E402
```

- [ ] **Step 2: Update metadata and tags in `build_run_config`**

Find the current `build_run_config` body:

```python
    run_name = f"debate:{topic[:40]}"
    tags = ["debate", "learning", "experiment:v1"]
    # Langfuse v4 requires metadata values to be strings; coerce.
    metadata = {
        "topic": topic,
        "max_rounds": str(max_rounds),
        "model": "claude-sonnet-4-6",
        "prompts_version": "v1",
    }
```

Change to:

```python
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
```

- [ ] **Step 3: Verify the config builds**

```bash
python -c "
import os; os.environ.setdefault('TAVILY_API_KEY', 'stub')
from dotenv import load_dotenv; load_dotenv()
from main import build_run_config
cfg = build_run_config('test topic', 3)
print(cfg['tags'])
print(cfg['metadata'])
"
```

Expected: you see `tools:v1` in the tags list and `max_tool_calls: '4'`, `prompts_version: 'v2'`, `search_provider: 'tavily'` in the metadata dict.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "$(cat <<'EOF'
feat: bump prompts_version to v2 and add tools metadata

Adds max_tool_calls and search_provider to run metadata and
tools:v1 to tags, so LangSmith and Langfuse can filter and
compare v1 (no tools) runs against v2 (Tavily-enabled) runs
on the same topic.
EOF
)"
```

---

## Task 9: Smoke test — any topic, end-to-end

**Files:** none modified. This is a validation gate.

- [ ] **Step 1: Run a single short debate**

```bash
python main.py "Is Python a better first language than JavaScript?"
```

Expected:
- The debate runs to completion with a VERDICT.
- At least one `tavily_search` call appears in the terminal output (LangChain logs them at INFO level; you may need to scroll).
- Two trace URLs print at the end.

- [ ] **Step 2: Open a LangSmith trace**

Open the LangSmith URL. Find the most recent run.

Expand the tree. Confirm you see:
- Iteration spans: `proposer-argument-r1-iter0`, `proposer-argument-r1-iter1`, … (at least `iter0` must exist).
- At least one nested `tavily_search` span with a `query` input.
- No `*-forced-close` spans (if you see these, the LLM hit the budget cap, which is suspicious on a light topic — look at the actual tool calls it made).

- [ ] **Step 3: Open a Langfuse trace**

Open the Langfuse URL. Confirm the same information is visible.

In the Observations table, filter by `name = tavily_search` and see how many calls were made. On a light topic this should be 1–2 per debater turn.

- [ ] **Step 4: Note anything odd**

Write down anything that surprised you in a scratch note — too many searches? None at all? Tool errors? This observation set will inform whether further tuning is needed.

**Do not commit anything in this task — it's a validation gate.** If this fails (e.g., zero tool spans appear anywhere), revisit Task 6 (wiring) and Task 7 (prompts) before continuing.

---

## Task 10: Validation — pure-principles topic (expect little-to-no tool use)

**Files:** none modified. Validation gate.

The point: verify the LLM has learned *when not to search*. A pure-ethics topic should generate minimal or zero tool calls.

- [ ] **Step 1: Run the debate**

```bash
python main.py "Is lying ever morally required?"
```

- [ ] **Step 2: Inspect the trace**

Open LangSmith or Langfuse. Count the total `tavily_search` + `tavily_extract` calls across the whole debate.

Expected: **0–4 total** across all rounds and both debaters. If you see 10+, something's off — either the prompt is over-encouraging search or the model has a quirk. Note it; don't fix yet.

- [ ] **Step 3: Read the transcript**

Scroll the terminal output. Read one proposer turn and one critic turn. Are there inline `[claim](https://url)` citations?

Expected: possibly a few, possibly none. On an ethics topic citations are a bonus, not required. The argument should stand on principle.

- [ ] **Step 4: Note observations**

Write down tool call counts and any weirdness. This is validation data.

---

## Task 11: Validation — research-hungry topic (expect heavy tool use)

**Files:** none modified. Validation gate.

The other half of the contrast: a topic requiring recent facts should trigger significant search.

- [ ] **Step 1: Run the debate**

```bash
python main.py "Should EU AI Act general-purpose AI rules be delayed past August 2026?"
```

This topic is deliberately policy-specific, recent, and controversial. Expect 1–2 minutes runtime and noticeable search activity.

- [ ] **Step 2: Count tool spans**

Open the trace. Expected across the full debate:
- **3–10 `tavily_search` calls** (both debaters, across all rounds).
- **0–3 `tavily_extract` calls** (only when a snippet was too thin).
- **0 `*-forced-close` spans** (the budget cap should be enough).

- [ ] **Step 3: Check for inline citations in the transcript**

At least one proposer or critic turn should contain at least one `[text](https://url)` link. If there are zero, the LLM searched but didn't cite — might be a prompt issue worth revisiting later.

- [ ] **Step 4: Note any tool errors**

Filter the trace for spans with errors. Tavily can be flaky. If you see any, that's real-world data — note the error type; the prompted fallback should have kicked in.

---

## Task 12: v1 vs v2 A/B comparison

**Files:** none modified. Final validation — the payoff of the whole project.

- [ ] **Step 1: Run the same research-hungry topic on the v2 build**

If you still have the Task 11 trace URL, skip to Step 2. Otherwise:

```bash
python main.py "Should EU AI Act general-purpose AI rules be delayed past August 2026?"
```

Note the trace URLs.

- [ ] **Step 2: Find a v1 (pre-tools) trace on the same topic**

The cleanest option: temporarily check out the state before Task 1 was committed (that's commit `05b6096` — the spec-commit on `main` before any implementation landed), run the same topic, then come back to `main`:

```bash
git stash  # if you have any uncommitted work
git checkout 05b6096
python main.py "Should EU AI Act general-purpose AI rules be delayed past August 2026?"
# note the trace URL, then:
git checkout main
git stash pop  # only if you stashed in the first step
```

Alternative if you don't want to re-run: filter LangSmith or Langfuse by `prompts_version: v1` and grab any historical trace on a similar topic. Less apples-to-apples but faster.

- [ ] **Step 3: Compare the two transcripts side by side**

Open both final verdicts in adjacent browser tabs. Questions to answer:
1. Is the v2 argument more specific / concrete / evidence-grounded?
2. Does it cite real URLs? Do those URLs point to relevant articles?
3. Is it longer? Is the extra length useful or padding?
4. Did the v2 judge render a different verdict, or the same one with more conviction?

- [ ] **Step 4: Compare cost and latency**

In LangSmith or Langfuse, open the root run and note:
- Total input tokens (v1 vs v2)
- Total output tokens (should be similar; `max_tokens=500` unchanged)
- Total cost
- Total latency

Expected: v2 input tokens ~5–10× higher, cost ~3–5× higher, latency ~2–3× higher. That's the real price of research.

- [ ] **Step 5: Write a one-paragraph reflection**

Not a commit; a personal note. Was the v2 argument worth the extra cost? What would you tune next? (Candidate: try `search_depth="advanced"` on a single run to see if it improves citation quality.)

### Optional: stress the rate-limit path

The spec mentions a rate-limit/error-handling check. This is optional — only do it if you're curious about the error path, because it will burn ~20 Tavily credits:

```bash
for i in 1 2 3 4 5; do python main.py "Same research-hungry topic" & done
wait
```

Then in LangSmith/Langfuse, filter for failed spans. You're looking for: did the prompted fallback ("argue from principle if search fails") actually kick in? Are the arguments still readable when search errored?

---

## Completion checklist

At the end of Task 12, the following should all be true:

- [ ] All 8 implementation tasks (1–8) have been committed.
- [ ] All 4 validation tasks (9–12) have been run and observations noted.
- [ ] The `proposer` and `critic` debate nodes call Tavily via a hand-rolled ReAct loop.
- [ ] Moderator, judge, setup, state, and graph.py are unchanged.
- [ ] LangSmith and Langfuse both show per-iteration spans and tool spans.
- [ ] `prompts_version: v2` and `tools:v1` tag are visible in trace metadata.
- [ ] The pure-principles topic showed minimal tool use; the research-hungry topic showed significant tool use.
- [ ] One git log command (`git log --oneline ^<pre-task-1> HEAD`) shows 7 commits from this plan (one per implementation task — no validation commits).
