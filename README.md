# LangChain Debate Agent

> A multi-role debate agent built with **LangGraph** + **Claude Sonnet 4.6**. The proposer and critic have **web research tools** (Tavily search + extract) and use them via a hand-rolled ReAct loop — arguments are grounded in live web evidence rather than model memory alone. Dual tracing to **LangSmith** and **Langfuse** captures every tool call, every ReAct iteration, every decision. A personal learning playground for getting hands-on with LangChain / LangGraph, agent tool use, and modern LLM observability.

---

## What it does

Given a proposition, it runs a structured Oxford-Union-style debate. Five roles share one Claude model; each has its own system prompt and temperature:

- **Setup** — initialises the state.
- **Moderator (opening)** — frames a sharp focus question from the topic.
- **Proposer** — argues *for* the proposition, engages directly with the critic's previous turn. **Has research tools.**
- **Critic** — argues *against*, must steelman the proposer's position before attacking. **Has research tools.**
- **Moderator (distillation)** — between rounds, distils agreement/disagreement and poses a sharper follow-up.
- **Judge** — at the end of each round, decides whether to continue or emit a verdict. Honors an optional `max_rounds` cap.

The graph loops through proposer → critic → moderator-distill → judge until the judge rules or the cap is hit. Inside `proposer` and `critic`, each turn runs a **ReAct tool-call loop**: the LLM can call `tavily_search` / `tavily_extract` up to `MAX_TOOL_CALLS=4` times before producing its final prose. Sources appear inline as markdown links `[claim](https://url)`.

### Example output (excerpt)

```
TOPIC: Is transparent algorithmic curation preferable to editorial human curation...
[MODERATOR, round 0]
Should news platforms be required to replace editorial human curation with
transparent algorithmic curation, given that algorithms can be audited for bias
while human editors cannot be held to the same standard of systematic accountability?

[PROPOSER, round 1]
## The Case for Transparent Algorithmic Curation
The core claim is straightforward: transparent algorithmic curation is preferable...

[CRITIC, round 1]
## Counter-Claim: Auditability Is Not Accountability, and Transparency Is Not Neutrality
The proposer has constructed the strongest version of the algorithmic case...

[JUDGE, round 2]
VERDICT: The critic carried the debate on the motion as framed. The proposer's
case rested on the structural superiority of auditable, correctable algorithmic
failure, but the critic successfully narrowed this to a contingent empirical
claim about interpretability that has not been discharged for production systems
at operational scale...
```

Debate outcome, full transcript, and both trace URLs are printed on completion.

---

## Architecture

The whole agent is a LangGraph `StateGraph` with six nodes and one conditional edge:

```
                  ┌─────────┐
  START ─▶ setup ─▶moderator│ (opens: frames question)
                  └────┬────┘
                       ▼
                 ┌─────────┐
           ┌────▶│proposer │
           │     └────┬────┘
           │          ▼
           │     ┌─────────┐
           │     │ critic  │
           │     └────┬────┘
           │          ▼
           │     ┌─────────┐
           │     │moderator│ (distils round, poses sharper follow-up)
           │     └────┬────┘
           │          ▼
           │     ┌─────────┐
           │     │ judge   │ (continue? or emit VERDICT)
           │     └────┬────┘
           │          │
           │    ┌─────┴─────┐
           └────│ continue? │───▶ END
                └───────────┘
```

Notable design details:

- **Single moderator function, two graph positions.** `moderator_node` is registered as both `moderator_open` and `moderator_distill`. It branches internally on `len(transcript) == 0`. This is the idiomatic LangGraph pattern for reusing a callable across flow positions.
- **Termination via state, not a counter.** `verdict: str | None` in state is the single source of truth for "are we done?" — the judge parses its own output to set it. A `max_rounds` cap is defence-in-depth.
- **Per-node trace labels.** Each LLM call inside a node is tagged with a `run_name` like `proposer-argument-r1`, so you can filter traces by role-and-round in the UI. Tool-using turns add `-iter0`, `-iter1`, … suffixes for each ReAct iteration.
- **ReAct loops inside debater nodes.** `proposer` and `critic` are single nodes in the graph, but internally each runs a tool-call loop (`LLM → tool_calls → ToolMessages → LLM → …`) until the LLM produces prose or `MAX_TOOL_CALLS=4` is hit. The moderator, judge, and setup stay pure-LLM — they arbitrate structure, not evidence.

For the full design and decision history, see [`docs/superpowers/specs/`](docs/superpowers/specs/) and [`docs/superpowers/plans/`](docs/superpowers/plans/).

---

## Web research (how the debaters scrape online info)

Only **proposer and critic** have tools. Moderator and judge stay pure-LLM — they arbitrate the debate; they don't gather evidence.

Each debater turn runs a hand-rolled ReAct loop inside `_run_with_tools()` in `debate/nodes.py`:

1. Invoke the LLM on `[SystemMessage, HumanMessage, …]` with tools bound via `.bind_tools([search, extract])`.
2. If the response has no `tool_calls`, return its prose — that's the final argument.
3. Otherwise append the `AIMessage` (tool_calls intact), execute each call, append results as `ToolMessage`s, loop.
4. Hard cap at `MAX_TOOL_CALLS = 4`. Overrun → force one tool-free invocation ("use only the evidence above") to produce a clean close.

Two tools, both from [`langchain-tavily`](https://pypi.org/project/langchain-tavily/):

| Tool | Returns | When the LLM picks it |
|---|---|---|
| `tavily_search` | 4 results with `title`, `url`, ~500-char snippet | First call of a turn; snippets are often enough for citation. |
| `tavily_extract` | Full page as clean markdown | Only when a snippet is too thin for a proper quote. |

Sources appear in the transcript as inline markdown links `[claim](https://url)`. Citation discipline is enforced via prompt: fabricated URLs are explicitly forbidden.

**Tool errors don't crash the debate** — they're caught at the loop level, surfaced as `ToolMessage(content="Tool error: ...")`, and the prompt instructs the LLM to argue from principle without apologising. Errors **stay visible** in LangSmith/Langfuse — the rate-limit day isn't hidden, you'll see the 429 span.

See [`debate/tools.py`](debate/tools.py) for retrieval knobs (`max_results=4`, `search_depth="basic"`, `include_answer=False`) — tune in one place.

---

## Stack

| Piece | Why |
|---|---|
| **LangGraph** | State machine for agent orchestration. Modern replacement for `AgentExecutor`. Gives inspectable per-node trace spans. |
| **langchain-anthropic** | Claude integration. Currently points at `claude-sonnet-4-6`. |
| **langchain-tavily** | `TavilySearch` + `TavilyExtract` as LangChain `Tool` instances. Bound to debater models via `.bind_tools()`; auto-trace as their own spans. |
| **Tavily** | Web search + page-extract backend. LLM-friendly API: pre-trimmed snippets for search, clean markdown for extract. |
| **LangSmith** | Hosted tracing. Auto-attaches via env vars — zero code. Best in class for LangChain-native observability. |
| **Langfuse** | Alternative hosted tracing, attached via an explicit callback handler. Open-source, self-hostable. |
| **python-dotenv** | Loads `.env` before LangChain imports so LangSmith's auto-attach actually sees the env vars. |

Both tracers are wired simultaneously. This is *on purpose* — it demonstrates that tracing in LangChain is callback-based and vendor-neutral; you can A/B compare the UIs from the same run.

---

## Quick start

You'll need accounts at four services:

1. **Anthropic** — `console.anthropic.com`, get an API key. Costs ~$0.10–0.20 per full debate on Sonnet 4.6 with tools (up from ~$0.05 without — tool calls inflate context).
2. **LangSmith** — `smith.langchain.com`. Free tier: 5k traces/month.
3. **Langfuse** — `cloud.langfuse.com` (EU) or `us.cloud.langfuse.com` (US). Free tier: 50k observations/month.
4. **Tavily** — `tavily.com`. Free tier: 1,000 credits/month (1 credit per basic search or extract). Budget per debate is ≤24 calls, so ~40+ debates/month comfortably.

Then:

```bash
# clone, enter
git clone <this-repo>
cd PythonProject

# Python 3.13+ (ideally via uv, pyenv, or a PyCharm venv)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# keys
cp .env.example .env
# edit .env and paste in ANTHROPIC_API_KEY, LANGSMITH_API_KEY,
# LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, TAVILY_API_KEY.
# Leave the other fields as-is. main.py will fail fast if any required key is missing.

# run
python main.py "Should remote work be the default for knowledge workers?"
```

Output: the full transcript, a `VERDICT:` line, and two trace URLs. Open both.

---

## Reading the traces

You'll see the same run from two angles.

**LangSmith** — run list shows `debate:<first-40-chars-of-topic>`. Open a run; the trace tree nests: `LangGraph` root → `setup` → `moderator_open` → `proposer` → `critic` → ... Each LLM call is a child span labeled with `role-action-rN` (e.g., `proposer-argument-r1`). Click any span for inputs, outputs, token counts, cost, latency. Filter by tags (`debate`, `experiment:v1`) or metadata (`prompts_version`, `max_rounds`).

**Langfuse** — same shape, different UI. The Observations table gives you a flat per-span view (every node and LLM call). The Trace view gives you the tree.

Things worth noticing in both UIs:

- `judge_router` outputs "continue" or "end" — the routing decision is a captured span of its own.
- `langgraph_checkpoint_ns` in metadata — LangGraph adds it for free; shows exact graph position.
- `ls_temperature` varies by role: 0.7 for proposer/critic (creative), 0.3 for moderator/judge (consistent).
- `tavily_search` / `tavily_extract` — every tool call is its own child span under the debater node, with the query/URL as input and the raw JSON response as output. Latency and cost attached.
- `proposer-argument-r1-iter0`, `-iter1`, … — the per-iteration run names inside a ReAct loop. `iterN` tells you how many rounds the LLM took before producing prose.
- `-forced-close` — only appears when `MAX_TOOL_CALLS` was exhausted. Rare; flag as pathological and tune prompts.
- Tags now include `tools:v1`; metadata has `max_tool_calls`, `search_provider`, and `prompts_version: v2`. Filter on these to A/B v1 (pre-tools) vs v2 (tools) runs on the same topic.

---

## Project structure

```
.
├── .env.example            # env template
├── .gitignore
├── main.py                 # entry point: env + CLI + graph invocation
├── requirements.txt
├── debate/
│   ├── __init__.py
│   ├── state.py            # DebateState + Turn TypedDicts
│   ├── prompts.py          # 5 system prompts — main iteration surface
│   ├── tools.py            # Tavily search + extract as LangChain Tool instances
│   ├── nodes.py            # node functions + model factory + ReAct loop helper
│   └── graph.py            # build_graph() — topology
└── docs/
    └── superpowers/
        ├── specs/          # design doc with trade-offs and rationale
        └── plans/          # task-by-task implementation plan
```

Each module has one responsibility:

- `state.py` — types only, no logic.
- `prompts.py` — constants only, no imports beyond docstring. **This is where you iterate most** when tuning voice.
- `tools.py` — Tavily configuration, isolated from orchestration so retrieval knobs (`max_results`, `search_depth`) don't bleed into `nodes.py`.
- `nodes.py` — node functions that read state + config, call the LLM, return state deltas. Also houses `_run_with_tools()` (the ReAct loop) and `MAX_TOOL_CALLS`. The model factory is here too.
- `graph.py` — short but architecturally important. Registers nodes, wires edges, compiles.
- `main.py` — thin entry point: load env, validate keys, build graph, invoke, print.

---

## What to experiment with

Iterations are cheap. The feedback loop is: edit → run → inspect trace → repeat.

| Tweak | Where | What you'll learn |
|---|---|---|
| Rewrite a prompt | `debate/prompts.py` | How much voice/quality depends on prompt wording. Bump `prompts_version` in `main.py` metadata so you can compare versions in traces. |
| Soften the "always search" line | `PROPOSER_SYSTEM` / `CRITIC_SYSTEM` | The current prompt says *"Open your turn with one tavily_search query"* — pushes search even on ethics topics. Soften to *"search if you need evidence"* and compare tool-call counts in traces. |
| Change termination rule | `judge_node` in `debate/nodes.py` | Early-stop vs always-run-max-rounds: does forcing more rounds produce richer debates? Measure avg-rounds-to-verdict in traces. |
| Try a new topic | `python main.py "..."` | How the moderator's opening question shapes the debate's entire trajectory. |
| Tweak `max_rounds` | `main.py` (hardcoded to 3) | Does a 5-round debate converge or just repeat? |
| Tweak `MAX_TOOL_CALLS` | `debate/nodes.py` | 2 forces laser-focused queries; 8 invites loops. Does more research mean better arguments, or just more cost? |
| Toggle `search_depth="advanced"` | `debate/tools.py` | 2× Tavily credit cost per search. Does it meaningfully improve citation quality or just burn credits? |
| Swap model per role | `_ROLE_TEMPERATURES` + `_model_for` | Try Haiku for moderator (cheap/fast) + Opus for judge (deep reasoning). Cost vs quality trade-off in traces. |
| Add a new role | `debate/nodes.py` + `debate/graph.py` | E.g., a "devil's advocate" that introduces a fresh angle mid-round. Tests your understanding of node registration + conditional edges. |
| Extend tools to the judge (Approach 2) | `debate/nodes.py` + prompts | Judge becomes a fact-checker with receipts. The full migration is mapped in [`docs/superpowers/specs/2026-04-21-tavily-tool-use-design.md`](docs/superpowers/specs/2026-04-21-tavily-tool-use-design.md). |

---

## Scope / non-goals

Deliberately out of scope — this is a learning playground, not a product:

- No unit tests (manual validation via traces).
- No persistence (each run is ephemeral).
- No UI or API — CLI only.
- No streaming output (the full transcript prints on completion).
- No tool caching — same Tavily query costs a fresh credit each run.
- No cross-turn source dedup (the agent may re-fetch the same URL it pulled a round ago).
- Tool use scoped to proposer + critic only (Approach 1 / narrow). Extending to the judge for fact-checking is Approach 2; the spec maps it out but it's not wired in.
- No multi-user session handling.

If you want to extend into any of those, fork freely.

---

## License

MIT — do whatever you want with it.
