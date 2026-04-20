# LangChain Debate Agent

> A multi-role debate agent built with **LangGraph** + **Claude Sonnet 4.6**, instrumented with dual tracing to **LangSmith** and **Langfuse**. A personal learning playground for getting hands-on with LangChain / LangGraph and modern LLM observability.

---

## What it does

Given a proposition, it runs a structured Oxford-Union-style debate. Five roles share one Claude model; each has its own system prompt and temperature:

- **Setup** — initialises the state.
- **Moderator (opening)** — frames a sharp focus question from the topic.
- **Proposer** — argues *for* the proposition, engages directly with the critic's previous turn.
- **Critic** — argues *against*, must steelman the proposer's position before attacking.
- **Moderator (distillation)** — between rounds, distils agreement/disagreement and poses a sharper follow-up.
- **Judge** — at the end of each round, decides whether to continue or emit a verdict. Honors an optional `max_rounds` cap.

The graph loops through proposer → critic → moderator-distill → judge until the judge rules or the cap is hit.

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
- **Per-node trace labels.** Each LLM call inside a node is tagged with a `run_name` like `proposer-argument-r1`, so you can filter traces by role-and-round in the UI.

For the full design and decision history, see [`docs/superpowers/specs/`](docs/superpowers/specs/) and [`docs/superpowers/plans/`](docs/superpowers/plans/).

---

## Stack

| Piece | Why |
|---|---|
| **LangGraph** | State machine for agent orchestration. Modern replacement for `AgentExecutor`. Gives inspectable per-node trace spans. |
| **langchain-anthropic** | Claude integration. Currently points at `claude-sonnet-4-6`. |
| **LangSmith** | Hosted tracing. Auto-attaches via env vars — zero code. Best in class for LangChain-native observability. |
| **Langfuse** | Alternative hosted tracing, attached via an explicit callback handler. Open-source, self-hostable. |
| **python-dotenv** | Loads `.env` before LangChain imports so LangSmith's auto-attach actually sees the env vars. |

Both tracers are wired simultaneously. This is *on purpose* — it demonstrates that tracing in LangChain is callback-based and vendor-neutral; you can A/B compare the UIs from the same run.

---

## Quick start

You'll need accounts at three services:

1. **Anthropic** — `console.anthropic.com`, get an API key. Costs ~$0.05 per full debate on Sonnet 4.6.
2. **LangSmith** — `smith.langchain.com`. Free tier: 5k traces/month.
3. **Langfuse** — `cloud.langfuse.com` (EU) or `us.cloud.langfuse.com` (US). Free tier: 50k observations/month.

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
# LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY. Leave the other fields as-is.

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
│   ├── nodes.py            # node functions + model factory
│   └── graph.py            # build_graph() — topology
└── docs/
    └── superpowers/
        ├── specs/          # design doc with trade-offs and rationale
        └── plans/          # task-by-task implementation plan
```

Each module has one responsibility:

- `state.py` — types only, no logic.
- `prompts.py` — constants only, no imports beyond docstring. **This is where you iterate most** when tuning voice.
- `nodes.py` — node functions that read state + config, call the LLM, return state deltas. The model factory is here too.
- `graph.py` — short but architecturally important. Registers nodes, wires edges, compiles.
- `main.py` — thin entry point: load env, build graph, invoke, print.

---

## What to experiment with

Iterations are cheap. The feedback loop is: edit → run → inspect trace → repeat.

| Tweak | Where | What you'll learn |
|---|---|---|
| Rewrite a prompt | `debate/prompts.py` | How much voice/quality depends on prompt wording. Bump `prompts_version` in `main.py` metadata so you can compare v1/v2 in traces. |
| Change termination rule | `judge_node` in `debate/nodes.py` | Early-stop vs always-run-max-rounds: does forcing more rounds produce richer debates? Measure avg-rounds-to-verdict in traces. |
| Try a new topic | `python main.py "..."` | How the moderator's opening question shapes the debate's entire trajectory. |
| Tweak `max_rounds` | `main.py` (hardcoded to 3) | Does a 5-round debate converge or just repeat? |
| Swap model per role | `_ROLE_TEMPERATURES` + `_model_for` | Try Haiku for moderator (cheap/fast) + Opus for judge (deep reasoning). Cost vs quality trade-off in traces. |
| Add a new role | `debate/nodes.py` + `debate/graph.py` | E.g., a "devil's advocate" that introduces a fresh angle mid-round. Tests your understanding of node registration + conditional edges. |

---

## Scope / non-goals

Deliberately out of scope — this is a learning playground, not a product:

- No unit tests (manual validation via traces).
- No persistence (each run is ephemeral).
- No UI or API — CLI only.
- No streaming output (the full transcript prints on completion).
- No retrieval / tool use — the agent is purely LLM + state.
- No multi-user session handling.

If you want to extend into any of those, fork freely.

---

## The learning journey

The `docs/superpowers/` directory contains the design spec and implementation plan that shaped this build, written before any code. Worth reading if you want to see the decision-making:

- [`docs/superpowers/specs/2026-04-20-langchain-debate-agent-design.md`](docs/superpowers/specs/2026-04-20-langchain-debate-agent-design.md) — architecture, state shape, observability strategy, per-role trade-offs.
- [`docs/superpowers/plans/2026-04-20-langchain-debate-agent.md`](docs/superpowers/plans/2026-04-20-langchain-debate-agent.md) — task-by-task build plan with commit boundaries.

`git log --oneline` also tells the whole story cleanly — scaffolder commits alternating with "implement X" commits for each of the four decision points (prompts, moderator branching, judge termination, run-config).

---

## License

MIT — do whatever you want with it.
# langchain_debater
