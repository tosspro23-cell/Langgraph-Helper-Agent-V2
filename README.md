# LangGraph Helper Agent

An AI agent, built with **LangGraph** and **LangChain (v1)**, that answers
practical developer questions about LangGraph and LangChain. It supports
two operating modes — **offline** (a locally pre-built index of the
official docs) and **online** (live web search) — selectable via a CLI
flag or environment variable.

```bash
python main.py --mode offline "How do I add persistence to a LangGraph agent?"
python main.py --mode online  "What changed in the latest LangGraph release?"
```

---

## Table of contents

- [Architecture overview](#architecture-overview)
- [Operating modes](#operating-modes)
- [Data freshness strategy](#data-freshness-strategy)
- [Setup](#setup)
- [Usage](#usage)
- [Choosing an LLM provider](#choosing-an-llm-provider)
- [Choosing a web search backend (online mode)](#choosing-a-web-search-backend-online-mode)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Design notes / tradeoffs](#design-notes--tradeoffs)

---

## Architecture overview

The agent is a small LangGraph `StateGraph` with one conditional branch:

```
                 START
                   │
                   ▼
            route_by_mode(state)          # reads state["mode"]
              │            │
        "offline"        "online"
              │            │
              ▼            ▼
     retrieve_offline   retrieve_online
              │            │
              └─────┬──────┘
                     ▼
              generate_answer
                     │
                     ▼
                    END
```

**State** (`langgraph_helper/state.py`) is a single `TypedDict` (`AgentState`)
threaded through the whole graph:

| field | set by | description |
|---|---|---|
| `question`, `mode` | caller (`main.py`) | the input |
| `context_chunks` | retrieval node | list of `{title, url, text, score}` |
| `retrieval_backend` | retrieval node | `"bm25"` \| `"tavily"` \| `"ddgs"` \| `"none"` |
| `retrieval_warning` | retrieval node | non-fatal issue to surface to the user |
| `answer`, `sources` | generation node | final output |

**Nodes** (`langgraph_helper/nodes.py`) are plain functions that take the
state and return a partial-state update — LangGraph merges dict updates
into state automatically, so nodes stay small and independently testable
(see `tests/test_graph_smoke.py`, which exercises `retrieve_offline`
directly with no LLM/network involved).

- `route_by_mode` — a conditional-edge function; picks the retrieval branch
  from `state["mode"]`. Mode lives in the *state*, not in the graph
  structure, so one compiled graph serves both modes (and would extend
  cleanly to e.g. a "hybrid" mode later without rebuilding the graph).
- `retrieve_offline` — BM25 lexical search over a locally pre-built index
  (see below). Never touches the network.
- `retrieve_online` — live web search (Tavily or DuckDuckGo). Falls back to
  the offline index if the search call fails for any reason, so a flaky
  network degrades gracefully instead of crashing.
- `generate_answer` — builds a grounded prompt from the retrieved passages
  and calls the configured LLM (`langgraph_helper/llm.py`).

The graph is compiled once per process (`graph.py:get_compiled_graph`) and
invoked per question from `main.py`, which is a thin CLI wrapper (argument
parsing, mode/provider resolution, and a REPL loop when no question is
given on the command line).

---

## Operating modes

### Offline mode

`--mode offline` (default). Retrieval is 100% local: a BM25 index built
ahead of time from LangChain/LangGraph's official `llms.txt` /
`llms-full.txt` documentation dumps. No web requests are made during
retrieval — only the LLM call itself needs internet (see
[Choosing an LLM provider](#choosing-an-llm-provider) for a fully local,
zero-network option via Ollama).

### Online mode

`--mode online`. Retrieval is a live web search biased towards the
official LangChain/LangGraph docs and GitHub, used to answer questions
about things not in the static snapshot (e.g. "what's new in the latest
release"). See
[Choosing a web search backend](#choosing-a-web-search-backend-online-mode).

### Switching modes

```bash
# CLI flag (highest priority)
python main.py --mode offline "How do I use checkpointers?"
python main.py --mode online  "What are the latest LangGraph features?"

# Environment variable (used when --mode is omitted)
export AGENT_MODE=online
python main.py "What are the latest LangGraph features?"
```

---

## Data freshness strategy

### Offline: how the data was prepared

1. **`data_prep/download_docs.py`** fetches the raw documentation. This is
   the *only* step in the pipeline that needs internet access.

   The assignment names `https://langchain-ai.github.io/langgraph/llms.txt`
   / `llms-full.txt` as the LangGraph sources. As of this writing those
   URLs redirect to a stub page: **LangGraph's docs have been merged into
   `docs.langchain.com`**, alongside LangChain's, both under `/oss/python/`.
   The script fetches the legacy URLs anyway (for the record) and
   additionally fetches `https://docs.langchain.com/oss/python/llms-full.txt`,
   which is the actual current full-text corpus — confirmed by grepping it
   for `Source: .../oss/python/langgraph/...` entries (43 LangGraph pages
   + 76 LangChain-core pages + supporting sections, 538 pages / ~6.8 MB
   total). **That file is the one actually indexed.**

2. **`data_prep/build_index.py`** parses the raw dump (each page begins
   with a `# Title` line followed by `Source: <url>`), splits it into
   ~1200-character paragraph-aware chunks with a 150-character overlap,
   and writes `data/index/chunks.jsonl` (7,007 chunks from 538 pages).
   Fully offline — no network, no ML model download.

3. At **query time**, `langgraph_helper/retrievers/offline.py` loads
   `chunks.jsonl` and scores it with **BM25** (`rank-bm25`, pure Python).
   Index build is ~0.8 s on process start; queries after that are
   sub-10 ms. See [Design notes](#design-notes--tradeoffs) for why BM25
   over embeddings.

The prebuilt index (`data/index/chunks.jsonl`) and the raw dumps
(`data/raw/*.txt`) are committed to this repo so the offline mode works
immediately after `pip install`, with no prep step required. To refresh
the snapshot later:

```bash
python data_prep/download_docs.py --force
python data_prep/build_index.py
```

Re-run this whenever LangGraph/LangChain ship a new release you want
reflected offline. There's no automatic staleness detection — it's a
point-in-time snapshot by design, which is exactly what "offline" implies;
`--mode online` is the path for anything that must be current in real time.

### Online: what services are used and why

| Service | Role | Free tier | API key |
|---|---|---|---|
| **DuckDuckGo (`ddgs`)** | default web search backend | Unlimited, no signup | none needed |
| **Tavily** | optional higher-quality search backend, used automatically if configured | 1,000 searches/month | [app.tavily.com](https://app.tavily.com) |

DuckDuckGo is the default specifically so the online mode works out of the
box with zero configuration. If `TAVILY_API_KEY` is set, the agent prefers
Tavily (it returns cleaner, LLM-oriented content snippets); if the Tavily
call fails for any reason it falls back to DuckDuckGo automatically.

---

## Setup

**Requirements:** Python 3.11–3.12 (this was built and tested on 3.12;
LangGraph/LangChain's dependency chain does not yet have wheels for very
new Python releases like 3.14 on all platforms).

```bash
git clone git@github.com:tosspro23-cell/Langgraph-Helper-Agent-V2.git
cd Langgraph-Helper-Agent-V2

python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set GOOGLE_API_KEY (free at https://aistudio.google.com/app/apikey)
```

That's it — the offline index ships in the repo, so you can run a query
immediately:

```bash
python main.py --mode offline "How do I add persistence to a LangGraph agent?"
```

If you'd rather rebuild the offline index yourself (e.g. to pick up newer
docs), see [Data freshness strategy](#data-freshness-strategy).

---

## Usage

```bash
# single question
python main.py --mode offline "What's the difference between StateGraph and MessageGraph?"

# interactive REPL (omit the question)
python main.py --mode offline

# see which sources were retrieved
python main.py --mode online --verbose "What are best practices for state management in LangGraph?"

# override the LLM provider for one run
python main.py --provider ollama --mode offline "Show me how to implement human-in-the-loop with LangGraph"
```

More sample questions: [`examples/example_questions.md`](examples/example_questions.md).

---

## Choosing an LLM provider

Set `LLM_PROVIDER` in `.env` (or pass `--provider`). All three speak the
same LangChain `BaseChatModel` interface, so nothing else in the agent
changes.

| Provider | `LLM_PROVIDER` | Needs internet | API key |
|---|---|---|---|
| **Google Gemini** (default) | `gemini` | yes | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) — free tier, see [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) |
| **OpenRouter** | `openrouter` | yes | [openrouter.ai/keys](https://openrouter.ai/keys) — many models have a free tier, e.g. `google/gemini-2.0-flash-exp:free` |
| **Ollama** | `ollama` | **no** | none — install from [ollama.com](https://ollama.com), then `ollama pull llama3.1` |

Gemini and OpenRouter are hosted APIs, so they need internet even in
`--mode offline` (the assignment explicitly allows this: "offline" governs
document retrieval, not the LLM call). For a setup with **zero** network
dependency end-to-end, use `LLM_PROVIDER=ollama` with `--mode offline`.

### A note on Gemini free-tier rate limits

`GEMINI_MODEL` defaults to `gemini-3.5-flash-lite`. "-lite" models get the
most generous free-tier request quota; brand-new/preview models (including
`-latest` aliases, which can silently start pointing at a preview model)
have much stricter limits — as low as **20 requests/day** was observed on
a preview flash model during development. If you see a `429
RESOURCE_EXHAUSTED` error, wait for the quota to reset (daily quotas reset
at midnight Pacific time) or switch to another model/provider — no code
changes needed, just edit `GEMINI_MODEL` (or `LLM_PROVIDER`) in `.env`.
This is expected behavior on a free tier, not a bug; the agent surfaces
the error message and exits cleanly rather than hanging or crashing.

---

## Choosing a web search backend (online mode)

No configuration needed — DuckDuckGo (`ddgs`) is used by default and
requires no API key. To use Tavily instead, set `TAVILY_API_KEY` in `.env`
(free tier: 1,000 searches/month, sign up at
[app.tavily.com](https://app.tavily.com)).

---

## Tests

```bash
python -m pytest tests/ -v
```

All tests run offline with no API key required — they exercise the BM25
retriever and the graph's routing/compilation directly, without calling
any LLM.

---

## Project layout

```
main.py                        CLI entry point (argparse, mode/provider resolution, REPL)
langgraph_helper/
  config.py                    env-driven Settings
  llm.py                       provider factory (gemini | openrouter | ollama)
  state.py                     AgentState TypedDict
  prompts.py                   system/user prompt templates
  nodes.py                     graph node functions
  graph.py                     StateGraph assembly
  retrievers/
    offline.py                 BM25 over the local chunk index
    online.py                  DuckDuckGo / Tavily web search
data_prep/
  download_docs.py             step 1: fetch raw llms.txt / llms-full.txt (only step needing internet)
  build_index.py                step 2: chunk raw docs -> data/index/chunks.jsonl
data/
  raw/                         downloaded documentation dumps
  index/chunks.jsonl           prebuilt retrieval index (committed, ready to use)
tests/                         offline, no-API-key-required tests
examples/example_questions.md  sample prompts for both modes
```

---

## Design notes / tradeoffs

- **BM25 over embeddings for offline retrieval.** No ML model download or
  inference needed — `rank-bm25` is pure Python and installs in seconds,
  which matters a lot for "runnable on another machine" portability. For
  API-reference-heavy docs full of exact identifiers (`StateGraph`,
  `add_conditional_edges`, `PostgresSaver`, ...), lexical matching is also
  a strong, predictable baseline. The tradeoff is weaker matching for
  purely conceptual/paraphrased questions than a semantic/embedding index
  would give — a reasonable place to extend this project if needed (e.g.
  swap in `langchain-google-genai`'s embedding model, still queryable
  without extra heavy local dependencies).
- **Mode as state, not graph topology.** `route_by_mode` is a conditional
  edge off a single `START`, so one compiled graph handles both modes.
  This keeps `main.py` and any future callers simple (one graph object)
  and makes adding a third mode (e.g. "hybrid": offline + online merged) a
  matter of adding one more branch, not restructuring the graph. The graph
  is a simple branch-then-join with no cycles, since this task doesn't
  need iteration (e.g. no repeated tool calls or self-correction loops) —
  a natural extension point if that changes.
- **Online mode never hard-fails on search errors.** `retrieve_online`
  catches exceptions from the search backend and falls back to the local
  BM25 index (surfacing the failure via `retrieval_warning` instead of
  crashing the CLI) so a flaky network doesn't lose the whole answer.
- **Chunking is paragraph-aware, not fixed-width.** Splitting the raw
  corpus on blank lines before enforcing the ~1200-char budget avoids
  slicing code blocks or bullet lists mid-line, which keeps individual
  BM25-retrieved passages coherent enough to hand an LLM directly.
