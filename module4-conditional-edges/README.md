# Module 4 — Picky Joke Bot 🤖

A LangGraph agent that won't accept a bad joke.
It generates, rates, and retries until it finds
something genuinely funny — or hits 3 attempts.

Part of the [LangChain Mastery](../README.md) portfolio.

---

## What it does

Enter a topic → programming
[Attempt 1] Generating... Score: 6/10 → not good enough
[Attempt 2] Generating... Score: 9/10 →
FINAL JOKE:   A SQL query walks into a bar...
FINAL RATING: 9/10 — classic CS humour
ATTEMPTS:     2


---

## The key concept — conditional edges

Normal edges always go A → B.
Conditional edges ask "where should we go?" at runtime.

START → generate_joke → rate_joke → router()
│
score >= 7 → END ✅
attempts >= 3 → END ✅
score < 7 → generate_joke 🔄

The router is a plain Python function that reads
state and returns a string — the name of the next node.

---

## New concepts vs Module 3

| Concept | What it does |
|---------|-------------|
| `add_conditional_edges` | Wires a router function after a node |
| Router function | Reads state · returns next node name as string |
| `attempts` counter | Circuit breaker — prevents infinite loops |
| `score` field | Machine-readable number for router decisions |
| Immutable state | Never mutate state directly — always return new values |

---

## Architecture

```python
class JokeState(TypedDict):
    topic: str      # user input
    joke: str       # generated joke
    rating: str     # human-readable rating
    score: int      # machine-readable score for router
    attempts: int   # loop counter — safety net
```

---

## Key learning — the router function rules

1. Returns a STRING — the name of next node
2. NEVER modifies state — only reads
3. Every return value must exist in the edges map
4. Always handle the max attempts case — never infinite loops

---

## Tech

- LangGraph 1.0 · LangChain · OpenAI GPT-4o-mini
- Python 3.11+ · uv · LangSmith

## Run it

```bash
cd module4-conditional-edges
uv sync
uv run python src/graph.py
```