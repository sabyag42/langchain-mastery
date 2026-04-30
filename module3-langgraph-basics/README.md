# LangGraph Joke Bot 🤖

My first LangGraph project — a two-node stateful graph that
generates and rates jokes about any topic.

Built as part of my 30-day LangGraph mastery sprint.

## What it does

Enter a topic → cats
Joke:   Why was the cat sitting on the computer?
Because it wanted to keep an eye on the mouse!
Rating: 7/10 — clever double meaning, works for any audience


## How it works

Two nodes. One shared state. Zero direct node-to-node communication.

START → generate_joke → rate_joke → END

- **generate_joke** reads the topic from state, calls the LLM,
  writes the joke back to state
- **rate_joke** reads the joke from state, calls the LLM,
  writes the rating back to state
- Nodes never talk to each other directly — State is the only
  communication channel

## Key concepts

- `StateGraph` — the graph container
- `TypedDict` — typed shared state (the whiteboard)
- Nodes — plain Python functions
- Normal edges — unconditional wiring
- LangSmith — automatic tracing of every node

## Run it

```bash
cd module3-langgraph-basics
uv sync
uv run python src/graph.py
```

## Tech

- LangGraph 1.0 · LangChain · OpenAI GPT-4o-mini
- Python 3.11+ · uv · LangSmith
