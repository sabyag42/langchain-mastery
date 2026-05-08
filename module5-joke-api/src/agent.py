import os
from typing import TypedDict
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END

# ─────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────
path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=path)


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class JokeState(TypedDict):
    topic: str
    joke: str
    rating: str
    score: int
    attempts: int

# ─────────────────────────────────────────────
# LLM — built once at module load time
# ─────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ─────────────────────────────────────────────
# NODE 1 — generate_joke
# ─────────────────────────────────────────────

def generate_joke(state: JokeState) -> dict:
    topic = state["topic"]
    attempts = state["attempts"] + 1

    print(f"[Attempt {attempts}] Generating joke about '{topic}'...")

    response = llm.invoke(
        [HumanMessage(content=(
           f"Tell me a short, clever joke about {topic}. "
            f"Just the joke — no intro, no explanation. "
            f"Make it genuinely funny and witty."
        ))
    ])

    return {
        "joke": response.content.strip(),
        "attempts": attempts,
        "score": 0,
    }

# ─────────────────────────────────────────────
# NODE 2 — rate_joke
# ─────────────────────────────────────────────
def rate_joke(state: JokeState) -> dict:
    joke = state["joke"]

    print(f"[Rating joke...] '{joke[:50]}'")

    response = llm.invoke([
        HumanMessage(content=(
            f"Rate this joke on a scale of 1-10 for funniness.\n\n"
            f"Joke: {joke}\n\n"
            f"Respond in EXACTLY this format and nothing else:\n"
            f"SCORE: <number>\n"
            f"REASON: <one sentence>"
        ))
    ])

    raw = response.content.strip()
    score = 5

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 5

    print(f"[Score: {score}/10]")

    return {
        "rating": raw,
        "score": score,
        "attempts": state["attempts"]  # explicitly pass through
    }


# ─────────────────────────────────────────────
# ROUTER — conditional edge brain
# ─────────────────────────────────────────────
def router(state: JokeState) -> str:
    score = state["score"]
    attempts = state["attempts"]

    if attempts >= 3:
        print(f"[Max attempts reached] Accepting score {score}/10")
        return "end"

    if score >= 7:
        print(f"[Great joke! {score}/10] Done in {attempts} attempt(s)")
        return "end"

    print(f"[Score {score}/10 not good enough] Retrying...")
    return "generate_joke"


# ─────────────────────────────────────────────
# BUILD GRAPH — called once at API startup
# ─────────────────────────────────────────────
def build_graph():
    """
    Builds and compiles the LangGraph.
    Called ONCE when FastAPI starts.
    Reused for every incoming request.
    """
    graph_builder = StateGraph(JokeState)

    graph_builder.add_node("generate_joke", generate_joke)
    graph_builder.add_node("rate_joke", rate_joke)

    graph_builder.add_edge(START, "generate_joke")
    graph_builder.add_edge("generate_joke", "rate_joke")

    graph_builder.add_conditional_edges(
        "rate_joke",
        router,
        {
            "generate_joke": "generate_joke",
            "end": END
        }
    )

    return graph_builder.compile()






