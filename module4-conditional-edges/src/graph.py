import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,START,END


env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class JokeState(TypedDict):
    topic: str
    joke: str
    rating: str
    score: int
    attempts: int


# ─────────────────────────────────────────────
# THE LLM
# ─────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", 
                 temperature=0.7,
                 openai_api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# NODE 1 — generate_joke
# ─────────────────────────────────────────────

def generate_joke(state: JokeState):
    """
    Generates a joke about the topic.
    Increments attempts counter every time it runs.
    This is the node we loop back to on retry.
    """
    topic = state["topic"]

    #Increment attempts FIRST — before calling LLM
    # So even if the LLM call fails we've counted this attempt

    attempts = state["attempts"] + 1

    print(f"\n[Attempt {attempts}] Generating joke about '{topic}'...")

    response = llm.invoke([
        HumanMessage(content= f"Tell me a short, clever joke about {topic}. "
            f"Just the joke — no intro, no explanation. "
            f"Make it genuinely funny and witty.")
    ])

    return {"joke":response.content.strip(), "attempts": attempts,"score": 0}
    # Reset score to 0 on each new joke
    # so stale scores don't confuse the router


# ─────────────────────────────────────────────
# NODE 2 — rate_joke
# ─────────────────────────────────────────────

def rate_joke(state: JokeState):
    """
    Rates the joke on a scale of 1-10.
    Extracts BOTH a human-readable rating AND
    a clean integer score for the router to use.
    """
    joke = state["joke"]

    print(f"Rating the joke: '{joke[:50]}'")

    response = llm.invoke([
        HumanMessage(content= ( f"Rate this joke on a scale of 1-10 for funniness.\n\n"
            f"Joke: {joke}\n\n"
            f"Respond in EXACTLY this format and nothing else:\n"
            f"SCORE: <number>\n"
            f"REASON: <one sentence>"))
    ])

    raw = response.content.strip()

    # Parse the structured response
    # Expected format:
    # SCORE: 7
    # REASON: Clever wordplay that most people will get

    score = 5 # safe default if parsing fails
    rating = raw

    for line in raw.splitlines("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(line.replace("SCORE: ", "").strip())
            except ValueError:
                score = 5 # keep default score if parsing fails

    print(f"Score: {score}/10")

    return {"rating": rating,
            "score": score}

# ─────────────────────────────────────────────
# THE ROUTER — the brain of conditional edges
# ─────────────────────────────────────────────

def router(state: JokeState)-> str:
    """
    Reads state and decides what runs next.
    Returns a STRING — the name of the next node.

    This is the ONLY job of a router function.
    It never modifies state. It only reads and decides.
    """
    score = state["score"]
    attempts = state["attempts"]

    if attempts >= 3:
        # Safety net — accept whatever we have
        # Never let an agent loop forever
        print(f"\n[Max attempts reached] Accepting joke with score {score}/10")
        return "end"
    
    if score >=7:
       # Good enough — stop here
        print(f"\n[Great joke! Score {score}/10] Done after {attempts} attempt(s)")
        return "end"
    
    # Not good enough and still have attempts left — retry
    print(f"\n[Score {score}/10 not good enough] Retrying...")
    return "generate_joke"

# ─────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph():
    
    graph_builder = StateGraph(JokeState)

    # Register nodes
    graph_builder.add_node("generate_joke", generate_joke)
    graph_builder.add_node("rate_joke", rate_joke)

    # Entry point — same as before
    graph_builder.add_edge(START, "generate_joke")  
    graph_builder.add_edge("generate_joke", "rate_joke")  # After generating, always rate
    graph_builder.add_conditional_edges("rate_joke", 
                  router,{
            "generate_joke": "generate_joke",  # if router returns "generate_joke"
            "end": END                          # if router returns "end"
        })  # After rating, decide what
    
    graph = graph_builder.compile()
    return graph

"""
START
  │
  ▼
generate_joke ──────────────────────────────┐
  │                                         │ (loop back)
  ▼ (normal edge — always)                  │
rate_joke                                   │
  │                                         │
  ▼ (conditional edge — router decides)     │
router(state)                               │
  ├── score >= 7  → "end"   → END ✅        │
  ├── attempts >= 3 → "end" → END ✅        │
  └── score < 7   → "generate_joke" ────────┘

"""


# ─────────────────────────────────────────────
# MAIN — run the graph
# ─────────────────────────────────────────────

def main():
    print("=" * 50)
    print("   Picky Joke Bot — LangGraph Conditional Edges")
    print("   Won't stop until it finds a funny joke!")
    print("=" * 50)

    graph = build_graph()

    while True:
        topic = input("\nEnter a topic (or 'exit'): ").strip()
        
        if topic.lower() == "exit":
            print("Goodbye!")
            break

        if not topic:
            print("Please enter a valid topic.")
            continue

        # Pass initial state
        # attempts starts at 0 — generate_joke increments it
        # score starts at 0 — rate_joke fills it

        final_state = graph.invoke({ "topic": topic,
            "joke": "",
            "rating": "",
            "score": 0,
            "attempts": 0
        })

        print("\n" + "=" * 50)
        print(f"Final Joke: {final_state['joke']}")
        print(f"Rating: {final_state['rating']}")
        print(f"Score: {final_state['score']}/10")
        print(f"Total Attempts: {final_state['attempts']}")
        print("=" * 50)

if __name__ == "__main__":
    main()

        




    
       

