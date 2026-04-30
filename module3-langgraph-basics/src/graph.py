
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class JokeState(TypedDict):
    topic: str
    joke: str
    rating: str


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


def generate_joke(state: JokeState) -> dict:
    topic = state["topic"]
    response = llm.invoke([
        HumanMessage(content=(
            f"Tell me a short, clever joke about {topic}. "
            f"Just the joke — no intro, no explanation."
        ))
    ])
    return {"joke": response.content}


def rate_joke(state: JokeState) -> dict:
    joke = state["joke"]
    response = llm.invoke([
        HumanMessage(content=(
            f"Rate this joke on a scale of 1-10 for funniness. "
            f"Give the score and ONE sentence explaining why.\n\n"
            f"Joke: {joke}"
        ))
    ])
    return {"rating": response.content}


def build_graph():
    graph_builder = StateGraph(JokeState)

    graph_builder.add_node("generate_joke", generate_joke)
    graph_builder.add_node("rate_joke", rate_joke)

    graph_builder.add_edge(START, "generate_joke")
    graph_builder.add_edge("generate_joke", "rate_joke")
    graph_builder.add_edge("rate_joke", END)

    graph = graph_builder.compile()
    return graph


def main():
    print("=" * 50)
    print("   LangGraph Joke Bot")
    print("=" * 50)

    graph = build_graph()

    while True:
        topic = input("\nEnter a topic for a joke (or 'exit'): ").strip()

        if topic.lower() == "exit":
            print("Bye!")
            break

        if not topic:
            continue

        print("\nRunning graph...\n")

        final_state = graph.invoke({"topic": topic})

        print(f"Joke:   {final_state['joke']}")
        print(f"Rating: {final_state['rating']}")
        print("-" * 50)


if __name__ == "__main__":
    main()



# EXPLANATIONS:-


# import os
# from pathlib import Path
# from typing import TypedDict

# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI


# env_path = Path(__file__).parent.parent / ".env"
# load_dotenv(dotenv_path=env_path)

# # ─────────────────────────────────────────────
# # THE STATE — the shared whiteboard
# # ─────────────────────────────────────────────

# class Jokestate(TypedDict):
#     """The shared state of the joke-telling process."""

#     topic: str
#     """The topic of the current joke."""
#     joke: str
#     """The current joke being worked on."""
#     rating: str
#     """Feedback on the current joke."""

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7,
#                     openai_api_key=os.getenv("OPENAI_API_KEY"))

# # ─────────────────────────────────────────────
# # NODE 1 — generate_joke
# # ─────────────────────────────────────────────
    
# def generate_joke(state: Jokestate) :
#     """
#     Reads topic from state.
#     Calls the LLM to generate a joke.
#     Returns partial state update with the joke.
#     """
#     topic = state["topic"]

#     response = llm.invoke([
#         HumanMessage(content=f"Write a joke about {topic}.")])
    
#     joke = response.content.strip()

#     return {"joke": joke}


# # ─────────────────────────────────────────────
# # NODE 2 — rate_joke
# # ─────────────────────────────────────────────

# def rate_joke(state: Jokestate) :
#     """
#     Reads joke from state (written by Node 1).
#     Calls the LLM to rate how funny it is.
#     Returns partial state update with the rating.
#     """

#     joke = state["joke"]

#     response = llm.invoke([
#         HumanMessage(content=f"On a scale of 1 to 10, how funny is this joke: {joke}?")])
    
#     rating = response.content.strip()

#     return {"rating": rating}


# # ─────────────────────────────────────────────
# # BUILD THE GRAPH
# # ─────────────────────────────────────────────

# from langgraph.graph import StateGraph,START,END

# def build_graph():
#     # Step 1 — create a graph that uses JokeState
#     # as its shared whiteboard
#     graph_builder = StateGraph[Jokestate]
    
#     # Step 2 — register the nodes
#     # "generate_joke" is the name LangGraph uses internally
#     # generate_joke (no quotes) is the actual function

#     graph_builder.add_node("generate_joke", generate_joke)
#     graph_builder.add_node("rate_joke", rate_joke)

#      # Step 3 — wire the edges
#     # START → generate_joke (entry point)
#     graph_builder.add_edge(START, "generate_joke")

#     # generate_joke → rate_joke (normal edge, always)
#     graph_builder.add_edge("generate_joke", "rate_joke")

#      # rate_joke → END (graph stops here)
#     graph_builder.add_edge("rate_joke", END)

#      # Step 4 — compile
#     # This validates the graph structure and
#     # returns a runnable object

#     graph = graph_builder.compile()

#     return graph

# """
# START
#   │
#   ▼
# generate_joke ──reads──▶ state["topic"]
#               ◀─writes── {"joke": "..."}
#   │
#   ▼ (normal edge — always)
# rate_joke ─────reads──▶ state["joke"]
#            ◀───writes── {"rating": "..."}
#   │
#   ▼
# END

# """
# # ─────────────────────────────────────────────
# # MAIN — run the graph
# # ─────────────────────────────────────────────

# def main():
#     print("="*20)
#     print(" Langraph joke Bot")
#     print("="*20)

#     # Build and compile the graph once
#     graph = build_graph()

#     while True:
#         topic = input("\nEnter a topic for a joke (or 'exit'): ").strip()
#         if topic.lower() == "exit":
#             break

#         if not topic:
#             continue

#         print("\nRunning graph...\n")

#         # THIS is where the entire graph executes
#         # Pass initial state — only topic is known at start
#         # joke and rating are None until nodes fill them
#         final_state = graph.invoke({"topic": topic})

#         # final_state is the complete JokeState dict
#         # after ALL nodes have run and updated it

#         print(f"Joke:   {final_state['joke']}")
#         print(f"Rating: {final_state['rating']}")
#         print("-" * 50)

# if __name__ == "__main__":
#     main()


