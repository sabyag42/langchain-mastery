from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from src.agent import build_graph



from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.agent import build_graph


class JokeRequest(BaseModel):
    topic: str = Field(
        min_length=1,
        max_length=100,
        description="The topic you want a joke about"
    )


class JokeResponse(BaseModel):
    joke: str
    rating: str
    score: int
    attempts: int


graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph
    print("[Startup] Building LangGraph...")
    graph = build_graph()
    print("[Startup] Graph ready")
    yield
    print("[Shutdown] Cleaning up...")


app = FastAPI(
    title="Joke Agent API",
    description=(
        "A LangGraph agent that generates and rates jokes. "
        "Won't stop until it finds something funny!"
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/joke", response_model=JokeResponse)
async def get_joke(request: JokeRequest):
    try:
        final_state = graph.invoke({
            "topic": request.topic,
            "joke": "",
            "rating": "",
            "score": 0,
            "attempts": 0
        })
        return JokeResponse(
            joke=final_state["joke"],
            rating=final_state["rating"],
            score=final_state["score"],
            attempts=final_state["attempts"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "joke-bot",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    return {
        "message": "Joke Agent API",
        "docs": "/docs",
        "health": "/health"
    }


# # ─────────────────────────────────────────────
# # REQUEST / RESPONSE MODELS
# # ─────────────────────────────────────────────

# class JokeRequest(BaseModel):
#     topic: str = Field(min_length=1,
#         max_length=100,
#         description="The topic you want a joke about")
    
# class JokeResponse(BaseModel):
#     joke: str
#     rating: str
#     score: int
#     attempts: int

# # ─────────────────────────────────────────────
# # LIFESPAN — runs on startup and shutdown
# # ─────────────────────────────────────────────
# # graph is stored here — shared across all requests


# # Interview angle — why Pydantic for request/response models?
# # Three reasons that matter in production:
# # Validation  → bad input rejected before it reaches your agent
# #               no "NoneType has no attribute" errors deep in your code

# # Documentation → FastAPI reads your Pydantic models and generates
# #                 Swagger UI automatically — zero extra effort

# # Type safety   → IDE knows exactly what shape request and response are
# #                 autocomplete works, typos get caught immediately

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # STARTUP — runs once when FastAPI starts
#     # Build the graph once, reuse for every request
#     global graph
#     print("[Startup] Building graph...")
#     graph = build_graph()

#     print("[Startup] Graph built successfully.")
    
#     yield # app runs here — handles all requests
    
#     # SHUTDOWN — runs once when FastAPI shuts down
#     print("[Shutdown] Cleaning up...")


# # ─────────────────────────────────────────────
# # FASTAPI APP
# # ─────────────────────────────────────────────

# app = FastAPI(title="Joke Generator API", description=(
#         "A LangGraph agent that generates and rates jokes. "
#         "Won't stop until it finds something funny!"
#     ),
#     version="1.0.0",
#     lifespan=lifespan
# )

# # ─────────────────────────────────────────────
# # ENDPOINT 1 — POST /joke
# # ─────────────────────────────────────────────

# @app.post("/joke", response_model=JokeResponse)
# async def generate_joke(request: JokeRequest):
    
#     """
#     Send a topic, get back a joke.
#     The agent retries until it scores 7 or above.
#     Maximum 3 attempts.
#     """

#     try:
#         final_state = graph.invoke({
#             "topic": request.topic,
#             "joke": "",
#             "rating": "",   
#             "score": 0,
#             "attempts": 0
#         })

#         return JokeResponse(
#             joke=final_state["joke"],
#             rating=final_state["rating"],
#             score=final_state["score"],
#             attempts=final_state["attempts"]
#         )


#     except Exception as e:
#         print(f"[Error] {e}")
#         raise HTTPException(status_code=500, detail="Agent Error")    
    
# @app.get("/health")
# async def health_check():
#     return {"status": "ok",
#             "agent": "joke-bot",
#             "version": "1.0.0"
#             }

# @app.get("/")
# async def root():
#     return {
#         "message": "Joke Agent API",
#         "docs": "/docs",
#         "health": "/health"
#     }

