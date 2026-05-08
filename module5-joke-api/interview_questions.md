🎯 Interview angle — why build the graph in lifespan and not at module level?


python# Option 1 — module level (runs when Python imports the file)
graph = build_graph()  # runs at import time

# Option 2 — lifespan (runs when FastAPI starts)
async def lifespan(app):
    graph = build_graph()  # runs at startup
Module level sounds simpler but has a problem — it runs during testing too. When pytest imports api.py to test it, build_graph() runs immediately, which calls load_dotenv(), which needs your .env file. In CI/CD environments that file might not exist yet. Lifespan gives you control — the graph only builds when the actual server starts, not during imports.

