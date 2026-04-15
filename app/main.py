"""
main.py — FastAPI application entry point.

HOW TO RUN:
  uvicorn app.main:app --reload --port 8000

SWAGGER UI (interactive docs):
  http://localhost:8000/docs

REDOC (clean docs):
  http://localhost:8000/redoc

HOW FASTAPI WORKS (for viva):
  FastAPI is an ASGI web framework. It uses Python type hints
  and Pydantic to auto-validate requests and auto-generate
  OpenAPI documentation. It's 2-3x faster than Flask because
  it's built on Starlette and supports async natively.
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path (so all imports work regardless of cwd)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.routers import stylist
from config import settings

# ---------------------------------------------------------------
# Configure logging for the entire application
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Lifespan: code that runs on startup / shutdown
# ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    - On startup: verify ChromaDB is accessible
    - On shutdown: clean up resources
    """
    # --- STARTUP ---
    logger.info("=" * 55)
    logger.info("  👗 AI Fashion Stylist API — Starting Up")
    logger.info("=" * 55)
    logger.info(f"  Vision model : {settings.gemini_vision_model}")
    logger.info(f"  LLM model    : {settings.gemini_llm_model}")
    logger.info(f"  Embed model  : {settings.gemini_embedding_model}")
    logger.info(f"  ChromaDB path: {settings.chroma_db_path}")

    # Check if ChromaDB knowledge base has been built
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_collection(name=settings.chroma_collection_name)
        count = collection.count()
        logger.info(f"  ChromaDB     : ✅ Connected ({count} documents)")
    except Exception as e:
        logger.warning(f"  ChromaDB     : ⚠️  Not ready — {e}")
        logger.warning("  Run: python knowledge_base/builder.py")

    logger.info("=" * 55)
    logger.info("  API docs: http://localhost:8000/docs")
    logger.info("=" * 55)

    yield   # Application runs here

    # --- SHUTDOWN ---
    logger.info("AI Fashion Stylist API shutting down...")


# ---------------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------------
app = FastAPI(
    title="AI-Powered Fashion Stylist API",
    description="""
## 👗 AI Fashion Stylist — RAG-Powered Recommendation Engine

**B.Tech Final Year Project** | AI/ML Developer: Sparsh

### How it works
1. Upload a photo → Gemini Vision extracts face shape, skin tone, body type
2. ChromaDB retrieves relevant fashion rules (RAG)
3. Gemini LLM generates a personalized recommendation

### Quick Start
1. Run `python knowledge_base/builder.py` to build the knowledge base
2. Start the server: `uvicorn app.main:app --reload`
3. Test at `/docs`
""",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------
# CORS Middleware
# Allows your frontend team to call this API from a browser
# ---------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production: restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------
app.include_router(
    stylist.router,
    prefix="/api/v1/stylist",
    tags=["AI Stylist"],
)


# ---------------------------------------------------------------
# Root endpoint — quick check the server is alive
# ---------------------------------------------------------------
@app.get("/", tags=["System"])
async def root():
    return {
        "message": "AI Fashion Stylist API is running! 👗",
        "docs": "/docs",
        "health": "/api/v1/stylist/health",
        "version": "1.0.0",
    }


# ---------------------------------------------------------------
# Run directly: python app/main.py
# (For development — prefer: uvicorn app.main:app --reload)
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
