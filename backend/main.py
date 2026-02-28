"""
CORTEX Backend API

FastAPI application for chat memory visualization and retrieval.
Provides endpoints for:
- Chat ingestion (HTML upload)
- Hybrid search (semantic + keyword)
- Conversation retrieval
- 3D visualization data
"""

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import database and models
from backend.database import init_db, engine
from backend.schemas import HealthResponse
from backend.services.provider import get_embedding_provider, get_chat_provider

# Import routers
from backend.api.ingest import router as ingest_router
from backend.api.chats import router as chats_router
from backend.api.search import router as search_router
from backend.api.prompt import router as prompt_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    print("Starting CORTEX backend")

    # Initialize database
    try:
        init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

    # Log active providers
    emb_provider = get_embedding_provider()
    chat_provider = get_chat_provider()
    print(f"[CONFIG] Embedding provider: {emb_provider}")
    print(f"[CONFIG] Chat provider: {chat_provider}")

    # Validate API keys for cloud providers
    if emb_provider == "huggingface":
        if os.getenv("HF_API_TOKEN"):
            print("[OK] HuggingFace API token is set")
        else:
            print("[WARNING] EMBEDDING_PROVIDER=huggingface but HF_API_TOKEN is not set!")

    if chat_provider == "groq":
        if os.getenv("GROQ_API_KEY"):
            print("[OK] Groq API key is set")
        else:
            print("[WARNING] CHAT_PROVIDER=groq but GROQ_API_KEY is not set!")

    # Check Ollama if either provider uses it
    if emb_provider == "ollama" or chat_provider == "ollama":
        try:
            import httpx
            resp = httpx.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags", timeout=3)
            model_names = [m["name"] for m in resp.json().get("models", [])]
            print(f"[OK] Ollama connected - models: {', '.join(model_names)}")
        except Exception:
            print("[WARNING] Ollama not reachable at localhost:11434 (local provider will fail)")

    print("CORTEX backend ready!")

    yield

    # Shutdown
    print("Shutting down CORTEX backend")
    engine.dispose()
    print("Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="CORTEX API",
    description="AI Chat Memory Visualization & Retrieval System",
    version="1.0.0",
    lifespan=lifespan
)


# Configure CORS — allow all origins in development
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors with JSON response."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors with JSON response."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.
    Returns system status and configuration info.
    """
    emb_provider = get_embedding_provider()
    chat_prov = get_chat_provider()

    # Check database connection
    database_connected = False
    try:
        with engine.connect():
            database_connected = True
    except Exception:
        pass

    # Check embedding provider readiness
    embedding_ready = False
    ollama_connected = False
    if emb_provider == "huggingface":
        embedding_ready = bool(os.getenv("HF_API_TOKEN"))
    else:
        # Ollama — check connectivity
        try:
            import httpx
            resp = httpx.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags", timeout=2)
            ollama_connected = resp.status_code == 200
            embedding_ready = ollama_connected
        except Exception:
            pass

    # Check chat provider readiness
    chat_ready = False
    if chat_prov == "groq":
        chat_ready = bool(os.getenv("GROQ_API_KEY"))
    else:
        # Ollama — reuse check or do fresh
        if ollama_connected:
            chat_ready = True
        else:
            try:
                import httpx
                resp = httpx.get(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags", timeout=2)
                ollama_connected = resp.status_code == 200
                chat_ready = ollama_connected
            except Exception:
                pass

    # Check vector store
    vector_store_ready = False
    try:
        from backend.services.vector_store import get_vector_store_service
        svc = get_vector_store_service()
        vector_store_ready = svc.count() >= 0
    except Exception:
        pass

    is_healthy = database_connected and embedding_ready and chat_ready

    return HealthResponse(
        status="healthy" if is_healthy else "degraded",
        version="1.0.0",
        database_connected=database_connected,
        ollama_connected=ollama_connected,
        chroma_ready=vector_store_ready,
        embedding_provider=emb_provider,
        chat_provider=chat_prov,
        embedding_ready=embedding_ready,
        chat_ready=chat_ready,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CORTEX API",
        "version": "1.0.0",
        "description": "AI Chat Memory Visualization & Retrieval System",
        "docs": "/docs",
        "health": "/health"
    }


# Include routers
app.include_router(ingest_router)
app.include_router(chats_router)
app.include_router(search_router)
app.include_router(prompt_router)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
