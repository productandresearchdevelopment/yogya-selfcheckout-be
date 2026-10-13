import sys
from pathlib import Path

# Ensure project virtual environment packages are accessible in all execution contexts
_backend_dir = Path(__file__).resolve().parent.parent
_venv_site = _backend_dir / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import STORAGE_DIR, IMAGES_DIR, MODELS_DIR, CHECKPOINTS_DIR
from app.core.database import Base, engine, SessionLocal
from app.core.logging_config import configure_logging
import app.models  # Registers all SQLAlchemy models
from app.api import api_router
from app.core.device import get_system_environment_info
from app.services.model_registry_service import ensure_baseline_models_registered, get_active_model

logger = configure_logging()

# Initialize database tables on startup
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation per Requirement 10
    logger.info("Initializing Product Recognition Studio Backend...")
    env_info = get_system_environment_info()

    # Validate storage permissions
    for d in [STORAGE_DIR, IMAGES_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
        if not os.access(d, os.W_OK):
            logger.error(f"Storage path {d} is not writable!")

    # Validate database connection and models
    db = SessionLocal()
    try:
        ensure_baseline_models_registered(db)
        active_emb = get_active_model(db, model_type="embedding")
        logger.info(f"Active Visual Backbone: {active_emb.model_id} ({active_emb.embedding_dim}-dim)")
    except Exception as e:
        logger.error(f"Startup model registration warning: {e}")
    finally:
        db.close()

    print("=" * 60)
    print(" Product Recognition Studio Backend (Production Mode)")
    print(f" Active Hardware Acceleration: {env_info['active_device']}")
    from app.core.config import DB_HOST, DB_PORT, DB_NAME
    print(f" PostgreSQL Database: {DB_NAME} on {DB_HOST}:{DB_PORT}")
    print("=" * 60)
    logger.info("Studio backend successfully online.")
    yield
    # Shutdown handling per Requirement 5
    logger.info("Shutting down Product Recognition Studio Backend cleanly...")


app = FastAPI(
    title="Product Recognition Studio API",
    description="Local-first Computer Vision Studio for SKU Registration, Image Dataset Management, Localization, and Baseline Recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount storage directory for static image serving
if IMAGES_DIR.exists():
    app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/")
def root():
    return {
        "studio": "Product Recognition Studio",
        "phase": "Section 1 - Studio Foundation",
        "status": "online",
        "docs_url": "/docs",
    }
