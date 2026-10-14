import os
from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BACKEND_DIR / "storage"
IMAGES_DIR = STORAGE_DIR / "images"
ORIGINAL_IMAGES_DIR = IMAGES_DIR / "original"
CROPPED_IMAGES_DIR = IMAGES_DIR / "crops"
DATASETS_DIR = STORAGE_DIR / "datasets"
MODELS_DIR = STORAGE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
EXPORTS_DIR = STORAGE_DIR / "exports"

# Create storage directories if they do not exist
for directory in [
    STORAGE_DIR,
    IMAGES_DIR,
    ORIGINAL_IMAGES_DIR,
    CROPPED_IMAGES_DIR,
    DATASETS_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    EXPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
except ImportError:
    # Graceful zero-dependency fallback parser if python-dotenv is not installed
    _env_file = BACKEND_DIR / ".env"
    if _env_file.exists():
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip('"').strip("'")
                    if _k not in os.environ:
                        os.environ[_k] = _v

from sqlalchemy.engine import URL

# PostgreSQL Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5433"))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "yogya-c50")

# Construct PostgreSQL connection URL via sqlalchemy.engine.URL.create()
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=DB_USER,
    password=DB_PASSWORD if DB_PASSWORD else None,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

# Recognition Defaults
DEFAULT_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv("DEFAULT_DETECTION_CONFIDENCE_THRESHOLD", "0.35"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("DEFAULT_SIMILARITY_THRESHOLD", "0.60"))
DEFAULT_MARGIN_THRESHOLD = float(os.getenv("DEFAULT_MARGIN_THRESHOLD", "0.08"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))

# Baseline Model Names
DETECTOR_MODEL_NAME = "fasterrcnn_mobilenet_v3_large_fpn"
EMBEDDING_MODEL_NAME = "dinov3-vits16-baseline"
EMBEDDING_DIM = 384  # DINOv3 ViT-S/16 feature dimension

# DINO Configuration
_checkpoint_env = os.getenv("DINOV3_CHECKPOINT_PATH")
if _checkpoint_env:
    _p = Path(_checkpoint_env)
    DINOV3_CHECKPOINT_PATH = str(_p if _p.is_absolute() else (BACKEND_DIR / _p).resolve())
else:
    _default_ckpt = CHECKPOINTS_DIR / "dinov3_vits16.pth"
    DINOV3_CHECKPOINT_PATH = str(_default_ckpt.resolve()) if _default_ckpt.exists() else None

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
