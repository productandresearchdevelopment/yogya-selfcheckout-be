import sys
from pathlib import Path

# Auto-link project virtual environment site-packages so running with global Python (e.g. Laragon)
# seamlessly accesses PyTorch, NumPy, torchvision, and all project dependencies without errors.
_backend_dir = Path(__file__).resolve().parent.parent
_venv_site = _backend_dir / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))
