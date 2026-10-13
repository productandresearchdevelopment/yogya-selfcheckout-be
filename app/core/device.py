import os
import platform
from typing import Dict, Any

_device = None


def get_torch_device():
    """
    Dynamically inspect available hardware acceleration:
    1. CUDA if available and functional
    2. Apple MPS if available and functional
    3. CPU fallback
    """
    global _device
    if _device is not None:
        return _device

    try:
        import torch
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _device = torch.device("mps")
        else:
            _device = torch.device("cpu")
    except ImportError:
        _device = "cpu"

    return _device


def get_system_environment_info() -> Dict[str, Any]:
    """
    Returns environmental diagnostics for the Settings and Dashboard pages.
    """
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "torch_available": False,
        "torch_version": None,
        "active_device": "cpu",
        "cuda_available": False,
        "cuda_device_name": None,
        "mps_available": False,
        "cpu_count": os.cpu_count() or 1,
    }

    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["active_device"] = f"cuda (0: {info['cuda_device_name']})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
            info["active_device"] = "mps (Apple Silicon)"
        else:
            info["active_device"] = "cpu"
    except ImportError:
        info["active_device"] = "cpu (PyTorch loading or not installed)"

    return info
