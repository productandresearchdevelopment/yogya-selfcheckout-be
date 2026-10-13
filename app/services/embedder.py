from typing import Optional, List
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms

from app.core.device import get_torch_device

_baseline_model = None
_device = None
_baseline_transform = None

_dino_models = {}
_dino_transforms = {}


def get_embedder():
    """Lazy loader for baseline pretrained MobileNetV3 visual embedding feature extractor backbone."""
    global _baseline_model, _device, _baseline_transform
    if _baseline_model is not None:
        return _baseline_model, _device, _baseline_transform

    _device = get_torch_device()
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)

    # Replace final classification head with Identity to extract global pooled feature vector (960-dim)
    model.classifier = nn.Identity()
    model.to(_device)
    model.eval()

    # Preprocessing transform: Resize to 224x224, standard ImageNet normalization
    _baseline_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _baseline_model = model
    return _baseline_model, _device, _baseline_transform


def get_dino_embedder(variant: str = "dinov2_vits14"):
    """
    Lazy loader for Meta DINO Vision Transformer backbones.
    Supported variants:
      - 'dinov2_vits14': Vision Transformer Small (384-dim, 88MB, fast ~430ms CPU latency, high discriminative power)
      - 'dinov2_vitb14': Vision Transformer Base (768-dim, 346MB, ~1100ms CPU latency)
      - 'dinov3_vitb16': Meta DINOv3 ViT-B/16 (768-dim, requires local weights or gated HF access)
      - 'dinov3_vits16': Meta DINOv3 ViT-S/16 (384-dim, requires local weights or gated HF access)

    Preprocessing:
      Input PIL image -> RGB -> Resize((224, 224)) -> ToTensor() -> ImageNet Normalization
    Feature representation:
      Forward pass in eval mode without gradients, flattened to 1D vector, and L2 normalized.
    """
    global _dino_models, _device, _dino_transforms

    if _device is None:
        _device = get_torch_device()

    if variant in _dino_models:
        return _dino_models[variant], _device, _dino_transforms[variant]

    # Preprocessing expected by DINO Vision Transformers
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Check for DINOv3 variant request
    if "dinov3" in variant:
        import os
        checkpoint_env = os.environ.get("DINOV3_CHECKPOINT_PATH")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        if checkpoint_env and os.path.exists(checkpoint_env):
            # Load local DINOv3 checkpoint
            import sys
            import importlib
            hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")
            if os.path.exists(hub_dir) and hub_dir not in sys.path:
                sys.path.insert(0, hub_dir)
            try:
                backbones = importlib.import_module("dinov3.hub.backbones")
            except ImportError:
                raise RuntimeError(
                    f"DINOv3 repository code not found in {hub_dir}. "
                    "Ensure DINOv3 source repository is cloned to torch hub cache."
                )
            arch_func = getattr(backbones, variant, getattr(backbones, "dinov3_vitb16", None))
            if arch_func is None:
                raise ValueError(f"Unknown DINOv3 architecture variant: {variant}")
            model = arch_func(pretrained=False)
            state_dict = torch.load(checkpoint_env, map_location=_device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            model.load_state_dict(state_dict, strict=False)
        elif hf_token:
            # Load via Hugging Face with user token
            import importlib
            try:
                transformers_mod = importlib.import_module("transformers")
                auto_model = getattr(transformers_mod, "AutoModel")
            except ImportError:
                raise RuntimeError("Hugging Face transformers package is required. Install with `pip install transformers`.")
            hf_id = f"facebook/{variant.replace('_', '-')}-pretrain-lvd1689m"
            model = auto_model.from_pretrained(hf_id, token=hf_token)
        else:
            # Report the exact access restriction blocker as specified in Phase B
            raise RuntimeError(
                f"Cannot load official '{variant}' pretrained weights automatically: "
                "Meta DINOv3 weights are gated on Hugging Face (requiring license approval + HF_TOKEN) "
                "and Meta CDN returns HTTP 403 Forbidden. "
                "To use DINOv3, supply DINOV3_CHECKPOINT_PATH or HF_TOKEN. "
                "Alternatively, use verified Meta Vision Transformer 'dinov2_vits14' (384-dim) or 'dinov2_vitb14' (768-dim)."
            )
    else:
        # Load official open Meta DINO model via PyTorch Hub
        model = torch.hub.load("facebookresearch/dinov2", variant)

    model.to(_device)
    model.eval()

    _dino_models[variant] = model
    _dino_transforms[variant] = transform
    return model, _device, transform


class BaselineEmbedder:
    """
    Visual Embedding Pipeline.
    Supports either the baseline pretrained MobileNetV3 (960-dim)
    or any active fine-tuned checkpoint from ModelRegistry (e.g. 512-dim).
    """

    def __init__(self, model_id: Optional[str] = None, db_session=None):
        self.model_id = model_id or "mobilenetv3-pretrained-baseline"
        self._model = None
        self._device = None
        self._transform = None
        self._embedding_dim = 960

        if db_session:
            try:
                from app.services.model_registry_service import get_active_model, load_model_instance
                active = get_active_model(db_session, model_type="embedding")
                self.model_id = active.model_id
                self._embedding_dim = active.embedding_dim
                self._model, self._device, self._transform = load_model_instance(active)
            except Exception as e:
                # Fallback to baseline if registry lookup fails
                print(f"Warning: Could not load active model from registry ({e}), using baseline.")
                self.model_id = "mobilenetv3-pretrained-baseline"
                self._embedding_dim = 960

    @property
    def model_name(self) -> str:
        return self.model_id

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _ensure_loaded(self):
        if self._model is None:
            if "dino" in self.model_id.lower():
                variant = "dinov2_vitb14" if "vitb" in self.model_id.lower() else "dinov2_vits14"
                if "dinov3" in self.model_id.lower():
                    variant = "dinov3_vitb16" if "vitb" in self.model_id.lower() else "dinov3_vits16"
                self._model, self._device, self._transform = get_dino_embedder(variant=variant)
                self._embedding_dim = 768 if "vitb" in self.model_id.lower() else 384
            else:
                self._model, self._device, self._transform = get_embedder()
                self._embedding_dim = 960

    def extract_embedding_from_crop(self, pil_crop: Image.Image) -> np.ndarray:
        """
        Preprocesses a PIL Image crop and returns a 1D L2-normalized float32 numpy vector.
        """
        self._ensure_loaded()
        rgb_img = pil_crop.convert("RGB")
        tensor_img = self._transform(rgb_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model(tensor_img)
            # Flatten to 1D vector
            features = features.squeeze().cpu().numpy().astype(np.float32)

        # L2 Normalization so dot-product equals cosine similarity
        norm = np.linalg.norm(features)
        if norm > 1e-12:
            features = features / norm
        else:
            features = np.zeros_like(features)

        return features

    def extract_embeddings_batched(self, pil_crops: List[Image.Image]) -> np.ndarray:
        """
        Preprocesses a batch of PIL Image crops and returns a 2D (B, D) L2-normalized float32 numpy array.
        Significantly reduces inference latency by executing a single batched neural forward pass.
        """
        if not pil_crops:
            return np.empty((0, self._embedding_dim), dtype=np.float32)

        if len(pil_crops) == 1:
            return self.extract_embedding_from_crop(pil_crops[0]).reshape(1, -1)

        self._ensure_loaded()
        tensors = torch.stack([self._transform(c.convert("RGB")) for c in pil_crops]).to(self._device)

        with torch.no_grad():
            features = self._model(tensors)
            features = features.cpu().numpy().astype(np.float32)

        # Ensure 2D shape (B, D)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        # Batch L2 Normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.where(norms > 1e-12, features / norms, np.zeros_like(features))

        return features

    def extract_embedding_from_file(self, image_path: str) -> np.ndarray:
        """Loads image from disk and extracts embedding."""
        with Image.open(image_path) as img:
            return self.extract_embedding_from_crop(img)
