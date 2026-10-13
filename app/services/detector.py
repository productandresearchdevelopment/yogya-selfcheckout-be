import os
import uuid
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np

from app.core.config import CROPPED_IMAGES_DIR, DEFAULT_DETECTION_CONFIDENCE_THRESHOLD
from app.core.device import get_torch_device

# COCO Category names for FasterRCNN
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

_detector_model = None
_device = None


def get_detector():
    """Lazy loader for FasterRCNN MobileNetV3 object detector."""
    global _detector_model, _device
    if _detector_model is not None:
        return _detector_model, _device

    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

    _device = get_torch_device()
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.to(_device)
    model.eval()
    _detector_model = model
    return _detector_model, _device


class BaselineDetector:
    """
    Object Localization Pipeline using pretrained detector.
    Localizes candidate products, extracts bounding boxes, and generates crops.
    """

    def __init__(self, confidence_threshold: float = DEFAULT_DETECTION_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def detect_and_crop(
        self,
        image_path: str,
        confidence_threshold: float = None,
        allow_fallback: bool = True,
        min_crop_size: int = 10,
    ) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
        """
        Runs object detection on the image.
        Returns:
            - List of detected objects with bounding boxes, scores, and crop paths.
            - Boolean flag indicating whether fallback was required.
            - List of warnings / notes.
        """
        import torch
        from torchvision import transforms

        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        warnings: List[str] = []

        pil_img = Image.open(image_path).convert("RGB")
        img_w, img_h = pil_img.size

        model, device = get_detector()
        transform = transforms.ToTensor()
        tensor_img = transform(pil_img).to(device)

        with torch.no_grad():
            predictions = model([tensor_img])[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        detected_objects: List[Dict[str, Any]] = []
        box_idx = 0

        for box, score, label_idx in zip(boxes, scores, labels):
            if score < thresh:
                continue

            x1, y1, x2, y2 = box
            # Clamp coordinates to image boundaries
            x1 = max(0, min(float(x1), img_w - 1))
            y1 = max(0, min(float(y1), img_h - 1))
            x2 = max(x1 + 1, min(float(x2), img_w))
            y2 = max(y1 + 1, min(float(y2), img_h))

            # Skip tiny degenerate boxes
            px_w = int(round(x2 - x1))
            px_h = int(round(y2 - y1))
            if px_w < min_crop_size or px_h < min_crop_size:
                continue

            # Normalized coordinates
            norm_x_min = x1 / img_w
            norm_y_min = y1 / img_h
            norm_x_max = x2 / img_w
            norm_y_max = y2 / img_h

            class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"class_{label_idx}"

            # Extract crop
            crop = pil_img.crop((x1, y1, x2, y2))
            crop_filename = f"crop_{uuid.uuid4().hex[:12]}_{box_idx}.jpg"
            crop_path = CROPPED_IMAGES_DIR / crop_filename
            crop.save(crop_path, quality=95)

            detected_objects.append({
                "box_id": box_idx,
                "x_min": round(norm_x_min, 4),
                "y_min": round(norm_y_min, 4),
                "x_max": round(norm_x_max, 4),
                "y_max": round(norm_y_max, 4),
                "px_x": int(x1),
                "px_y": int(y1),
                "px_width": px_w,
                "px_height": px_h,
                "detector_confidence": round(float(score), 4),
                "detector_class": class_name,
                "is_fallback": False,
                "crop_path": str(crop_path),
                "crop_filename": crop_filename,
            })
            box_idx += 1

        is_fallback = False
        if len(detected_objects) == 0 and allow_fallback:
            # Fallback handling:
            # Requirements: "Avoid treating the entire frame as a detected product without clearly indicating that it is a fallback."
            is_fallback = True
            warnings.append(
                f"No object detected above threshold ({thresh:.2f}). "
                f"Generic COCO detector may not detect custom packaging shapes. "
                f"Using full-frame fallback crop for baseline evaluation."
            )

            fallback_crop_filename = f"crop_fallback_{uuid.uuid4().hex[:12]}.jpg"
            fallback_crop_path = CROPPED_IMAGES_DIR / fallback_crop_filename
            pil_img.save(fallback_crop_path, quality=95)

            detected_objects.append({
                "box_id": 0,
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 1.0,
                "y_max": 1.0,
                "px_x": 0,
                "px_y": 0,
                "px_width": img_w,
                "px_height": img_h,
                "detector_confidence": 0.0,
                "detector_class": "fallback_full_frame",
                "is_fallback": True,
                "crop_path": str(fallback_crop_path),
                "crop_filename": fallback_crop_filename,
            })

        return detected_objects, is_fallback, warnings
