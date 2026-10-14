import os
import uuid
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

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

# Non-retail categories that should not be localized as products on a scanning counter
NON_RETAIL_CLASSES = {
    "person", "bed", "couch", "chair", "dining table", "toilet",
    "bench", "sink", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "train", "bus", "airplane", "boat", "bicycle",
    "car", "motorcycle", "truck", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
}

_detector_model = None
_device = None


def get_detector():
    """Lazy loader for FasterRCNN MobileNetV3 object detector."""
    global _detector_model, _device
    if _detector_model is not None:
        return _detector_model, _device

    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights

    _device = get_torch_device()
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    model.to(_device)
    model.eval()
    _detector_model = model
    return _detector_model, _device


def compute_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    boxBArea = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0


def apply_nms(boxes_data: List[Dict[str, Any]], iou_threshold: float = 0.50) -> List[Dict[str, Any]]:
    """
    Suppresses redundant overlapping bounding boxes on the same physical object.
    Preserves highest detector_confidence detections.
    """
    if not boxes_data or len(boxes_data) <= 1:
        return boxes_data

    sorted_boxes = sorted(boxes_data, key=lambda x: x["detector_confidence"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for item in sorted_boxes:
        box = (item["px_x"], item["px_y"], item["px_x"] + item["px_width"], item["px_y"] + item["px_height"])
        overlap = False
        for k in kept:
            k_box = (k["px_x"], k["px_y"], k["px_x"] + k["px_width"], k["px_y"] + k["px_height"])
            if compute_box_iou(box, k_box) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(item)

    # Re-index box_ids
    for idx, item in enumerate(kept):
        item["box_id"] = idx

    return kept


def detect_adaptive_foreground_box(
    pil_img: Image.Image,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.95,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Lightweight saliency heuristic for supermarket products on scanning beds/tables.
    Finds high-contrast foreground bounding contour when generic COCO classes miss packaged goods.
    """
    try:
        import cv2
        np_img = np.array(pil_img.convert("RGB"))
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Otsu thresholding with inverse mask
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        total_area = pil_img.width * pil_img.height
        best_box = None
        best_area = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            area_ratio = area / total_area
            if min_area_ratio <= area_ratio <= max_area_ratio:
                if area > best_area:
                    best_area = area
                    best_box = (x, y, x + w, y + h)

        return best_box
    except Exception:
        return None


class BaselineDetector:
    """
    Object Localization Pipeline using pretrained detector.
    Localizes candidate products, extracts bounding boxes, and generates crops.
    Supports Non-Maximum Suppression (NMS), crop provenance tracking, and adaptive saliency fallback.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        nms_iou_threshold: float = 0.50,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def detect_and_crop(
        self,
        image_path_or_pil: Any,
        confidence_threshold: float = None,
        allow_fallback: bool = True,
        min_crop_size: int = 10,
        save_crops: bool = False,
        enable_adaptive_saliency: bool = True,
    ) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
        """
        Runs object detection on the image.
        Returns:
            - List of detected objects with bounding boxes, scores, crop provenance, and crop images.
            - Boolean flag indicating whether fallback was required.
            - List of warnings / notes.
        """
        from torchvision import transforms

        thresh = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        warnings: List[str] = []

        if isinstance(image_path_or_pil, Image.Image):
            pil_img = image_path_or_pil.convert("RGB")
        else:
            pil_img = Image.open(str(image_path_or_pil)).convert("RGB")
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

            # Skip degenerate boxes
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

            # Filter out humans, furniture, and non-retail background classes
            if class_name in NON_RETAIL_CLASSES:
                continue

            # Filter out giant bounding boxes covering almost the entire frame (> 85% area)
            area_ratio = (px_w * px_h) / (img_w * img_h)
            if area_ratio > 0.85:
                continue

            # Extract crop
            crop = pil_img.crop((x1, y1, x2, y2))
            crop_filename = None
            crop_path = None
            if save_crops:
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
                "crop_provenance": "detector_roi",
                "is_fallback": False,
                "crop_img": crop,
                "crop_path": str(crop_path) if crop_path else None,
                "crop_filename": crop_filename,
            })
            box_idx += 1

        # Apply Non-Maximum Suppression to remove overlapping duplicate detections
        if len(detected_objects) > 1:
            detected_objects = apply_nms(detected_objects, iou_threshold=self.nms_iou_threshold)

        is_fallback = False

        # If Faster R-CNN produced zero detections, attempt adaptive saliency foreground localization
        if len(detected_objects) == 0 and enable_adaptive_saliency:
            salient_box = detect_adaptive_foreground_box(pil_img)
            if salient_box:
                sx1, sy1, sx2, sy2 = salient_box
                spx_w = sx2 - sx1
                spx_h = sy2 - sy1
                if spx_w >= min_crop_size and spx_h >= min_crop_size:
                    s_crop = pil_img.crop((sx1, sy1, sx2, sy2))
                    s_filename = None
                    s_path = None
                    if save_crops:
                        s_filename = f"crop_saliency_{uuid.uuid4().hex[:12]}.jpg"
                        s_path = CROPPED_IMAGES_DIR / s_filename
                        s_crop.save(s_path, quality=95)

                    detected_objects.append({
                        "box_id": 0,
                        "x_min": round(sx1 / img_w, 4),
                        "y_min": round(sy1 / img_h, 4),
                        "x_max": round(sx2 / img_w, 4),
                        "y_max": round(sy2 / img_h, 4),
                        "px_x": sx1,
                        "px_y": sy1,
                        "px_width": spx_w,
                        "px_height": spx_h,
                        "detector_confidence": 0.45,
                        "detector_class": "retail_foreground_roi",
                        "crop_provenance": "adaptive_saliency_roi",
                        "is_fallback": False,
                        "crop_img": s_crop,
                        "crop_path": str(s_path) if s_path else None,
                        "crop_filename": s_filename,
                    })
                    warnings.append(
                        "COCO detector produced no detections for packaged good; "
                        "adaptive foreground saliency localized candidate retail product."
                    )

        # Fallback handling if still empty and allow_fallback is enabled
        if len(detected_objects) == 0 and allow_fallback:
            is_fallback = True
            warnings.append(
                f"No object detected above threshold ({thresh:.2f}). "
                f"Using full-frame fallback crop for baseline evaluation."
            )

            fallback_crop_filename = None
            fallback_crop_path = None
            if save_crops:
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
                "crop_provenance": "fallback_full_frame",
                "is_fallback": True,
                "crop_img": pil_img,
                "crop_path": str(fallback_crop_path) if fallback_crop_path else None,
                "crop_filename": fallback_crop_filename,
            })

        return detected_objects, is_fallback, warnings


# ---------------------------------------------------------------------------
# Retail Detector Training Infrastructure
# ---------------------------------------------------------------------------

class RetailDetectionDataset(Dataset):
    """
    PyTorch Dataset infrastructure for fine-tuning Faster R-CNN on retail grocery bounding boxes.
    Loads annotations from PostgreSQL annotations table and returns (image_tensor, target_dict).
    """

    def __init__(self, samples: List[Dict[str, Any]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        from torchvision import transforms

        item = self.samples[idx]
        image_path = item["image_path"]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            w, h = img.size

            boxes = []
            labels = []

            for ann in item["annotations"]:
                # Convert normalized [0, 1] to pixel [x1, y1, x2, y2]
                x1 = ann["x_min"] * w
                y1 = ann["y_min"] * h
                x2 = ann["x_max"] * w
                y2 = ann["y_max"] * h
                boxes.append([x1, y1, x2, y2])
                labels.append(1)  # Class 1 = retail_product (class-agnostic proposal head)

            if not boxes:
                # Fallback box
                boxes = [[0.0, 0.0, float(w), float(h)]]
                labels = [1]

            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([idx]),
                "area": torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            }

            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)

            return img_tensor, target


def check_retail_annotation_readiness(db) -> Dict[str, Any]:
    """
    Inspects annotation table to assess readiness for training a custom retail object detector.
    Reports the exact data blocker without fabricating training progress.
    """
    from app.models.annotation import Annotation
    total_anns = db.query(Annotation).count()
    manual_anns = db.query(Annotation).filter(Annotation.source == "manual").count()

    is_ready = total_anns >= 50 and manual_anns >= 50
    return {
        "total_annotations": total_anns,
        "manual_annotations": manual_anns,
        "is_ready_for_fine_tuning": is_ready,
        "minimum_required": 50,
        "status": "READY" if is_ready else "BLOCKED_INSUFFICIENT_LABELED_DATA",
        "diagnostic_message": (
            f"Found {total_anns} annotations in database. Training a robust retail detector requires "
            f"at least 50-100 manual bounding box annotations across packaged grocery goods."
            if not is_ready else "Annotation dataset meets minimum threshold for detector fine-tuning."
        ),
    }


# Architectural alias per Refactor Specification
ObjectDetector = BaselineDetector
