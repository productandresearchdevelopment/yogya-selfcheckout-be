import hashlib
import os
import shutil
import zipfile
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import cv2
import numpy as np

from app.core.config import ORIGINAL_IMAGES_DIR, EXPORTS_DIR


def compute_file_hash(file_bytes: bytes) -> str:
    """Computes SHA-256 hash for duplicate image detection."""
    return hashlib.sha256(file_bytes).hexdigest()


def inspect_image_quality(image_path: str) -> Dict[str, Any]:
    """
    Inspects image quality metrics:
    - Blur score using Laplacian variance (sharp > 100, blurry < 100)
    - Brightness score (mean luminance 0..255)
    - Contrast score (standard deviation of luminance)
    - Adequate resolution (>= 200x200)
    """
    issues = []
    
    # Check PIL readability
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            format_name = img.format or "UNKNOWN"
    except Exception as e:
        return {
            "width": 0,
            "height": 0,
            "file_size_bytes": os.path.getsize(image_path) if os.path.exists(image_path) else 0,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "is_sharp": False,
            "is_adequate_resolution": False,
            "is_adequate_lighting": False,
            "is_valid": False,
            "issues": [f"File is corrupt or unreadable: {str(e)}"],
        }

    file_size = os.path.getsize(image_path)

    # Resolution check
    is_adequate_res = width >= 200 and height >= 200
    if not is_adequate_res:
        issues.append(f"Low resolution: {width}x{height} (recommended at least 200x200)")

    # Read with OpenCV for CV metrics
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        return {
            "width": width,
            "height": height,
            "file_size_bytes": file_size,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "is_sharp": False,
            "is_adequate_resolution": is_adequate_res,
            "is_adequate_lighting": False,
            "is_valid": False,
            "issues": issues + ["OpenCV cannot decode image data"],
        }

    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Laplacian variance for blur
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_sharp = laplacian_var >= 80.0
    if not is_sharp:
        issues.append(f"Image appears blurry (sharpness score {laplacian_var:.1f} < 80.0)")

    # Brightness (mean)
    brightness = float(np.mean(gray))
    # Contrast (standard deviation)
    contrast = float(np.std(gray))

    is_adequate_lighting = 30.0 <= brightness <= 235.0
    if brightness < 30.0:
        issues.append(f"Image is underexposed / very dark (brightness {brightness:.1f})")
    elif brightness > 235.0:
        issues.append(f"Image is overexposed / washed out (brightness {brightness:.1f})")

    if contrast < 20.0:
        issues.append(f"Image has low dynamic contrast ({contrast:.1f})")

    return {
        "width": width,
        "height": height,
        "file_size_bytes": file_size,
        "blur_score": round(laplacian_var, 2),
        "brightness_score": round(brightness, 2),
        "contrast_score": round(contrast, 2),
        "is_sharp": is_sharp,
        "is_adequate_resolution": is_adequate_res,
        "is_adequate_lighting": is_adequate_lighting,
        "is_valid": len(issues) == 0 or (is_adequate_res and not any("corrupt" in iss for iss in issues)),
        "issues": issues,
    }


def export_dataset_yolo(db) -> str:
    """
    Exports dataset in standard YOLO format:
    dataset/
      images/
        train/
        val/
      labels/
        train/
        val/
      data.yaml
    """
    from app.models.sku import SKU
    from app.models.image import ProductImage
    from app.models.annotation import Annotation

    skus = db.query(SKU).filter(SKU.is_active == True).order_by(SKU.id).all()
    sku_to_idx = {sku.sku_code: idx for idx, sku in enumerate(skus)}
    idx_to_name = {idx: sku.sku_code for idx, sku in enumerate(skus)}

    export_dir = EXPORTS_DIR / "yolo_dataset"
    if export_dir.exists():
        shutil.rmtree(export_dir)

    for split in ["train", "val", "test"]:
        (export_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (export_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    images = db.query(ProductImage).all()
    count_images = 0
    count_annotations = 0

    for img in images:
        split = img.split if img.split in ["train", "val", "test"] else "train"
        if not os.path.exists(img.file_path):
            continue

        dest_img_path = export_dir / "images" / split / img.filename
        shutil.copy2(img.file_path, dest_img_path)
        count_images += 1

        # Build label file
        label_filename = Path(img.filename).stem + ".txt"
        label_path = export_dir / "labels" / split / label_filename

        annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
        label_lines = []
        for ann in annotations:
            class_id = sku_to_idx.get(ann.label, 0)
            # YOLO expects: class_id x_center y_center width height (normalized)
            x_center = (ann.x_min + ann.x_max) / 2.0
            y_center = (ann.y_min + ann.y_max) / 2.0
            box_w = max(0.001, ann.x_max - ann.x_min)
            box_h = max(0.001, ann.y_max - ann.y_min)
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            count_annotations += 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

    # Write data.yaml
    yaml_content = f"""# Product Recognition Studio - YOLO Detection Dataset
path: {export_dir.as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(skus)}
names: {list(idx_to_name.values())}
"""
    with open(export_dir / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Create zip archive
    zip_path = EXPORTS_DIR / "yolo_dataset.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for file in files:
                abs_f = os.path.join(root, file)
                rel_f = os.path.relpath(abs_f, export_dir)
                zipf.write(abs_f, rel_f)

    return str(zip_path)


def export_dataset_coco(db) -> str:
    """Exports dataset in standard COCO JSON format."""
    from app.models.sku import SKU
    from app.models.image import ProductImage
    from app.models.annotation import Annotation

    skus = db.query(SKU).filter(SKU.is_active == True).order_by(SKU.id).all()
    categories = [{"id": sku.id, "name": sku.sku_code, "supercategory": sku.category or "product"} for sku in skus]
    sku_id_map = {sku.sku_code: sku.id for sku in skus}

    images = db.query(ProductImage).all()
    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img in images:
        coco_images.append({
            "id": img.id,
            "file_name": img.filename,
            "width": img.width,
            "height": img.height,
            "split": img.split,
            "sku_id": img.sku_id,
        })

        annotations = db.query(Annotation).filter(Annotation.image_id == img.id).all()
        for ann in annotations:
            cat_id = sku_id_map.get(ann.label, ann.sku_id or 1)
            px_x = ann.px_x or int(ann.x_min * img.width)
            px_y = ann.px_y or int(ann.y_min * img.height)
            px_w = ann.px_width or int((ann.x_max - ann.x_min) * img.width)
            px_h = ann.px_height or int((ann.y_max - ann.y_min) * img.height)
            area = float(px_w * px_h)

            coco_annotations.append({
                "id": ann_id,
                "image_id": img.id,
                "category_id": cat_id,
                "bbox": [px_x, px_y, px_w, px_h],
                "area": area,
                "iscrowd": 0,
                "confidence": ann.confidence,
                "source": ann.source,
            })
            ann_id += 1

    coco_data = {
        "info": {
            "description": "Product Recognition Studio COCO Dataset Export",
            "version": "1.0",
        },
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    coco_json_path = EXPORTS_DIR / "coco_annotations.json"
    with open(coco_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)

    return str(coco_json_path)
