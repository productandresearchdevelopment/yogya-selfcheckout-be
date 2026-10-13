import os
import uuid
import mimetypes
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import ORIGINAL_IMAGES_DIR, CROPPED_IMAGES_DIR, EXPORTS_DIR
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.annotation import Annotation
from app.models.embedding import Embedding
from app.schemas.dataset import (
    AnnotationCreate,
    AnnotationResponse,
    DatasetSplitUpdate,
    DatasetExportResponse,
)
from app.schemas.sku import ProductImageSummary
from app.services.dataset_service import (
    compute_file_hash,
    inspect_image_quality,
    export_dataset_yolo,
    export_dataset_coco,
)
from app.services.gallery import GalleryService

router = APIRouter(prefix="/datasets", tags=["Datasets"])
gallery_service = GalleryService()


@router.post("/upload", response_model=List[ProductImageSummary], status_code=status.HTTP_201_CREATED)
async def upload_images(
    files: List[UploadFile] = File(...),
    sku_id: int = Form(...),
    role: str = Form("reference"),  # 'reference', 'training', 'validation', 'test'
    split: str = Form("unassigned"),  # 'train', 'val', 'test', 'reference', 'unassigned'
    db: Session = Depends(get_db),
):
    """
    Uploads images, performs quality checks, detects duplicates via SHA-256 hash,
    and indexes embeddings for reference images.
    """
    sku = db.query(SKU).filter(SKU.id == sku_id).first()
    if not sku:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SKU {sku_id} not found.")

    if role not in ["reference", "training", "validation", "test"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{role}'. Allowed: reference, training, validation, test.",
        )

    # Set default split if not provided
    if split == "unassigned":
        if role == "reference":
            split = "reference"
        elif role == "training":
            split = "train"
        elif role == "validation":
            split = "val"
        elif role == "test":
            split = "test"

    results = []

    for file in files:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            continue

        file_hash = compute_file_hash(file_bytes)

        # Check for duplicate image hash across active dataset
        existing_duplicate = db.query(ProductImage).filter(ProductImage.file_hash == file_hash).first()
        if existing_duplicate:
            # We reject duplicate file to maintain dataset integrity
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Duplicate file detected: '{file.filename}' has identical SHA-256 hash to existing image ID {existing_duplicate.id} ({existing_duplicate.filename})."
            )

        # Generate unique filename
        ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
        unique_name = f"{sku.sku_code}_{role}_{uuid.uuid4().hex[:10]}{ext}"
        saved_path = ORIGINAL_IMAGES_DIR / unique_name

        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        # Run quality inspection
        quality = inspect_image_quality(str(saved_path))
        mime = file.content_type or mimetypes.guess_type(file.filename)[0] or "image/jpeg"

        validation_notes = "; ".join(quality["issues"]) if quality["issues"] else "Quality check passed"

        image_record = ProductImage(
            sku_id=sku.id,
            role=role,
            split=split,
            filename=unique_name,
            file_path=str(saved_path),
            file_hash=file_hash,
            file_size_bytes=quality["file_size_bytes"],
            width=quality["width"],
            height=quality["height"],
            mime_type=mime,
            blur_score=quality["blur_score"],
            brightness_score=quality["brightness_score"],
            contrast_score=quality["contrast_score"],
            is_valid=quality["is_valid"],
            validation_notes=validation_notes,
        )
        db.add(image_record)
        db.commit()
        db.refresh(image_record)

        # Automatically index embedding for reference images
        if role == "reference" and image_record.is_valid:
            try:
                gallery_service.index_image_embedding(db, image_record)
            except Exception as e:
                image_record.validation_notes = f"{validation_notes} (Embedding indexing warning: {str(e)})"
                db.commit()

        results.append(
            ProductImageSummary(
                id=image_record.id,
                role=image_record.role,
                split=image_record.split,
                filename=image_record.filename,
                file_path=image_record.file_path,
                file_hash=image_record.file_hash,
                file_size_bytes=image_record.file_size_bytes,
                width=image_record.width,
                height=image_record.height,
                blur_score=image_record.blur_score,
                brightness_score=image_record.brightness_score,
                contrast_score=image_record.contrast_score,
                is_valid=image_record.is_valid,
                validation_notes=image_record.validation_notes,
                created_at=image_record.created_at,
                has_embedding=image_record.embedding is not None,
            )
        )

    return results


@router.get("/images", response_model=List[ProductImageSummary])
def list_images(
    sku_id: Optional[int] = Query(None),
    role: Optional[str] = Query(None),
    split: Optional[str] = Query(None),
    is_valid: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
):
    """Lists images with optional filters."""
    query = db.query(ProductImage)
    if sku_id is not None:
        query = query.filter(ProductImage.sku_id == sku_id)
    if role:
        query = query.filter(ProductImage.role == role)
    if split:
        query = query.filter(ProductImage.split == split)
    if is_valid is not None:
        query = query.filter(ProductImage.is_valid == is_valid)

    images = query.order_by(ProductImage.created_at.desc()).all()
    return [
        ProductImageSummary(
            id=img.id,
            role=img.role,
            split=img.split,
            filename=img.filename,
            file_path=img.file_path,
            file_hash=img.file_hash,
            file_size_bytes=img.file_size_bytes,
            width=img.width,
            height=img.height,
            blur_score=img.blur_score,
            brightness_score=img.brightness_score,
            contrast_score=img.contrast_score,
            is_valid=img.is_valid,
            validation_notes=img.validation_notes,
            created_at=img.created_at,
            has_embedding=img.embedding is not None,
        )
        for img in images
    ]


@router.get("/images/{image_id}/file")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    """Serves the actual image file."""
    img = db.query(ProductImage).filter(ProductImage.id == image_id).first()
    if not img or not os.path.exists(img.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file not found.")
    return FileResponse(img.file_path, media_type=img.mime_type)


@router.get("/crops/{filename}")
def get_crop_file(filename: str):
    """Serves cropped detection images."""
    crop_path = CROPPED_IMAGES_DIR / filename
    if not crop_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cropped image file not found.")
    return FileResponse(str(crop_path), media_type="image/jpeg")


@router.delete("/images/{image_id}", status_code=status.HTTP_200_OK)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    """Deletes an image file and database record."""
    img = db.query(ProductImage).filter(ProductImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image record not found.")

    if os.path.exists(img.file_path):
        try:
            os.remove(img.file_path)
        except Exception:
            pass

    db.delete(img)
    db.commit()
    return {"status": "success", "message": f"Image {image_id} deleted."}


@router.patch("/images/{image_id}/split", response_model=ProductImageSummary)
def update_image_split(image_id: int, split_in: DatasetSplitUpdate, db: Session = Depends(get_db)):
    """Updates image dataset split."""
    img = db.query(ProductImage).filter(ProductImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image record not found.")

    img.split = split_in.split
    db.commit()
    db.refresh(img)
    return ProductImageSummary(
        id=img.id,
        role=img.role,
        split=img.split,
        filename=img.filename,
        file_path=img.file_path,
        file_hash=img.file_hash,
        file_size_bytes=img.file_size_bytes,
        width=img.width,
        height=img.height,
        blur_score=img.blur_score,
        brightness_score=img.brightness_score,
        contrast_score=img.contrast_score,
        is_valid=img.is_valid,
        validation_notes=img.validation_notes,
        created_at=img.created_at,
        has_embedding=img.embedding is not None,
    )


# --- Bounding Box Annotation Endpoints ---

@router.post("/images/{image_id}/annotations", response_model=AnnotationResponse, status_code=status.HTTP_201_CREATED)
def create_annotation(image_id: int, ann_in: AnnotationCreate, db: Session = Depends(get_db)):
    """
    Saves a manual or predicted bounding box annotation for a dataset training image.
    Enables preparing datasets for subsequent object detector fine-tuning.
    """
    img = db.query(ProductImage).filter(ProductImage.id == image_id).first()
    if not img:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

    px_x = int(ann_in.x_min * img.width)
    px_y = int(ann_in.y_min * img.height)
    px_w = int((ann_in.x_max - ann_in.x_min) * img.width)
    px_h = int((ann_in.y_max - ann_in.y_min) * img.height)

    annotation = Annotation(
        image_id=img.id,
        sku_id=ann_in.sku_id or img.sku_id,
        x_min=ann_in.x_min,
        y_min=ann_in.y_min,
        x_max=ann_in.x_max,
        y_max=ann_in.y_max,
        px_x=px_x,
        px_y=px_y,
        px_width=px_w,
        px_height=px_h,
        label=ann_in.label,
        confidence=ann_in.confidence,
        source=ann_in.source,
    )
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


@router.get("/images/{image_id}/annotations", response_model=List[AnnotationResponse])
def get_image_annotations(image_id: int, db: Session = Depends(get_db)):
    """Returns all bounding box annotations for an image."""
    return db.query(Annotation).filter(Annotation.image_id == image_id).all()


@router.delete("/annotations/{annotation_id}", status_code=status.HTTP_200_OK)
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    """Deletes an annotation."""
    ann = db.query(Annotation).filter(Annotation.id == annotation_id).first()
    if not ann:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found.")
    db.delete(ann)
    db.commit()
    return {"status": "success", "message": f"Annotation {annotation_id} deleted."}


# --- Dataset Export & Gallery Re-indexing ---

@router.post("/export/yolo", response_model=DatasetExportResponse)
def export_yolo(db: Session = Depends(get_db)):
    """Exports dataset with bounding boxes in YOLO format ready for detector fine-tuning."""
    zip_path = export_dataset_yolo(db)
    total_imgs = db.query(ProductImage).count()
    total_anns = db.query(Annotation).count()
    return DatasetExportResponse(
        format="YOLO",
        total_images=total_imgs,
        total_annotations=total_anns,
        splits={"train": "images/train", "val": "images/val", "test": "images/test"},
        archive_path=zip_path,
        message="YOLO format dataset exported successfully with data.yaml and bounding boxes."
    )


@router.post("/export/coco", response_model=DatasetExportResponse)
def export_coco(db: Session = Depends(get_db)):
    """Exports dataset in standard COCO JSON format."""
    json_path = export_dataset_coco(db)
    total_imgs = db.query(ProductImage).count()
    total_anns = db.query(Annotation).count()
    return DatasetExportResponse(
        format="COCO",
        total_images=total_imgs,
        total_annotations=total_anns,
        splits={},
        archive_path=json_path,
        message="COCO annotations JSON exported successfully."
    )


@router.post("/reindex")
def reindex_gallery(db: Session = Depends(get_db)):
    """Reindexes visual embeddings for all active reference images."""
    count = gallery_service.reindex_all_references(db)
    return {"status": "success", "indexed_count": count, "message": f"Indexed {count} reference embeddings."}
