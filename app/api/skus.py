import os
import uuid
import mimetypes
from typing import List, Optional
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.database import get_db
from app.core.config import ORIGINAL_IMAGES_DIR
from app.models.sku import SKU
from app.models.image import ProductImage
from app.models.embedding import Embedding
from app.schemas.sku import (
    SKUCreate,
    SKUUpdate,
    SKUResponse,
    ProductImageSummary,
    FewShotEnrollmentResponse,
)
from app.services.dataset_service import compute_file_hash, inspect_image_quality
from app.services.model_registry_service import get_active_model
from app.services.embedder import BaselineEmbedder

router = APIRouter(prefix="/skus", tags=["SKUs"])


@router.post("", response_model=SKUResponse, status_code=status.HTTP_201_CREATED)
def create_sku(sku_in: SKUCreate, db: Session = Depends(get_db)):
    """
    Registers a new SKU.
    Enforces uniqueness on sku_code.
    """
    existing = db.query(SKU).filter(SKU.sku_code == sku_in.sku_code.strip()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"SKU code '{sku_in.sku_code}' already exists."
        )

    new_sku = SKU(
        sku_code=sku_in.sku_code.strip().upper(),
        name=sku_in.name.strip(),
        category=sku_in.category.strip() if sku_in.category else None,
        description=sku_in.description.strip() if sku_in.description else None,
        is_active=True,
    )
    db.add(new_sku)
    db.commit()
    db.refresh(new_sku)

    return _build_sku_response(new_sku, db)


@router.post("/enroll-few-shot", response_model=FewShotEnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_sku_few_shot(
    sku_code: str = Form(..., description="Unique SKU code identifier"),
    name: str = Form(..., description="Human-readable product name"),
    category: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    reference_images: List[UploadFile] = File(..., description="1 or more reference images of the product"),
    db: Session = Depends(get_db),
):
    """
    SECTION 3: Few-Shot SKU Enrollment.
    Registers a new SKU and immediately indexes its reference embeddings into the active gallery
    WITHOUT retraining or fine-tuning the embedding model backbone.
    Atomic and transactional: rolls back if reference embedding generation fails.
    """
    clean_code = sku_code.strip().upper()
    clean_name = name.strip()

    if not clean_code or not clean_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both SKU code and Product Name are required."
        )

    # Check uniqueness
    existing = db.query(SKU).filter(SKU.sku_code == clean_code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"SKU code '{clean_code}' already exists."
        )

    if not reference_images or len(reference_images) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one reference image is required for few-shot enrollment."
        )

    # Verify active embedding model is available and ready
    active_model = get_active_model(db, model_type="embedding")
    if not active_model:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Active embedding model is not configured in the model registry."
        )

    embedder = BaselineEmbedder(db_session=db)

    # Create SKU entity in database
    new_sku = SKU(
        sku_code=clean_code,
        name=clean_name,
        category=category.strip() if category else None,
        description=description.strip() if description else None,
        is_active=True,
    )
    db.add(new_sku)
    db.flush()

    enrolled_count = 0
    saved_file_paths = []

    try:
        for file in reference_images:
            file_bytes = await file.read()
            if len(file_bytes) == 0:
                continue

            # Validate file size (max 15MB)
            if len(file_bytes) > 15 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{file.filename}' exceeds maximum allowed size of 15MB."
                )

            # Check duplicate file hash across dataset
            file_hash = compute_file_hash(file_bytes)
            existing_img = db.query(ProductImage).filter(ProductImage.file_hash == file_hash).first()
            if existing_img:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Duplicate reference image: '{file.filename}' has identical hash to existing image ID {existing_img.id}."
                )

            # Save file to storage
            ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                ext = ".jpg"
            unique_name = f"{new_sku.sku_code}_ref_{uuid.uuid4().hex[:10]}{ext}"
            saved_path = ORIGINAL_IMAGES_DIR / unique_name

            with open(saved_path, "wb") as f:
                f.write(file_bytes)
            saved_file_paths.append(saved_path)

            # Inspect quality
            quality = inspect_image_quality(str(saved_path))
            if not quality["is_valid"]:
                continue

            mime = file.content_type or mimetypes.guess_type(file.filename)[0] or "image/jpeg"
            val_notes = "; ".join(quality["issues"]) if quality["issues"] else "Quality check passed"

            image_record = ProductImage(
                sku_id=new_sku.id,
                role="reference",
                split="reference",
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
                is_valid=True,
                validation_notes=val_notes,
            )
            db.add(image_record)
            db.flush()

            # Extract L2-normalized embedding using active model without retraining
            vector = embedder.extract_embedding_from_file(str(saved_path))
            if len(vector) != active_model.embedding_dim:
                raise ValueError(
                    f"Generated vector dim ({len(vector)}) does not match active model dim ({active_model.embedding_dim})"
                )

            emb = Embedding(
                image_id=image_record.id,
                sku_id=new_sku.id,
                model_name=active_model.model_id,
                embedding_dim=len(vector),
                vector_bytes=vector.astype(np.float32).tobytes(),
                is_active=True,
            )
            db.add(emb)
            enrolled_count += 1

        if enrolled_count == 0:
            db.rollback()
            for p in saved_file_paths:
                if os.path.exists(p):
                    os.remove(p)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="None of the uploaded reference images could be decoded or indexed."
            )

        db.commit()
        db.refresh(new_sku)

    except HTTPException:
        db.rollback()
        for p in saved_file_paths:
            if os.path.exists(p):
                os.remove(p)
        raise
    except Exception as e:
        db.rollback()
        for p in saved_file_paths:
            if os.path.exists(p):
                os.remove(p)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Few-shot enrollment failed during reference embedding indexing: {str(e)}"
        )

    return FewShotEnrollmentResponse(
        status="success",
        sku_id=new_sku.id,
        sku_code=new_sku.sku_code,
        name=new_sku.name,
        category=new_sku.category,
        enrolled_images_count=enrolled_count,
        model_id=active_model.model_id,
        embedding_dim=active_model.embedding_dim,
        message=f"Successfully enrolled '{new_sku.sku_code}' with {enrolled_count} reference embedding(s) into active gallery without retraining (Model: {active_model.model_id}).",
        retrained=False,
        sku=_build_sku_response(new_sku, db),
    )



@router.get("", response_model=List[SKUResponse])
def list_skus(
    include_inactive: bool = Query(False, description="Include deactivated SKUs"),
    search: Optional[str] = Query(None, description="Search by SKU code or name"),
    db: Session = Depends(get_db),
):
    """Lists registered SKUs with image counts."""
    query = db.query(SKU)
    if not include_inactive:
        query = query.filter(SKU.is_active == True)
    if search:
        search_fmt = f"%{search.strip()}%"
        query = query.filter((SKU.sku_code.ilike(search_fmt)) | (SKU.name.ilike(search_fmt)))

    skus = query.order_by(SKU.created_at.desc()).all()
    return [_build_sku_response(sku, db) for sku in skus]


@router.get("/{sku_id}", response_model=SKUResponse)
def get_sku(sku_id: int, db: Session = Depends(get_db)):
    """Retrieves a SKU with separated reference and training images."""
    sku = db.query(SKU).filter(SKU.id == sku_id).first()
    if not sku:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SKU with id {sku_id} not found.")
    return _build_sku_response(sku, db)


@router.patch("/{sku_id}", response_model=SKUResponse)
def update_sku(sku_id: int, sku_update: SKUUpdate, db: Session = Depends(get_db)):
    """Updates an existing SKU."""
    sku = db.query(SKU).filter(SKU.id == sku_id).first()
    if not sku:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SKU with id {sku_id} not found.")

    if sku_update.name is not None:
        sku.name = sku_update.name.strip()
    if sku_update.category is not None:
        sku.category = sku_update.category.strip()
    if sku_update.description is not None:
        sku.description = sku_update.description.strip()
    if sku_update.is_active is not None:
        sku.is_active = sku_update.is_active

    db.commit()
    db.refresh(sku)
    return _build_sku_response(sku, db)


@router.post("/{sku_id}/deactivate", response_model=SKUResponse)
def deactivate_sku(sku_id: int, db: Session = Depends(get_db)):
    """Deactivates a SKU without deleting associated historical data."""
    sku = db.query(SKU).filter(SKU.id == sku_id).first()
    if not sku:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SKU with id {sku_id} not found.")

    sku.is_active = False
    db.commit()
    db.refresh(sku)
    return _build_sku_response(sku, db)


@router.delete("/{sku_id}", status_code=status.HTTP_200_OK)
def delete_sku(sku_id: int, db: Session = Depends(get_db)):
    """Deletes a SKU and cascades to its images."""
    sku = db.query(SKU).filter(SKU.id == sku_id).first()
    if not sku:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"SKU with id {sku_id} not found.")

    db.delete(sku)
    db.commit()
    return {"status": "success", "message": f"SKU {sku_id} deleted successfully."}


def _build_sku_response(sku: SKU, db: Session) -> SKUResponse:
    images = db.query(ProductImage).filter(ProductImage.sku_id == sku.id).all()
    ref_imgs = [img for img in images if img.role == "reference"]
    train_imgs = [img for img in images if img.role != "reference"]

    ref_summaries = [
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
        for img in ref_imgs
    ]

    train_summaries = [
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
        for img in train_imgs
    ]

    return SKUResponse(
        id=sku.id,
        sku_code=sku.sku_code,
        name=sku.name,
        category=sku.category,
        description=sku.description,
        is_active=sku.is_active,
        created_at=sku.created_at,
        updated_at=sku.updated_at,
        reference_image_count=len(ref_imgs),
        training_image_count=len(train_imgs),
        total_image_count=len(images),
        reference_images=ref_summaries,
        training_images=train_summaries,
    )
