import os
import io
import time
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import (
    ORIGINAL_IMAGES_DIR,
    CROPPED_IMAGES_DIR,
    DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_MARGIN_THRESHOLD,
    DEFAULT_TOP_K,
    DETECTOR_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
)
from app.core.device import get_system_environment_info
from app.models.recognition_log import RecognitionLog
from app.models.sku import SKU
from app.models.embedding import Embedding
from app.models.image import ProductImage
from app.schemas.recognition import (
    RecognitionResponse,
    DetectedObject,
    RecognitionCandidate,
    RecognitionLogSummary,
    RecognitionStatusSummary,
)
from app.services.detector import BaselineDetector
from app.services.embedder import BaselineEmbedder
from app.services.gallery import GalleryService
from app.services.model_registry_service import get_active_model

router = APIRouter(prefix="/recognition", tags=["Recognition"])

baseline_detector = BaselineDetector()
gallery_service = GalleryService()

_loaded_embedders: Dict[str, BaselineEmbedder] = {}


def get_active_embedder(db: Session) -> Tuple[BaselineEmbedder, str, int]:
    """
    Dynamically retrieves or reuses the cached embedder corresponding to the active model in ModelRegistry.
    Guarantees synchronization with active gallery vectors.
    """
    active_model = get_active_model(db, model_type="embedding")
    model_id = active_model.model_id
    if model_id not in _loaded_embedders:
        _loaded_embedders[model_id] = BaselineEmbedder(model_id=model_id, db_session=db)
    return _loaded_embedders[model_id], model_id, active_model.embedding_dim


def execute_recognition_pipeline(
    db: Session,
    image_path: Any,
    confidence_threshold: float,
    similarity_threshold: float,
    top_k: int,
    margin_threshold: float = 0.0,
    min_crop_size: int = 10,
    allow_fallback: bool = True,
    save_crops: bool = False,
) -> Tuple[List[DetectedObject], str, Optional[str], float, bool, List[str], str, str]:
    active_embedder, active_model_id, embedding_dim = get_active_embedder(db)

    raw_detections, is_fallback, warnings = baseline_detector.detect_and_crop(
        image_path,
        confidence_threshold=confidence_threshold,
        allow_fallback=allow_fallback,
        min_crop_size=min_crop_size,
        save_crops=save_crops,
    )

    detected_objects: List[DetectedObject] = []
    overall_status = "UNKNOWN" if len(raw_detections) > 0 else "NO_OBJECTS_DETECTED"
    top_overall_sku = None
    top_overall_sim = 0.0 if len(raw_detections) == 0 else -1.0

    # High-Performance Batch Optimization: Load all crops and extract embeddings in a single neural forward pass
    loaded_crops: List[Image.Image] = []
    valid_obj_indices: List[int] = []

    for idx, obj in enumerate(raw_detections):
        crop_img = obj.get("crop_img")
        if crop_img is not None:
            loaded_crops.append(crop_img.convert("RGB"))
            valid_obj_indices.append(idx)
        elif obj.get("crop_path"):
            try:
                with Image.open(obj["crop_path"]) as c_img:
                    loaded_crops.append(c_img.convert("RGB"))
                    valid_obj_indices.append(idx)
            except Exception:
                pass

    # Single batched forward pass through active embedder
    batched_vectors = None
    if loaded_crops:
        try:
            batched_vectors = active_embedder.extract_embeddings_batched(loaded_crops)
        except Exception as batch_err:
            print(f"Warning: Batched embedding failed ({batch_err}), falling back to per-crop inference.")
            batched_vectors = None

    has_ambiguous = False
    for idx, obj in enumerate(raw_detections):
        try:
            if batched_vectors is not None and idx in valid_obj_indices:
                crop_idx = valid_obj_indices.index(idx)
                crop_vector = batched_vectors[crop_idx]
            else:
                crop_img = obj.get("crop_img")
                if crop_img is not None:
                    crop_vector = active_embedder.extract_embedding_from_crop(crop_img)
                elif obj.get("crop_path"):
                    with Image.open(obj["crop_path"]) as c_img:
                        crop_vector = active_embedder.extract_embedding_from_crop(c_img)
                else:
                    continue

            search_result = gallery_service.search_candidates(
                db,
                query_vector=crop_vector,
                similarity_threshold=similarity_threshold,
                top_k=top_k,
                margin_threshold=margin_threshold,
            )

            candidates = [
                RecognitionCandidate(**c) for c in search_result.get("candidates", [])
            ]

            rec_status = search_result.get("recognition_status", "UNKNOWN")
            is_unknown = search_result.get("is_unknown", True)
            pred_sku = search_result.get("predicted_sku")
            pred_sku_name = search_result.get("predicted_sku_name")
            top_sim = search_result.get("top_similarity_score", 0.0)
            margin_delta = search_result.get("margin_delta")
            rejection_reason = search_result.get("rejection_reason")

            if rec_status == "AMBIGUOUS":
                has_ambiguous = True

            if not is_unknown and overall_status != "MATCHED":
                overall_status = "MATCHED"

            if top_sim > top_overall_sim:
                top_overall_sim = top_sim
                top_overall_sku = pred_sku

            crop_url = f"/api/v1/datasets/crops/{obj['crop_filename']}" if obj.get("crop_filename") else None

            detected_objects.append(
                DetectedObject(
                    box_id=obj["box_id"],
                    x_min=obj["x_min"],
                    y_min=obj["y_min"],
                    x_max=obj["x_max"],
                    y_max=obj["y_max"],
                    px_x=obj["px_x"],
                    px_y=obj["px_y"],
                    px_width=obj["px_width"],
                    px_height=obj["px_height"],
                    detector_confidence=obj["detector_confidence"],
                    detector_class=obj["detector_class"],
                    crop_provenance=obj.get("crop_provenance", "detector_roi"),
                    is_fallback=obj["is_fallback"],
                    crop_image_url=crop_url,
                    is_unknown=is_unknown,
                    recognition_status=rec_status,
                    predicted_sku=pred_sku,
                    predicted_sku_name=pred_sku_name,
                    top_similarity_score=top_sim,
                    margin_threshold=margin_threshold,
                    margin_delta=margin_delta,
                    rejection_reason=rejection_reason,
                    candidates=candidates,
                )
            )

        except Exception as crop_err:
            detected_objects.append(
                DetectedObject(
                    box_id=obj["box_id"],
                    x_min=obj["x_min"],
                    y_min=obj["y_min"],
                    x_max=obj["x_max"],
                    y_max=obj["y_max"],
                    px_x=obj["px_x"],
                    px_y=obj["px_y"],
                    px_width=obj["px_width"],
                    px_height=obj["px_height"],
                    detector_confidence=obj["detector_confidence"],
                    detector_class=obj["detector_class"],
                    crop_provenance=obj.get("crop_provenance", "fallback_full_frame"),
                    is_fallback=obj["is_fallback"],
                    crop_image_url=f"/api/v1/datasets/crops/{obj['crop_filename']}" if obj.get("crop_filename") else None,
                    is_unknown=True,
                    recognition_status="UNCLASSIFIED",
                    predicted_sku="UNCLASSIFIED",
                    predicted_sku_name=f"Inference error: {str(crop_err)}",
                    top_similarity_score=0.0,
                    margin_threshold=margin_threshold,
                    margin_delta=None,
                    rejection_reason=f"Crop embedding failed: {str(crop_err)}",
                    candidates=[],
                )
            )

    if overall_status == "UNKNOWN" and has_ambiguous:
        overall_status = "AMBIGUOUS"

    return (
        detected_objects,
        overall_status,
        top_overall_sku,
        top_overall_sim,
        is_fallback,
        warnings,
        DETECTOR_MODEL_NAME,
        active_model_id,
    )


@router.get("/status", response_model=RecognitionStatusSummary)
def get_recognition_status(db: Session = Depends(get_db)):
    """Retrieves runtime model registry status and gallery readiness."""
    active_emb = get_active_model(db, model_type="embedding")
    active_det = get_active_model(db, model_type="detector")
    env_info = get_system_environment_info()

    ref_count = (
        db.query(Embedding)
        .filter(Embedding.model_name == active_emb.model_id, Embedding.is_active == True)
        .count()
    )
    sku_count = db.query(SKU).filter(SKU.is_active == True).count()

    return RecognitionStatusSummary(
        active_detector=active_det.model_id if active_det else DETECTOR_MODEL_NAME,
        active_embedding_model=active_emb.model_id,
        embedding_dim=active_emb.embedding_dim,
        gallery_reference_count=ref_count,
        registered_sku_count=sku_count,
        is_gallery_compatible=True,
        device_used=env_info["active_device"],
        default_detection_threshold=DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        default_similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
    )


@router.post("/predict", response_model=RecognitionResponse)
async def recognize_products(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(DEFAULT_DETECTION_CONFIDENCE_THRESHOLD),
    similarity_threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD),
    top_k: int = Form(DEFAULT_TOP_K),
    margin_threshold: float = Form(DEFAULT_MARGIN_THRESHOLD),
    min_crop_size: int = Form(10),
    allow_fallback: bool = Form(True),
    db: Session = Depends(get_db),
):
    """
    Executes the End-to-End Baseline Detection and Recognition Pipeline:
    1. Localizes objects using the baseline pretrained detector.
    2. Returns actual bounding box coordinates for each detected object.
    3. Crops each detected object independently.
    4. Extracts visual embeddings using the active visual backbone.
    5. Computes cosine similarity against gallery reference embeddings.
    6. Returns ranked candidates and applies configurable rejection rules to flag UNKNOWN.
    7. Logs run to recognition_logs.
    """
    start_time = time.time()
    env_info = get_system_environment_info()

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    query_image_name = f"query_{uuid.uuid4().hex[:12]}{ext}"
    query_image_path = ORIGINAL_IMAGES_DIR / query_image_name

    with open(query_image_path, "wb") as f:
        f.write(file_bytes)

    try:
        with Image.open(query_image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image file: {str(e)}",
        )

    (
        detected_objects,
        overall_status,
        top_overall_sku,
        top_overall_sim,
        is_fallback,
        warnings,
        detector_name,
        embedding_name,
    ) = execute_recognition_pipeline(
        db=db,
        image_path=str(query_image_path),
        confidence_threshold=confidence_threshold,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        margin_threshold=margin_threshold,
        min_crop_size=min_crop_size,
        allow_fallback=allow_fallback,
    )

    duration_ms = round((time.time() - start_time) * 1000, 2)

    # Log to database
    log_entry = RecognitionLog(
        query_image_name=query_image_name,
        query_image_path=str(query_image_path),
        num_detections=len(detected_objects),
        final_status=overall_status,
        top_predicted_sku=top_overall_sku,
        top_similarity=top_overall_sim if top_overall_sim >= 0 else None,
        detector_model=detector_name,
        embedding_model=embedding_name,
        results_json=json.dumps([obj.model_dump() for obj in detected_objects]),
        duration_ms=duration_ms,
    )
    db.add(log_entry)
    db.commit()

    return RecognitionResponse(
        query_image_name=query_image_name,
        query_image_url=f"/api/v1/datasets/images/query/{query_image_name}",
        image_width=img_w,
        image_height=img_h,
        detector_model=detector_name,
        embedding_model=embedding_name,
        device_used=env_info["active_device"],
        detection_threshold=confidence_threshold,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        objects_detected_count=len(detected_objects),
        has_fallback_detection=is_fallback,
        detected_objects=detected_objects,
        duration_ms=duration_ms,
        warnings=warnings,
    )


@router.post("/predict-frame", response_model=RecognitionResponse)
async def recognize_webcam_frame(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(DEFAULT_DETECTION_CONFIDENCE_THRESHOLD),
    similarity_threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD),
    top_k: int = Form(DEFAULT_TOP_K),
    margin_threshold: float = Form(DEFAULT_MARGIN_THRESHOLD),
    min_crop_size: int = Form(10),
    allow_fallback: bool = Form(False),
    frame_seq: Optional[int] = Form(None),
    db: Session = Depends(get_db),
):
    """
    SECTION 3: Real-Time Webcam Frame Inference Endpoint.
    Optimized for high-frequency live frames:
    - Does NOT save persistent query images or spam database recognition_logs.
    - Uses temporary frame for localization and crop embedding.
    - Default allow_fallback=False so empty frames return 0 detections and clear canvas.
    - Returns active model version, measured duration_ms, and typed bounding boxes.
    """
    start_time = time.time()
    env_info = get_system_environment_info()

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Webcam frame payload is empty.")

    try:
        pil_frame = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_w, img_h = pil_frame.size
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or invalid webcam frame: {str(e)}",
        )

    (
        detected_objects,
        overall_status,
        top_overall_sku,
        top_overall_sim,
        is_fallback,
        warnings,
        detector_name,
        embedding_name,
    ) = execute_recognition_pipeline(
        db=db,
        image_path=pil_frame,
        confidence_threshold=confidence_threshold,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        margin_threshold=margin_threshold,
        min_crop_size=min_crop_size,
        allow_fallback=allow_fallback,
        save_crops=False,
    )

    duration_ms = round((time.time() - start_time) * 1000, 2)

    # Phase D (Rule 1, Rule 2, Rule 3): Strict Bounding Box Filtering for Camera Recognition
    # Display and return bounding boxes ONLY when BOTH conditions are satisfied:
    # 1. detection_confidence >= configured_detection_threshold
    # 2. sku_similarity >= configured_similarity_threshold
    # If either fails, do NOT display bounding box, do NOT return SKU label, and do NOT return UNKNOWN.
    filtered_objects = [
        obj for obj in detected_objects
        if not obj.is_unknown
        and obj.predicted_sku
        and obj.predicted_sku not in ("UNKNOWN", "AMBIGUOUS")
        and obj.detector_confidence >= confidence_threshold
        and (obj.top_similarity_score or 0.0) >= similarity_threshold
    ]

    return RecognitionResponse(
        query_image_name=f"webcam_frame_{frame_seq or uuid.uuid4().hex[:8]}",
        query_image_url="",
        image_width=img_w,
        image_height=img_h,
        detector_model=detector_name,
        embedding_model=embedding_name,
        device_used=env_info["active_device"],
        detection_threshold=confidence_threshold,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        objects_detected_count=len(filtered_objects),
        has_fallback_detection=is_fallback,
        detected_objects=filtered_objects,
        duration_ms=duration_ms,
        frame_seq=frame_seq,
        warnings=warnings,
    )



@router.get("/history", response_model=List[RecognitionLogSummary])
def get_recognition_history(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Retrieves recent recognition evaluation runs."""
    logs = db.query(RecognitionLog).order_by(RecognitionLog.created_at.desc()).limit(limit).all()
    return logs


@router.get("/history/{log_id}")
def get_recognition_log_detail(log_id: int, db: Session = Depends(get_db)):
    """Retrieves details of a specific recognition run."""
    log = db.query(RecognitionLog).filter(RecognitionLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log entry not found.")
    return {
        "id": log.id,
        "query_image_name": log.query_image_name,
        "num_detections": log.num_detections,
        "final_status": log.final_status,
        "top_predicted_sku": log.top_predicted_sku,
        "top_similarity": log.top_similarity,
        "detector_model": log.detector_model,
        "embedding_model": log.embedding_model,
        "duration_ms": log.duration_ms,
        "created_at": log.created_at,
        "results": json.loads(log.results_json),
    }
