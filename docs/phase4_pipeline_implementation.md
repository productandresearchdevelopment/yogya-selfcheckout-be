# Phase 4 — Integrated SKU Recognition Pipeline Implementation & Camera Testing Guide

## Executive Summary

Phase 4 completes and unites the five foundational processes of the **YOGYA Self-Checkout SKU Recognition System** into a single operational, reproducible pipeline:
1. **Object Detection:** Faster R-CNN MobileNetV3-Large-FPN with Non-Maximum Suppression (NMS), crop provenance tagging, and adaptive foreground saliency fallback.
2. **Few-Shot Metric Learning:** Backbone + 2-layer MLP projection head with identity-aware triplet/margin loss, checkpoint state persistence, and non-destructive fine-tuning safeguards.
3. **Embedding Extraction:** DINOv2 ViT-S/14 (384-d) foundation model candidate alongside MobileNetV3 (960-d) baseline, with strict L2-normalization and vector compatibility gates.
4. **Cosine Similarity Gallery Matching:** High-throughput in-memory BLAS matrix multiplication with multi-reference SKU aggregation, top-1/top-2 score tracking, and automated cache invalidation.
5. **Open-Set Recognition:** Three-tier decision layer (`KNOWN`, `UNKNOWN`, `AMBIGUOUS`) enforcing both absolute similarity thresholds ($\tau_{\text{sim}}$) and top-1/top-2 margin separation ($\tau_{\text{margin}}$).

All 47 tests across the system test suite pass (100% pass rate). Existing SKU records, reference images, embeddings, and baseline configurations are strictly preserved.

---

## 1. Final Integrated Architecture

```mermaid
flowchart TD
    Frame[Live Camera Frame / Uploaded Image] --> Detector[Faster R-CNN MobileNetV3-Large-FPN]
    Detector --> NMS[Non-Maximum Suppression (IoU=0.50)]
    NMS --> CheckDet{Detections Found?}
    
    CheckDet -- Yes --> Crop[Crop Product ROI (crop_provenance='detector_roi')]
    CheckDet -- No (Fallback) --> Saliency{Adaptive Foreground Saliency?}
    
    Saliency -- High Contrast ROI --> CropSal[Crop Foreground (crop_provenance='adaptive_saliency_roi')]
    Saliency -- No Contrast --> CropFull[Fallback Frame (crop_provenance='fallback_full_frame')]
    
    Crop --> Preprocess[Resize & ImageNet Preprocessing]
    CropSal --> Preprocess
    CropFull --> Preprocess
    
    Preprocess --> Embedder[Active Embedder: DINOv2 / MobileNetV3]
    Embedder --> L2Norm[L2-Normalization: ||v|| = 1.0]
    
    L2Norm --> GallerySearch[BLAS Cosine Search against In-Memory Gallery]
    GallerySearch --> Top2[Compute Top-1 & Top-2 Candidates, Scores & Margin Delta]
    
    Top2 --> Gate1{Score >= Sim Threshold (0.60)?}
    Gate1 -- No --> StatusUnknown[UNKNOWN: Low Similarity Candidate]
    Gate1 -- Yes --> Gate2{Margin Delta >= Margin Threshold (0.08)?}
    Gate2 -- No --> StatusAmbiguous[AMBIGUOUS: Confusable Candidates]
    Gate2 -- Yes --> StatusKnown[KNOWN: Accepted SKU Match]
    
    StatusKnown --> Cart[Add SKU to Self-Checkout Cart]
    StatusUnknown --> PromptManual[Reject: Prompt Customer to Reposition or Enter Barcode]
    StatusAmbiguous --> PromptManual
```

---

## 2. Actual Files Changed & Created

| File | Type | Changes & Functional Rationale |
| :--- | :--- | :--- |
| `backend/app/services/detector.py` | Modified | Added Torchvision NMS post-processing (`apply_nms`, IoU=0.50); introduced `crop_provenance` tracking (`detector_roi`, `adaptive_saliency_roi`, `fallback_full_frame`); implemented Otsu adaptive saliency box localization; added `RetailDetectionDataset` & `check_retail_annotation_readiness` infrastructure. |
| `backend/app/services/gallery.py` | Modified | Updated candidate scoring to return explicit three-tier decisions: `AMBIGUOUS` (when $\text{sim} \ge \tau_{\text{sim}}$ but $\Delta < \tau_{\text{margin}}$) vs `UNKNOWN` (when $\text{sim} < \tau_{\text{sim}}$); kept full diagnostics (`margin_delta`, `top_candidates`). |
| `backend/app/schemas/recognition.py` | Modified | Added `crop_provenance` field to `DetectionBox` and `RecognitionCandidate`; expanded `recognition_status` to `"KNOWN" \| "UNKNOWN" \| "AMBIGUOUS" \| "NO_OBJECTS_DETECTED"`. |
| `backend/app/api/recognition.py` | Modified | Integrated three-tier open-set handling; returns `overall_status = "NO_OBJECTS_DETECTED"` on empty frames; excludes both `UNKNOWN` and `AMBIGUOUS` from live checkout cart additions. |
| `backend/app/api/skus.py` | Modified | Imported and invoked `invalidate_gallery_cache()` across SKU mutations (`enroll-few-shot`, `deactivate`, `delete`, `update`), ensuring instant gallery discovery without service restarts. |
| `frontend/src/lib/api.ts` | Modified | Updated TypeScript interface `DetectedObject` with `crop_provenance` and status union types (`"KNOWN" \| "UNKNOWN" \| "AMBIGUOUS" \| "NO_OBJECTS_DETECTED"`). |
| `backend/tests/test_phase4_integrated_pipeline.py` | Created | Added 8 comprehensive integration tests validating NMS, crop provenance, open-set rejections, metric-learning parameters, annotation readiness, few-shot registration, and model rollback safety. |
| `backend/tests/test_phase2_dino_recognition.py` | Modified | Synchronized assertion in `test_open_set_recognition_and_top_two_margin_gate` to accept both `UNKNOWN` and `AMBIGUOUS` per Phase 4 three-tier contract. |
| `backend/docs/phase4_pipeline_implementation.md` | Created | Authoritative pipeline implementation specification and live camera testing guide. |

---

## 3. Implementation of the Five Core Methods

### Method 1 — Object Detection
* **Model:** Faster R-CNN MobileNetV3-Large-FPN pretrained on COCO-80 (`fasterrcnn-pretrained-baseline`).
* **Non-Maximum Suppression (NMS):** To prevent duplicate overlapping bounding boxes on the same supermarket item, raw Faster R-CNN proposals are passed through `torchvision.ops.nms` with an IoU threshold of 0.50.
* **Crop Provenance Tracking:** Every detected box is tagged with `crop_provenance`:
  * `"detector_roi"`: High-confidence detection proposal produced by Faster R-CNN.
  * `"adaptive_saliency_roi"`: Localized foreground region computed via adaptive contrast & Otsu thresholding when COCO classes fail to identify retail packaging.
  * `"fallback_full_frame"`: Center crop of the original frame when neither method detects clear foreground products.
* **Diagnostics:** Both `confidence` and `crop_provenance` are surfaced in the API response and inspection UI.
* **Retail Fine-Tuning Infrastructure:** The `RetailDetectionDataset` class and `check_retail_annotation_readiness()` service verify the presence of retail bounding boxes in Pascal VOC or COCO JSON formats without fabricating training data.

### Method 2 — Few-Shot Metric Learning
* **Architecture:** Pretrained visual backbone (DINOv2 ViT-S or MobileNetV3) followed by a 2-layer non-linear projection head:
  $$\text{Input}(D_{\text{backbone}}) \longrightarrow \text{Linear}(D_{\text{backbone}} \to 512) \longrightarrow \text{ReLU} \longrightarrow \text{Linear}(512 \to D_{\text{proj}}) \longrightarrow \text{L2-Norm}$$
* **Loss Function:** Multi-class Triplet Margin Loss with identity-aware batch sampling ($P=4$ classes, $K=2$ instances per batch).
* **Gradient Verification:** Automated verification confirms that projection head parameters (`fc1.weight`, `fc2.weight`) update during training while keeping the foundational backbone frozen.
* **Safe Checkpoint Management:** Checkpoint metadata explicitly records `backbone_model`, `embedding_dim`, `input_size`, `model_version`, and weights. Checkpoints are never automatically activated into production without evaluation and explicit user promotion.

### Method 3 — Embedding Extraction
* **Foundation Models:**
  * **Active Baseline:** `mobilenetv3-pretrained-baseline` (960 dimensions).
  * **DINOv2 ViT-S/14:** `dinov2-vits14-baseline` (384 dimensions), loaded via PyTorch Hub (`facebookresearch/dinov2`).
  * **DINOv3:** Gated/license-restricted status is preserved; attempting to load DINOv3 without verified weights cleanly raises an informative runtime exception without breaking the pipeline.
* **Normalization:** All generated vectors are strictly L2-normalized ($||\mathbf{v}||_2 = 1.0$), ensuring Euclidean distance directly maps to cosine similarity:
  $$||\mathbf{u} - \mathbf{v}||_2^2 = 2(1 - \cos(\mathbf{u}, \mathbf{v}))$$
* **Dimension Guard:** Strict dimension guards prevent comparing or combining vectors from incompatible models (e.g., 384-d vs 960-d).

### Method 4 — Cosine Similarity Gallery Matching
* **Multi-Reference Aggregation:** Each SKU can have multiple reference images (1 to 10+). Gallery matrices stack all reference vectors. For a given SKU with reference vectors $\{\mathbf{r}_1, \dots, \mathbf{r}_k\}$, the similarity score is aggregated via maximum similarity:
  $$\text{Score}(\mathbf{q}, \text{SKU}_i) = \max_{j} \mathbf{q}^T \mathbf{r}_{i,j}$$
* **Top-1 / Top-2 Tracking:** Search returns the top candidate, the second candidate, their respective similarity scores, and the top-1/top-2 margin delta:
  $$\Delta_{\text{margin}} = \text{Score}_{\text{top1}} - \text{Score}_{\text{top2}}$$
* **BLAS-Accelerated In-Memory Cache:** In-memory normalized gallery matrix enables $O(N)$ dot products computed in sub-millisecond latency (<0.5ms for 300 reference vectors).
* **Automated Cache Invalidation:** Any mutation to SKUs (few-shot enrollment, deactivation, deletion) automatically triggers `invalidate_gallery_cache()`, reloading the matrix on the next query.

### Method 5 — Open-Set Recognition
The decision layer prevents unseen or ambiguous products from polluting the self-checkout cart:

$$\text{Decision} = \begin{cases} 
\text{KNOWN (Accepted)}, & \text{if } \text{Score}_{\text{top1}} \ge \tau_{\text{sim}} \text{ and } \Delta_{\text{margin}} \ge \tau_{\text{margin}} \\
\text{AMBIGUOUS (Rejected)}, & \text{if } \text{Score}_{\text{top1}} \ge \tau_{\text{sim}} \text{ and } \Delta_{\text{margin}} < \tau_{\text{margin}} \\
\text{UNKNOWN (Rejected)}, & \text{if } \text{Score}_{\text{top1}} < \tau_{\text{sim}}
\end{cases}$$

* **Configurable Thresholds:** Configured via system settings with operational baselines:
  * Default $\tau_{\text{sim}} = 0.60$
  * Default $\tau_{\text{margin}} = 0.08$
* **Cart Safety Guard:** In `recognition.py`, cart additions are guarded:
  ```python
  if candidate.recognition_status == "KNOWN" and not candidate.is_unknown and not candidate.is_ambiguous:
      cart_items.append(...)
  ```
  Neither `UNKNOWN` nor `AMBIGUOUS` predictions can ever be added to the customer's cart.

---

## 4. End-to-End Inference Flow

```
Camera Stream (Browser)
   │ (WebSocket / HTTP POST /api/v1/recognition/predict-frame)
   ▼
1. Decode Base64 JPEG frame into PIL RGB Image
   │
   ▼
2. BaselineDetector.detect_and_crop(frame)
   ├─ Faster R-CNN forward pass
   ├─ NMS (IoU=0.50) filtering
   ├─ If detections found: crop highest-confidence bbox (crop_provenance="detector_roi")
   └─ If empty: fallback to adaptive saliency or full frame
   │
   ▼
3. BaselineEmbedder.extract_embedding_from_crop(crop_img)
   ├─ Resize to 224x224 & Normalize with ImageNet mean/std
   ├─ ViT / CNN backbone forward pass
   └─ L2-normalize output vector (||v|| = 1.0)
   │
   ▼
4. GalleryService.search_candidates(query_vector)
   ├─ In-memory BLAS dot product: S = G · q
   ├─ Group by SKU & compute max reference score
   ├─ Sort top candidates: candidate[0] (top-1) and candidate[1] (top-2)
   └─ Calculate margin_delta = score[0] - score[1]
   │
   ▼
5. Open-Set Decision Layer
   ├─ Check similarity >= 0.60
   ├─ Check margin_delta >= 0.08
   └─ Assign KNOWN, UNKNOWN, or AMBIGUOUS status
   │
   ▼
6. Response Serialization & Frontend State
   ├─ Return recognition JSON with bounding box, provenance, candidate scores, and status
   └─ Frontend renders visual bounding box and updates cart ONLY if status == "KNOWN"
```

---

## 5. Enrollment & Gallery Refresh Behavior

SKU enrollment is supported through two production workflows:
1. **Webcam Image Capture:** Multiple frames captured in real time via the browser camera.
2. **File Upload:** High-resolution product images uploaded via form data.

### Transactional Enrollment Steps (`POST /api/v1/skus/enroll-few-shot`)
1. Create SKU database record (`sku_code`, `name`, `category`).
2. Persist reference image files to disk with SHA-256 deduplication and quality inspection (blur, brightness, contrast).
3. Extract reference embedding vectors using the currently **active** model.
4. Verify embedding dimension matches the active model dimension.
5. Commit SKU, ProductImage, and Embedding records in a single database transaction.
6. Call `invalidate_gallery_cache()`.
7. **Immediate Result:** The newly registered SKU is instantly searchable in subsequent recognition requests without server restart or catalog retraining.

---

## 6. Test Suite Execution & Verification

Comprehensive test suites were executed against the PostgreSQL runtime database:

```powershell
backend\.venv\Scripts\pytest backend\tests\ -v
```

### Complete Test Results Summary (47 of 47 Passed — 100%)

| Test Module | Tests | Result | Features Validated |
| :--- | :---: | :---: | :--- |
| `test_phase4_integrated_pipeline.py` | 8 | **8 PASSED** | Detector NMS, crop provenance, empty frame handling, open-set 3-tier decisions, metric-learning parameter update, retail annotation readiness checker, few-shot registration without retraining, model rollback integrity. |
| `test_phase2_dino_recognition.py` | 7 | **7 PASSED** | DINOv2 loading, DINOv3 blocker reporting, metric-learning training loop, fine-grained discrimination, top-2 margin gate, model activation & rollback, latency benchmarks. |
| `test_recognition_pipeline.py` | 4 | **4 PASSED** | Embedder L2 normalization, detector fallback, end-to-end predict pipeline, system health & stats. |
| `test_section2_metric_learning.py` | 7 | **7 PASSED** | Metric loss functions, identity-aware sampler, dataset audit, model registry preservation, cross-version compatibility guard, enrollment isolation. |
| `test_section3_webcam_enrollment.py` | 4 | **4 PASSED** | Recognition status endpoint, few-shot enrollment API, webcam predict frame pipeline, fallback handling. |
| `test_section4_evaluation_hard_negatives.py` | 8 | **8 PASSED** | Data leakage prevention, hard negative mining, error analysis, threshold calibration reproducibility, comparative benchmarks, promotion safeguards. |
| `test_section5_production_readiness.py` | 7 | **7 PASSED** | System readiness, resource monitoring, BLAS gallery search, backup creation & SHA-256 manifest, tamper detection, dry-run sandbox restore, artifact cleanup. |
| `test_skus_and_datasets.py` | 2 | **2 PASSED** | SKU CRUD operations, dataset upload quality inspection, duplicate hash prevention. |
| **Total** | **47** | **47 PASSED** | **100% Pass Rate** |

---

## 7. Data Preservation & Rollback Confirmation

All existing catalog data and checkpoints are strictly preserved:
* **SKU Records:** 117 SKUs intact in PostgreSQL database `yogya-c50`.
* **Product Images:** 321 product images verified on disk with valid SHA-256 hashes.
* **Embeddings:** 273 indexed embeddings intact.
* **Active Model Baseline:** `mobilenetv3-pretrained-baseline` (960-d) preserved as active baseline.
* **Candidate Model:** `dinov2-vits14-baseline` (384-d) verified ready for one-click activation.
* **Rollback Verification:** Tested activating DINOv2 and rolling back to MobileNetV3; confirmed zero data loss and zero vector mixing.

---

## 8. Known Limitations & Remaining Retail Detector Blocker

1. **COCO Pretrained Detector Limitations:**
   * The Faster R-CNN MobileNetV3-Large-FPN detector was pretrained on COCO-80 classes (bottles, cups, bananas, etc.).
   * While it reliably detects beverage bottles and canned goods, it struggles with flexible snack bags, boxed tea, and flat packaging.
   * **Mitigation Implemented in Phase 4:** The new `adaptive_saliency_roi` fallback automatically detects high-contrast foreground objects on the checkout surface when COCO object confidence is low.
2. **Retail Training Data Dependency:**
   * Full fine-tuning of the detector requires **50–100 supermarket scene images with retail bounding box annotations** (Pascal VOC XML or COCO JSON format).
   * The automated readiness checker (`check_retail_annotation_readiness()`) confirms that annotations are not yet present in `backend/storage/datasets/retail_detection/`.
   * **No synthetic or fabricated annotations were introduced.** The infrastructure is complete and will immediately train when real retail annotations are supplied.

---

## 9. Live Camera Testing Instructions

Follow these step-by-step instructions to test the complete pipeline with a live webcam:

### Step 1: Start the Backend Service
Open a PowerShell terminal in the repository root:
```powershell
cd backend
.\.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Verify the backend is live by opening [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

### Step 2: Start the Frontend Application
Open a second PowerShell terminal in the repository root:
```powershell
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in Google Chrome or Microsoft Edge.

### Step 3: Test Real-Time Self-Checkout Recognition
1. Navigate to the **"Self-Checkout"** tab on the navigation bar.
2. Allow browser camera permissions when prompted.
3. Place a registered supermarket product (e.g., Ultra Milk, Teh Pucuk, Aqua) in front of the camera:
   * **Bounding Box:** A bounding box will track the product in real time.
   * **Crop Provenance:** The display will show whether the product was localized via `detector_roi` or `adaptive_saliency_roi`.
   * **Top Candidates:** The top-1 and top-2 candidate SKUs and their similarity scores are displayed.
   * **Acceptance:** If similarity $\ge 0.60$ and margin $\ge 0.08$, the status shows `KNOWN (Green)` and the item is added to the checkout cart.
4. Test Open-Set Rejection:
   * Hold an **unregistered** item (e.g., keys, wallet, or phone) in front of the camera:
   * The system will reject it with status `UNKNOWN (Red)` (similarity $< 0.60$) and will **not** add it to the cart.
   * Hold two visually confusable items: if margin $< 0.08$, the system flags `AMBIGUOUS (Yellow)` and prompts the customer to adjust item orientation.

### Step 4: Test Few-Shot Webcam Enrollment
1. Navigate to the **"SKU Enrollment"** tab.
2. Fill in the SKU Code (e.g., `TEST-KOPI-01`), Product Name, and Category.
3. Click **"Capture from Webcam"** to take 3–5 angles of the new product.
4. Click **"Enroll SKU"**.
5. Return to the **Self-Checkout** screen and hold the new item in front of the camera:
   * It will be recognized immediately without restarting the backend or retraining the model.

---

## 10. Remaining Work Before Production Deployment

1. **Annotate Retail Checkout Scenes:** Collect 50–100 representative webcam images of packaged products on the self-checkout counter and label bounding boxes for retail detector fine-tuning.
2. **Live Camera Threshold Calibration:** Calibrate $\tau_{\text{sim}}$ and $\tau_{\text{margin}}$ using live camera captures under store ambient lighting.
3. **Model Promotion to DINOv2:** Run the automated promotion benchmark (`POST /api/v1/evaluation/benchmark`) to evaluate promoting `dinov2-vits14-baseline` to active production status based on live camera validation data.
