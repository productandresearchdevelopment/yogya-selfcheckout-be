import os
import sys
import time
import json
import psutil
import torch
import numpy as np
from PIL import Image

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.database import SessionLocal
from app.models.image import ProductImage
from app.models.sku import SKU
from app.services.detector import BaselineDetector
from app.services.embedder import BaselineEmbedder
from app.services.gallery import GalleryService
from app.services.model_registry_service import get_active_model
from app.api.recognition import execute_recognition_pipeline


def run_benchmark():
    process = psutil.Process()
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)

    print("=================================================================")
    print(" SECTION 5 — PERFORMANCE BENCHMARKING (BASELINE AUDIT)")
    print("=================================================================")

    # 1. Hardware & Software Environment
    cpu_count = os.cpu_count()
    torch_version = torch.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)

    print(f"Host Processor    : {cpu_count} CPU Cores")
    print(f"System RAM        : {ram_total_gb:.2f} GB Total ({ram_avail_gb:.2f} GB Available)")
    print(f"PyTorch Runtime   : {torch_version} on {device.upper()}")
    print(f"Process RAM Start : {ram_initial_mb:.2f} MB")
    print("-----------------------------------------------------------------")

    db = SessionLocal()
    active_emb = get_active_model(db, model_type="embedding")
    active_det = get_active_model(db, model_type="detector")
    print(f"Active Models     : Embedder={active_emb.model_id} ({active_emb.embedding_dim}-dim) | Detector={active_det.model_id}")

    # Find a test image in storage for benchmarking
    test_img_rec = db.query(ProductImage).filter(ProductImage.is_valid == True).first()
    if not test_img_rec or not os.path.exists(test_img_rec.file_path):
        print("Creating synthetic benchmark image...")
        test_img_path = "benchmark_test_frame.jpg"
        dummy = Image.new("RGB", (640, 480), color=(120, 160, 200))
        dummy.save(test_img_path)
    else:
        test_img_path = test_img_rec.file_path

    with Image.open(test_img_path) as im:
        img_w, img_h = im.size
    print(f"Benchmark Query   : {os.path.basename(test_img_path)} ({img_w}x{img_h})")

    # 2. Measure Model Initialization Latency
    t0 = time.time()
    detector = BaselineDetector()
    detector_init_ms = (time.time() - t0) * 1000

    t0 = time.time()
    embedder = BaselineEmbedder(model_id=active_emb.model_id, db_session=db)
    embedder_init_ms = (time.time() - t0) * 1000
    gallery_service = GalleryService()

    print(f"Detector Init Time: {detector_init_ms:.2f} ms")
    print(f"Embedder Init Time: {embedder_init_ms:.2f} ms")

    # 3. Detector Inference Latency (Warm-up vs Steady-State)
    detector_times = []
    # Warm-up run
    t0 = time.time()
    _ = detector.detect_and_crop(test_img_path, confidence_threshold=0.30)
    det_warmup_ms = (time.time() - t0) * 1000

    for _ in range(5):
        t0 = time.time()
        dets, _, _ = detector.detect_and_crop(test_img_path, confidence_threshold=0.30)
        detector_times.append((time.time() - t0) * 1000)

    det_mean_ms = float(np.mean(detector_times))
    det_median_ms = float(np.median(detector_times))
    det_p95_ms = float(np.percentile(detector_times, 95))

    print(f"\nDetector Latency  : Warm-up={det_warmup_ms:.2f} ms | Mean={det_mean_ms:.2f} ms | Median={det_median_ms:.2f} ms | P95={det_p95_ms:.2f} ms")
    print(f"Detections Found  : {len(dets)} object(s)")

    # 4. Visual Embedding Inference Latency
    sample_crop = Image.new("RGB", (224, 224), color=(200, 150, 100))
    # Warm-up
    t0 = time.time()
    _ = embedder.extract_embedding_from_crop(sample_crop)
    emb_warmup_ms = (time.time() - t0) * 1000

    emb_single_times = []
    for _ in range(10):
        t0 = time.time()
        _ = embedder.extract_embedding_from_crop(sample_crop)
        emb_single_times.append((time.time() - t0) * 1000)

    emb_single_mean_ms = float(np.mean(emb_single_times))
    print(f"Embedding Latency : Warm-up={emb_warmup_ms:.2f} ms | Single Crop Mean={emb_single_mean_ms:.2f} ms")

    # 5. Multi-Crop Sequential Embedding Latency (4 crops sequential)
    multi_seq_times = []
    dummy_crops = [Image.new("RGB", (224, 224), color=(i * 40, 100, 150)) for i in range(4)]
    for _ in range(5):
        t0 = time.time()
        for c in dummy_crops:
            _ = embedder.extract_embedding_from_crop(c)
        multi_seq_times.append((time.time() - t0) * 1000)
    emb_4crop_seq_mean_ms = float(np.mean(multi_seq_times))
    print(f"4 Crops Sequential: {emb_4crop_seq_mean_ms:.2f} ms (~{emb_4crop_seq_mean_ms / 4:.2f} ms/crop)")

    # 6. Gallery Search Latency (Current Uncached Path)
    dummy_vec = np.random.randn(active_emb.embedding_dim).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    search_times = []
    for _ in range(10):
        t0 = time.time()
        _ = gallery_service.search_candidates(db, query_vector=dummy_vec, similarity_threshold=0.50, top_k=5)
        search_times.append((time.time() - t0) * 1000)
    search_mean_ms = float(np.mean(search_times))
    print(f"Gallery Search (DB): Mean={search_mean_ms:.2f} ms over 10 queries")

    # 7. End-to-End Pipeline Latency
    e2e_times = []
    for _ in range(3):
        t0 = time.time()
        _ = execute_recognition_pipeline(
            db,
            image_path=test_img_path,
            confidence_threshold=0.30,
            similarity_threshold=0.50,
            top_k=5,
        )
        e2e_times.append((time.time() - t0) * 1000)
    e2e_mean_ms = float(np.mean(e2e_times))
    sustained_fps = 1000.0 / e2e_mean_ms if e2e_mean_ms > 0 else 0.0
    print(f"End-to-End Latency: Mean={e2e_mean_ms:.2f} ms (~{sustained_fps:.2f} FPS)")

    ram_final_mb = process.memory_info().rss / (1024 * 1024)
    ram_delta_mb = ram_final_mb - ram_initial_mb
    print(f"Process RAM End   : {ram_final_mb:.2f} MB (Delta: +{ram_delta_mb:.2f} MB)")
    print("=================================================================\n")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu_cores": cpu_count,
            "ram_total_gb": round(ram_total_gb, 2),
            "ram_available_gb": round(ram_avail_gb, 2),
            "device": device,
            "torch_version": torch_version,
        },
        "models": {
            "detector": active_det.model_id,
            "embedder": active_emb.model_id,
            "embedding_dim": active_emb.embedding_dim,
        },
        "benchmarks": {
            "detector_init_ms": round(detector_init_ms, 2),
            "embedder_init_ms": round(embedder_init_ms, 2),
            "detector_warmup_ms": round(det_warmup_ms, 2),
            "detector_mean_ms": round(det_mean_ms, 2),
            "detector_median_ms": round(det_median_ms, 2),
            "detector_p95_ms": round(det_p95_ms, 2),
            "embedding_warmup_ms": round(emb_warmup_ms, 2),
            "embedding_single_mean_ms": round(emb_single_mean_ms, 2),
            "embedding_4crop_sequential_mean_ms": round(emb_4crop_seq_mean_ms, 2),
            "gallery_search_db_mean_ms": round(search_mean_ms, 2),
            "e2e_pipeline_mean_ms": round(e2e_mean_ms, 2),
            "sustained_fps": round(sustained_fps, 2),
            "ram_initial_mb": round(ram_initial_mb, 2),
            "ram_final_mb": round(ram_final_mb, 2),
            "ram_delta_mb": round(ram_delta_mb, 2),
        },
    }

    output_path = os.path.join(os.path.dirname(__file__), "..", "storage", "benchmark_baseline.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline benchmark saved to: {output_path}")

    db.close()
    return results


if __name__ == "__main__":
    run_benchmark()
