import io
import httpx
from PIL import Image, ImageDraw

BASE_URL = "http://127.0.0.1:8000/api/v1"
client = httpx.Client(timeout=30.0)

def create_sample_box_image(label_text, box_color, size=(400, 400)):
    img = Image.new("RGB", size, color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Draw product package rectangle
    draw.rectangle([80, 80, 320, 320], fill=box_color, outline=(20, 20, 20), width=4)
    # Draw bottle/can shape
    draw.ellipse([140, 100, 260, 160], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.text((120, 200), label_text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    return buf

print("=" * 60)
print("1. Testing Health Check Endpoint...")
health_res = client.get(f"{BASE_URL}/system/health")
assert health_res.status_code == 200, f"Health check failed: {health_res.text}"
health_data = health_res.json()
print(f"Health Status: {health_data['status']} | Active Device: {health_data['device']}")

print("\n2. Registering SKU 'SKU-INDOMIE-MIE'...")
sku_payload = {
    "sku_code": "SKU-INDOMIE-MIE",
    "name": "Indomie Mi Goreng Special 85g",
    "category": "Instant Noodles",
    "description": "Indonesian instant fried noodles packet"
}
sku_res = client.post(f"{BASE_URL}/skus", json=sku_payload)
if sku_res.status_code == 409:
    print("SKU already exists, fetching existing...")
    skus_list = client.get(f"{BASE_URL}/skus").json()
    sku_obj = next(s for s in skus_list if s["sku_code"] == "SKU-INDOMIE-MIE")
    sku_id = sku_obj["id"]
else:
    assert sku_res.status_code == 201, f"SKU creation failed: {sku_res.text}"
    sku_id = sku_res.json()["id"]
print(f"SKU ID: {sku_id}")

print("\n3. Uploading Reference Image for SKU-INDOMIE-MIE...")
ref_buf = create_sample_box_image("INDOMIE", (180, 40, 30))
upload_res = client.post(
    f"{BASE_URL}/datasets/upload",
    data={"sku_id": sku_id, "role": "reference", "split": "reference"},
    files={"files": ("indomie_reference.jpg", ref_buf.getvalue(), "image/jpeg")}
)
if upload_res.status_code == 409:
    print("Reference image already exists in dataset.")
else:
    assert upload_res.status_code == 201, f"Upload failed: {upload_res.text}"
    img_record = upload_res.json()[0]
    print(f"Uploaded Image ID: {img_record['id']} | Dimensions: {img_record['width']}x{img_record['height']} | Blur Score: {img_record['blur_score']} | Embedding Indexed: {img_record['has_embedding']}")

print("\n4. Registering Second SKU 'SKU-POKKA-GREENTEA'...")
sku2_payload = {
    "sku_code": "SKU-POKKA-GREENTEA",
    "name": "Pokka Jasmine Green Tea 500ml",
    "category": "Beverages",
    "description": "Green tea bottle"
}
sku2_res = client.post(f"{BASE_URL}/skus", json=sku2_payload)
if sku2_res.status_code == 409:
    skus_list = client.get(f"{BASE_URL}/skus").json()
    sku2_obj = next(s for s in skus_list if s["sku_code"] == "SKU-POKKA-GREENTEA")
    sku2_id = sku2_obj["id"]
else:
    sku2_id = sku2_res.json()["id"]
print(f"SKU 2 ID: {sku2_id}")

print("\n5. Uploading Reference Image for SKU-POKKA-GREENTEA...")
ref_buf2 = create_sample_box_image("GREENTEA", (30, 140, 60))
upload2_res = client.post(
    f"{BASE_URL}/datasets/upload",
    data={"sku_id": sku2_id, "role": "reference", "split": "reference"},
    files={"files": ("greentea_reference.jpg", ref_buf2.getvalue(), "image/jpeg")}
)
if upload2_res.status_code == 201:
    print(f"Uploaded and indexed SKU 2 reference image successfully.")

print("\n6. Running End-to-End Recognition Pipeline with Test Query Image...")
# Create a test query image resembling Indomie
query_buf = create_sample_box_image("INDOMIE", (175, 45, 35))
recog_res = client.post(
    f"{BASE_URL}/recognition/predict",
    data={
        "confidence_threshold": 0.25,
        "similarity_threshold": 0.50,
        "top_k": 3
    },
    files={"file": ("query_indomie_test.jpg", query_buf.getvalue(), "image/jpeg")}
)
assert recog_res.status_code == 200, f"Recognition failed: {recog_res.text}"
recog_data = recog_res.json()

print(f"Duration: {recog_data['duration_ms']} ms")
print(f"Detector Backbone: {recog_data['detector_model']}")
print(f"Embedding Backbone: {recog_data['embedding_model']}")
print(f"Detected Objects Count: {recog_data['objects_detected_count']}")
print(f"Fallback Mode Triggered: {recog_data['has_fallback_detection']}")

for idx, obj in enumerate(recog_data["detected_objects"]):
    print(f"\n--- Detected Object #{idx + 1} ---")
    print(f"  Bounding Box Coords (Normalized): [{obj['x_min']:.4f}, {obj['y_min']:.4f}, {obj['x_max']:.4f}, {obj['y_max']:.4f}]")
    print(f"  Pixel Bounding Box: [x={obj['px_x']}, y={obj['px_y']}, w={obj['px_width']}, h={obj['px_height']}]")
    print(f"  Detector Confidence: {obj['detector_confidence']:.4f} (Localization certainty)")
    print(f"  Top Cosine Similarity: {obj['top_similarity_score']:.4f} (Vector cosine distance)")
    print(f"  Final Decision: {obj['predicted_sku']} ({'UNKNOWN' if obj['is_unknown'] else 'MATCH ACCEPTED'})")
    if obj['is_unknown']:
        print(f"  Rejection Reason: {obj['rejection_reason']}")
    print(f"  Top Candidate Matches:")
    for cand in obj["candidates"]:
        print(f"    Rank #{cand['rank']}: {cand['sku_code']} ({cand['sku_name']}) -> Cosine Sim: {cand['similarity_score']:.4f}")

print("\n7. Testing UNKNOWN Rejection Rule with High Similarity Acceptance Threshold (0.99)...")
unknown_recog_res = client.post(
    f"{BASE_URL}/recognition/predict",
    data={
        "confidence_threshold": 0.25,
        "similarity_threshold": 0.99,
        "top_k": 3
    },
    files={"file": ("query_unknown_test.jpg", query_buf.getvalue(), "image/jpeg")}
)
assert unknown_recog_res.status_code == 200
unknown_recog_data = unknown_recog_res.json()
first_obj = unknown_recog_data["detected_objects"][0]
print(f"  Result with threshold 0.99: {first_obj['predicted_sku']} (is_unknown={first_obj['is_unknown']})")
print(f"  Rejection Reason: {first_obj['rejection_reason']}")
assert first_obj["is_unknown"] is True, "Rejection rule did not reject as UNKNOWN"

print("\n" + "=" * 60)
print(" ALL REAL WORKFLOW VERIFICATIONS PASSED SUCCESSFULLY!")
print("=" * 60)
