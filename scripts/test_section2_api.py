import urllib.request
import urllib.error
import json

def test_api():
    print("--- 1. Testing Training Readiness ---")
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/v1/training/readiness")
    data = json.loads(req.read().decode())
    print("Status:", data["status"])
    print("Is Ready:", data["is_ready"])
    print("Blockers:", data["blockers"])

    print("\n--- 2. Testing Training Job Rejection on Insufficient Data ---")
    post_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/training/jobs",
        data=json.dumps({"epochs": 3}).encode(),
        headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(post_req)
        print("UNEXPECTED: Training job started when dataset was insufficient!")
    except urllib.error.HTTPError as e:
        print("Expected HTTP Status:", e.code)
        err = json.loads(e.read().decode())
        print("Error response:", err)

    print("\n--- 3. Testing Seed Dataset Samples ---")
    seed_req = urllib.request.Request("http://127.0.0.1:8000/api/v1/training/seed-samples", data=b"{}", headers={"Content-Type": "application/json"})
    seed_res = json.loads(urllib.request.urlopen(seed_req).read().decode())
    print("Seed result:", seed_res)

    print("\n--- 4. Re-checking Readiness after Seeding ---")
    req2 = urllib.request.urlopen("http://127.0.0.1:8000/api/v1/training/readiness")
    data2 = json.loads(req2.read().decode())
    print("Status:", data2["status"])
    print("Is Ready:", data2["is_ready"])
    print("Summary:", data2["summary"])

    print("\n--- 5. Starting Real Metric Learning Training Job ---")
    post_req2 = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/training/jobs",
        data=json.dumps({"epochs": 3, "embedding_dim": 512, "learning_rate": 0.0001, "margin": 0.2}).encode(),
        headers={"Content-Type": "application/json"}
    )
    job_res = json.loads(urllib.request.urlopen(post_req2).read().decode())
    print("Started Job:", job_res["job_id"], "Status:", job_res["status"])
    return job_res["job_id"]

if __name__ == "__main__":
    job_id = test_api()
