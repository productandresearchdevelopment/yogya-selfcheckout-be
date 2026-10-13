import urllib.request
import json

model_id = "mobilenetv3-metric-job_20261012_081448_843314"
print(f"Activating model: {model_id}...")

post_req = urllib.request.Request(
    f"http://127.0.0.1:8000/api/v1/models/{model_id}/activate",
    data=b"{}",
    headers={"Content-Type": "application/json"}
)

try:
    res = urllib.request.urlopen(post_req)
    data = json.loads(res.read().decode())
    print("Activation Success:")
    print(json.dumps(data, indent=2))
except urllib.error.HTTPError as e:
    print("HTTP Error:", e.code, e.read().decode())
