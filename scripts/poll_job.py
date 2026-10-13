import urllib.request
import json
import time
import sys

job_id = sys.argv[1] if len(sys.argv) > 1 else "job_20261012_081448_843314"

for i in range(12):
    req = urllib.request.urlopen(f"http://127.0.0.1:8000/api/v1/training/jobs/{job_id}")
    d = json.loads(req.read().decode())
    print(f"[{i+1}/12] Status: {d['status']} | Epoch: {d['current_epoch']}/{d['total_epochs']} | Train Loss: {d['train_loss']} | Val Loss: {d['val_loss']} | Val Top-1: {d['val_accuracy']}")
    if d['status'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
        print("\nFinal Job Logs:")
        print(d.get("logs", ""))
        print("\nCheckpoint Path:", d.get("checkpoint_path"))
        break
    time.sleep(2)
