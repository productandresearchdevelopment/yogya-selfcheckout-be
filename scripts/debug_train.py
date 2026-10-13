import traceback
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.metric_learning import _run_training_pipeline
from app.core.database import SessionLocal
from app.models.training_job import TrainingJob

job_id = "job_20261012_081448_843314"
config = {"epochs": 3, "embedding_dim": 512, "learning_rate": 0.0001, "margin": 0.2, "p_classes": 2, "k_instances": 2}

try:
    print("Running pipeline directly...")
    _run_training_pipeline(job_id, config, SessionLocal)
    print("Finished pipeline successfully!")
except Exception as e:
    traceback.print_exc()
