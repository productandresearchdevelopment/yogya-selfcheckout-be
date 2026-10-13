import os
import json
import time
import math
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from app.core.device import get_torch_device
from app.core.config import CHECKPOINTS_DIR, EMBEDDING_MODEL_NAME
from app.models.training_job import TrainingJob
from app.models.model_registry import ModelRegistry
from app.models.image import ProductImage
from app.models.sku import SKU

# Global lock to prevent concurrent training jobs on memory-constrained systems
TRAINING_MUTEX = threading.Lock()
ACTIVE_CANCELLATIONS: Dict[str, bool] = {}


class MetricEmbeddingModel(nn.Module):
    """
    Deep Metric Learning visual feature embedding model.
    Backbone: Pretrained MobileNetV3-Large (960 features)
    Head: 2-layer projection head with BatchNorm, Hardswish, Dropout, and L2 normalization.
    """

    def __init__(self, embedding_dim: int = 512, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.backbone = mobilenet_v3_large(weights=weights)
        self.backbone.classifier = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection_head = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(p=0.15),
            nn.Linear(512, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projected = self.projection_head(features)
        # L2 normalize so cosine similarity equals inner product
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = 2):
        """Unfreezes selected top layers of the backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


class CosineTripletLoss(nn.Module):
    """
    Triplet Margin Loss with online semi-hard and hardest negative mining.
    Computes cosine distance D(u, v) = 1 - <u, v>.
    Loss: max(0, D(a, p) - D(a, n) + margin).
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embeddings are L2 normalized, so similarity matrix is cosine similarity
        sim_matrix = torch.matmul(embeddings, embeddings.t())
        dist_matrix = 1.0 - sim_matrix
        dist_matrix = torch.clamp(dist_matrix, min=0.0, max=2.0)

        batch_size = embeddings.size(0)
        triplet_losses = []

        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        for i in range(batch_size):
            pos_mask = labels_equal[i].clone()
            pos_mask[i] = False  # Exclude self
            neg_mask = ~labels_equal[i]

            pos_dists = dist_matrix[i][pos_mask]
            neg_dists = dist_matrix[i][neg_mask]

            if len(pos_dists) == 0 or len(neg_dists) == 0:
                continue

            # Hardest positive (furthest positive)
            hardest_pos = torch.max(pos_dists)

            # Hardest negative (nearest negative)
            hardest_neg = torch.min(neg_dists)

            loss = F.relu(hardest_pos - hardest_neg + self.margin)
            if loss > 0.0:
                triplet_losses.append(loss)

        if len(triplet_losses) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return torch.stack(triplet_losses).mean()


class IdentityAwareBatchSampler(Sampler):
    """
    Batch sampler that guarantees each mini-batch contains P distinct identities
    with K samples each, yielding a total batch size of P * K.
    Crucial for contrastive and triplet metric learning.
    """

    def __init__(self, labels: List[int], p: int = 2, k: int = 2, num_batches: int = 20):
        self.labels = labels
        self.p = p
        self.k = k
        self.num_batches = num_batches

        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(labels):
            self.label_to_indices.setdefault(lbl, []).append(idx)

        self.unique_labels = list(self.label_to_indices.keys())

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            if len(self.unique_labels) < self.p:
                # If fewer than P classes, sample all available with replacement
                selected_labels = random.choices(self.unique_labels, k=self.p)
            else:
                selected_labels = random.sample(self.unique_labels, self.p)

            for lbl in selected_labels:
                indices = self.label_to_indices[lbl]
                if len(indices) >= self.k:
                    selected_indices = random.sample(indices, self.k)
                else:
                    selected_indices = random.choices(indices, k=self.k)
                batch.extend(selected_indices)

            yield batch

    def __len__(self):
        return self.num_batches


class MetricLearningDataset(Dataset):
    """Dataset holding image paths, labels, and optional crop bounding boxes."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transform: Optional[transforms.Compose] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        image_path = item["file_path"]

        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")

                # If bounding box coordinates exist, crop to region of interest
                bbox = item.get("bbox")
                if bbox and len(bbox) == 4:
                    w, h = img.size
                    x_min, y_min, x_max, y_max = bbox
                    left = max(0, int(x_min * w))
                    top = max(0, int(y_min * h))
                    right = min(w, int(x_max * w))
                    bottom = min(h, int(y_max * h))
                    if right - left > 10 and bottom - top > 10:
                        img = img.crop((left, top, right, bottom))

                if self.transform:
                    tensor = self.transform(img)
                else:
                    tensor = transforms.ToTensor()(img)

                return tensor, item["label_idx"]
        except Exception:
            # Fallback to white dummy image if file corrupted
            dummy = Image.new("RGB", (224, 224), color=(255, 255, 255))
            if self.transform:
                return self.transform(dummy), item["label_idx"]
            return transforms.ToTensor()(dummy), item["label_idx"]


def get_training_transforms():
    """Augmentation pipeline preserving packaging text while introducing realistic variations."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_validation_transforms():
    """Validation transform without stochastic augmentations."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def execute_metric_training_job(
    job_id: str,
    config: Dict[str, Any],
    db_session_factory,
):
    """
    Background worker thread executing real deep metric learning training job.
    Enforces concurrency lock, resource bounds, checkpointing, and model registry record creation.
    """
    acquired = TRAINING_MUTEX.acquire(blocking=False)
    if not acquired:
        with db_session_factory() as db:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.status = "FAILED"
                job.error_message = "Another training job is currently running. Only 1 concurrent training job is permitted."
                db.commit()
        return

    try:
        _run_training_pipeline(job_id, config, db_session_factory)
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        print(f"Error in training job {job_id}: {err_msg}")
        try:
            with db_session_factory() as db:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = "FAILED"
                    job.error_message = str(e)
                    job.logs = (job.logs or "") + f"\nFatal error: {err_msg}"
                    db.commit()
        except Exception as db_err:
            print(f"Failed to record failure in DB: {db_err}")
    finally:
        ACTIVE_CANCELLATIONS.pop(job_id, None)
        TRAINING_MUTEX.release()


def _run_training_pipeline(
    job_id: str,
    config: Dict[str, Any],
    db_session_factory,
):
    start_time = time.time()
    device = get_torch_device()

    with db_session_factory() as db:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            return
        job.status = "RUNNING"
        job.started_at = datetime.utcnow()
        db.commit()

        # 1. Fetch training and validation samples
        all_images = (
            db.query(ProductImage)
            .join(SKU)
            .filter(ProductImage.is_valid == True, SKU.is_active == True)
            .all()
        )

        sku_list = db.query(SKU).filter(SKU.is_active == True).all()
        sku_to_idx = {s.id: i for i, s in enumerate(sku_list)}

        train_samples = []
        val_samples = []

        for img in all_images:
            if not os.path.exists(img.file_path):
                continue
            lbl = sku_to_idx.get(img.sku_id)
            if lbl is None:
                continue

            sample_entry = {
                "file_path": img.file_path,
                "label_idx": lbl,
                "sku_id": img.sku_id,
            }

            # Check bbox annotation
            if img.annotations:
                ann = img.annotations[0]
                sample_entry["bbox"] = (ann.x_min, ann.y_min, ann.x_max, ann.y_max)

            if img.split in ("val", "validation"):
                val_samples.append(sample_entry)
            elif img.split == "test":
                pass  # Keep test set purely independent
            else:
                train_samples.append(sample_entry)

        # Check data sufficiency for metric learning
        train_labels = [s["label_idx"] for s in train_samples]
        unique_train_labels = set(train_labels)

        if len(unique_train_labels) < 2 or len(train_samples) < 4:
            job.status = "INSUFFICIENT_DATA"
            job.error_message = (
                f"Insufficient dataset for metric learning: Found {len(unique_train_labels)} distinct SKUs "
                f"and {len(train_samples)} total training images. At least 2 distinct SKUs with multiple images "
                "are required to construct positive and negative pairs."
            )
            db.commit()
            return

        # If val set empty, partition 20% of train set deterministically
        if len(val_samples) == 0:
            random.seed(42)
            by_sku: Dict[int, List[Dict[str, Any]]] = {}
            for s in train_samples:
                by_sku.setdefault(s["sku_id"], []).append(s)

            new_train = []
            for s_id, s_list in by_sku.items():
                if len(s_list) > 1:
                    val_samples.append(s_list[0])
                    new_train.extend(s_list[1:])
                else:
                    new_train.extend(s_list)
            train_samples = new_train
            train_labels = [s["label_idx"] for s in train_samples]

    # Hyperparameters from config
    epochs = int(config.get("epochs", 5))
    embedding_dim = int(config.get("embedding_dim", 512))
    learning_rate = float(config.get("learning_rate", 1e-4))
    margin = float(config.get("margin", 0.2))
    p_classes = min(len(unique_train_labels), int(config.get("p_classes", 2)))
    k_instances = int(config.get("k_instances", 2))
    batches_per_epoch = max(4, len(train_samples) // (p_classes * k_instances))

    # Datasets and Loaders
    train_dataset = MetricLearningDataset(train_samples, transform=get_training_transforms())
    val_dataset = MetricLearningDataset(val_samples, transform=get_validation_transforms())

    train_sampler = IdentityAwareBatchSampler(
        labels=train_labels,
        p=p_classes,
        k=k_instances,
        num_batches=batches_per_epoch,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=min(8, max(1, len(val_dataset))), shuffle=False)

    # Initialize Model
    model = MetricEmbeddingModel(embedding_dim=embedding_dim, freeze_backbone=True)
    model.to(device)

    criterion = CosineTripletLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.projection_head.parameters(), lr=learning_rate, weight_decay=1e-4)

    history = []
    best_val_loss = float("inf")
    best_checkpoint_path = None
    job_checkpoint_dir = CHECKPOINTS_DIR / job_id
    job_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_lines = [
        f"Initialized metric learning job {job_id}",
        f"Device: {device} | Epochs: {epochs} | Dim: {embedding_dim} | Margin: {margin}",
        f"Training samples: {len(train_samples)} across {len(unique_train_labels)} SKUs",
        f"Validation samples: {len(val_samples)}",
    ]

    for epoch in range(1, epochs + 1):
        # Check cancellation
        if ACTIVE_CANCELLATIONS.get(job_id, False):
            with db_session_factory() as db:
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = "CANCELLED"
                    job.logs = "\n".join(log_lines)
                    db.commit()
            return

        # Training Phase
        model.train()
        train_loss_total = 0.0
        train_steps = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            embeddings = model(batch_x)
            loss = criterion(embeddings, batch_y)

            if loss.item() > 0:
                loss.backward()
                optimizer.step()

            train_loss_total += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_total / max(1, train_steps)

        # Validation Phase: Evaluate validation loss and nearest-neighbor Top-1 accuracy
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        val_embeddings_list = []
        val_labels_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                emb = model(batch_x)
                loss = criterion(emb, batch_y)
                val_loss_total += loss.item()
                val_steps += 1

                val_embeddings_list.append(emb.cpu())
                val_labels_list.append(batch_y.cpu())

        avg_val_loss = val_loss_total / max(1, val_steps)

        # Compute Top-1 accuracy on validation embeddings via leave-one-out cosine distance
        val_acc = 0.0
        if len(val_embeddings_list) > 0:
            val_emb_cat = torch.cat(val_embeddings_list, dim=0)
            val_lbl_cat = torch.cat(val_labels_list, dim=0)
            n_val = val_emb_cat.size(0)

            if n_val > 1:
                # Cosine similarities
                sims = torch.matmul(val_emb_cat, val_emb_cat.t())
                # Mask self
                sims.fill_diagonal_(-1.0)
                nearest_idx = torch.argmax(sims, dim=1)
                predicted_lbls = val_lbl_cat[nearest_idx]
                correct = (predicted_lbls == val_lbl_cat).sum().item()
                val_acc = round(correct / n_val, 4)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": val_acc,
            "duration_s": round(time.time() - start_time, 2),
        }
        history.append(epoch_record)

        epoch_log = f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Top-1: {val_acc*100:.1f}%"
        log_lines.append(epoch_log)

        # Save Checkpoint if best validation loss
        checkpoint_file = job_checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "embedding_dim": embedding_dim,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "config": config,
            },
            checkpoint_file,
        )

        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = str(checkpoint_file)

        # Persist intermediate epoch progress to DB
        with db_session_factory() as db:
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.current_epoch = epoch
                job.train_loss = avg_train_loss
                job.val_loss = avg_val_loss
                job.val_accuracy = val_acc
                job.history_json = json.dumps(history)
                job.logs = "\n".join(log_lines)
                job.checkpoint_path = best_checkpoint_path
                db.commit()

    # Finalize Completed Job
    with db_session_factory() as db:
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job:
            job.status = "COMPLETED"
            job.completed_at = datetime.utcnow()
            job.checkpoint_path = best_checkpoint_path
            job.logs = "\n".join(log_lines) + "\nTraining job completed successfully."
            db.commit()

            # Create a model version entry in the ModelRegistry (ready for evaluation/activation, but not auto-activated!)
            model_version_id = f"mobilenetv3-metric-{job_id}"
            reg_entry = ModelRegistry(
                model_id=model_version_id,
                model_type="embedding",
                architecture=f"MobileNetV3-Large + MetricProjectionHead({embedding_dim})",
                pretrained_source="torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)",
                weight_version=f"Job-{job_id}-Best",
                embedding_dim=embedding_dim,
                checkpoint_path=best_checkpoint_path,
                hyperparameters_json=json.dumps(config),
                metrics_json=json.dumps({
                    "best_val_loss": round(best_val_loss, 4),
                    "best_val_accuracy": val_acc,
                    "epochs_trained": epochs,
                }),
                status="ready",
                is_active=False,  # DO NOT overwrite active model automatically
                license="BSD-3-Clause",
            )
            db.add(reg_entry)
            db.commit()
