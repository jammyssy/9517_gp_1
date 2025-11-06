"""Training utilities for AgroPest detection models."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from ..utils.metrics import ClassificationMetrics, DetectionMetrics, evaluate_classification, evaluate_detections


@dataclass
class TrainState:
    epoch: int
    best_map: float


def _forward_pass(model, images, targets, device: torch.device):
    model.train()
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    return losses, loss_dict


def _evaluate(model, data_loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in output.items()} for output in outputs]
            predictions.extend(outputs)
            targets_list.extend([{k: v.cpu() if torch.is_tensor(v) else v for k, v in tgt.items()} for tgt in targets])
    detection_metrics = evaluate_detections(predictions, targets_list, num_classes)
    classification_metrics = evaluate_classification(predictions, targets_list, num_classes)
    return detection_metrics, classification_metrics


def create_scheduler(optimizer: Optimizer, scheduler_type: str, num_epochs: int):
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "multistep":
        milestones = [int(num_epochs * 0.6), int(num_epochs * 0.8)]
        return MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    if scheduler_type == "none":
        return None
    raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def train(
    model,
    optimizer: Optimizer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    num_epochs: int,
    num_classes: int,
    scheduler_type: str = "cosine",
    output_dir: str | Path = "runs",
    gradient_clip: Optional[float] = 5.0,
    print_freq: int = 20,
):
    scheduler = create_scheduler(optimizer, scheduler_type, num_epochs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = TrainState(epoch=0, best_map=0.0)
    history = []

    for epoch in range(num_epochs):
        state.epoch = epoch
        model.train()
        for step, (images, targets) in enumerate(train_loader):
            losses, loss_dict = _forward_pass(model, images, targets, device)
            optimizer.zero_grad()
            losses.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            if step % print_freq == 0:
                losses_reduced = {k: float(v.item()) for k, v in loss_dict.items()}
                print(f"Epoch {epoch+1}/{num_epochs} Step {step}: loss={float(losses.item()):.4f} {losses_reduced}")
        if scheduler:
            scheduler.step()

        val_det_metrics: Optional[DetectionMetrics] = None
        val_cls_metrics: Optional[ClassificationMetrics] = None
        if val_loader is not None:
            val_det_metrics, val_cls_metrics = _evaluate(model, val_loader, device, num_classes)
            history.append({
                "epoch": epoch + 1,
                "mAP50-95": val_det_metrics.map_50_95,
                "mAP50": val_det_metrics.map_50,
                "precision": val_cls_metrics.precision if val_cls_metrics else None,
                "recall": val_cls_metrics.recall if val_cls_metrics else None,
                "f1": val_cls_metrics.f1 if val_cls_metrics else None,
                "accuracy": val_cls_metrics.accuracy if val_cls_metrics else None,
                "auc": val_cls_metrics.auc if val_cls_metrics else None,
            })
            if val_det_metrics.map_50_95 > state.best_map:
                state.best_map = val_det_metrics.map_50_95
                torch.save(model.state_dict(), output_dir / "best_model.pt")
        torch.save(model.state_dict(), output_dir / "last_model.pt")

    if history:
        with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return state
