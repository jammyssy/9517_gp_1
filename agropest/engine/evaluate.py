"""Evaluation utilities."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from ..utils.metrics import evaluate_classification, evaluate_detections


def evaluate_model(model, data_loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    predictions: List[Dict[str, torch.Tensor]] = []
    targets_list: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in output.items()} for output in outputs]
            predictions.extend(outputs)
            targets_list.extend([{k: v.cpu() if torch.is_tensor(v) else v for k, v in tgt.items()} for tgt in targets])

    detection_metrics = evaluate_detections(predictions, targets_list, num_classes)
    classification_metrics = evaluate_classification(predictions, targets_list, num_classes)

    metrics = {
        "map_50_95": detection_metrics.map_50_95,
        "map_50": detection_metrics.map_50,
        "precision": classification_metrics.precision,
        "recall": classification_metrics.recall,
        "f1": classification_metrics.f1,
        "accuracy": classification_metrics.accuracy,
    }
    if classification_metrics.auc is not None:
        metrics["auc"] = classification_metrics.auc
    return metrics
