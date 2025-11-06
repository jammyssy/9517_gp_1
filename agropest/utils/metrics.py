"""Metrics for evaluating detection and classification performance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .box_ops import box_iou


@dataclass
class DetectionMetrics:
    map_50_95: float
    map_50: float
    per_class_ap: Dict[int, float]


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    auc: float | None


def _accumulate_detections(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    iou_thresholds: Iterable[float],
    num_classes: int,
) -> Tuple[Dict[float, Dict[int, List[float]]], Dict[float, Dict[int, List[float]]]]:
    """Accumulate true positive flags and scores per class for AP computation."""

    scores: Dict[float, Dict[int, List[float]]] = {thr: {c: [] for c in range(1, num_classes)} for thr in iou_thresholds}
    matches: Dict[float, Dict[int, List[float]]] = {thr: {c: [] for c in range(1, num_classes)} for thr in iou_thresholds}

    for preds_img, target_img in zip(predictions, targets):
        gt_boxes = target_img["boxes"].cpu()
        gt_labels = target_img["labels"].cpu()
        gt_used = {thr: torch.zeros(len(gt_boxes), dtype=torch.bool) for thr in iou_thresholds}

        pred_boxes = preds_img["boxes"].cpu()
        pred_scores = preds_img["scores"].cpu()
        pred_labels = preds_img["labels"].cpu()

        if pred_boxes.numel() == 0:
            continue

        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_labels = pred_labels[order]

        ious = box_iou(pred_boxes, gt_boxes)

        for idx, (box_label, score) in enumerate(zip(pred_labels, pred_scores)):
            c = int(box_label.item())
            if c == 0:
                # Background class is ignored in AP computations
                continue

            for thr in iou_thresholds:
                scores[thr][c].append(float(score.item()))
                assigned = False
                if gt_boxes.numel() > 0:
                    iou_vals = ious[idx]
                    max_iou, max_idx = torch.max(iou_vals, dim=0)
                    if max_iou >= thr and gt_labels[max_idx] == c and not gt_used[thr][max_idx]:
                        matches[thr][c].append(1.0)
                        gt_used[thr][max_idx] = True
                        assigned = True
                if not assigned:
                    matches[thr][c].append(0.0)

    return scores, matches


def _compute_average_precision(scores: List[float], matches: List[float], num_gt: int) -> float:
    if len(scores) == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    matches = np.array(matches)[order]
    scores = np.array(scores)[order]

    tps = matches
    fps = 1 - matches
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)

    recalls = tp_cum / max(num_gt, 1)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # Ensure the precision-recall curve is monotonically decreasing
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    # Numerical integration
    recall_levels = np.concatenate(([0.0], recalls, [1.0]))
    precision_levels = np.concatenate(([precisions[0]], precisions, [0.0]))
    ap = np.trapz(precision_levels, recall_levels)
    return float(ap)


def evaluate_detections(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresholds: Iterable[float] | None = None,
) -> DetectionMetrics:
    """Compute mean average precision across IoU thresholds."""
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]

    iou_thresholds = list(iou_thresholds)

    scores, matches = _accumulate_detections(predictions, targets, iou_thresholds, num_classes)

    per_class_ap: Dict[int, float] = {}
    ap_per_thr: Dict[float, List[float]] = {thr: [] for thr in iou_thresholds}

    for c in range(1, num_classes):
        num_gt = sum(int((t["labels"] == c).sum().item()) for t in targets)
        ap_vals = []
        for thr in iou_thresholds:
            ap = _compute_average_precision(scores[thr][c], matches[thr][c], num_gt)
            ap_vals.append(ap)
            ap_per_thr[thr].append(ap)
        per_class_ap[c] = float(np.mean(ap_vals) if ap_vals else 0.0)

    map_50_95 = float(np.mean([np.mean(ap_per_thr[thr]) if ap_per_thr[thr] else 0.0 for thr in iou_thresholds]))
    map_50 = float(np.mean(ap_per_thr[0.5]) if ap_per_thr[0.5] else 0.0)

    return DetectionMetrics(map_50_95=map_50_95, map_50=map_50, per_class_ap=per_class_ap)


def evaluate_classification(
    predictions: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute classification metrics using IoU-matched predictions."""
    y_true: List[int] = []
    y_pred: List[int] = []
    y_scores: List[List[float]] = []

    for preds_img, target_img in zip(predictions, targets):
        gt_boxes = target_img["boxes"].cpu()
        gt_labels = target_img["labels"].cpu()
        pred_boxes = preds_img["boxes"].cpu()
        pred_scores = preds_img["scores"].cpu()
        pred_labels = preds_img["labels"].cpu()

        if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        for idx in range(len(pred_boxes)):
            iou_vals = ious[idx]
            max_iou, max_idx = torch.max(iou_vals, dim=0)
            if max_iou >= iou_threshold:
                gt_label = int(gt_labels[max_idx].item())
                pred_label = int(pred_labels[idx].item())
                if pred_label == 0:
                    continue
                y_true.append(gt_label)
                y_pred.append(pred_label)

                scores_vec = [0.0] * (num_classes - 1)
                cls_index = pred_label - 1
                scores_vec[cls_index] = float(pred_scores[idx].item())
                y_scores.append(scores_vec)

    if not y_true:
        return ClassificationMetrics(precision=0.0, recall=0.0, f1=0.0, accuracy=0.0, auc=None)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    auc = None
    try:
        y_true_binarized = np.eye(num_classes - 1)[np.array(y_true) - 1]
        auc = roc_auc_score(y_true_binarized, np.array(y_scores), average="macro", multi_class="ovr")
    except Exception:
        auc = None

    return ClassificationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        auc=None if auc is None else float(auc),
    )
