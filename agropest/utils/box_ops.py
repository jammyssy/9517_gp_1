"""Utility functions for bounding box operations."""
from __future__ import annotations

from typing import Tuple

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Computes area of set of boxes."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0,))
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return widths * heights


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two sets of boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top corners
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom corners

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def clip_boxes_to_image(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Clip boxes to image boundaries."""
    height, width = size
    boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=height)
    return boxes
