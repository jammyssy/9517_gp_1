"""Model factory for AgroPest detection methods."""
from __future__ import annotations

from typing import Literal

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


MethodName = Literal["faster_rcnn", "retinanet"]


def build_model(method: MethodName, num_classes: int):
    """Constructs a detection model with the requested head."""
    if method == "faster_rcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model
    if method == "retinanet":
        model = torchvision.models.detection.retinanet_resnet50_fpn(weights="DEFAULT")
        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(in_channels, num_anchors, num_classes)
        return model
    raise ValueError(f"Unknown method: {method}")
