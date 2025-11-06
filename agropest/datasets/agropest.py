"""PyTorch dataset for the AgroPest-12 pest detection task."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class AgroPestDetection(Dataset):
    """Detection dataset backed by COCO-style annotations."""

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        transforms: Optional[Callable[[Image.Image, Dict[str, torch.Tensor]], Tuple[Image.Image, Dict[str, torch.Tensor]]]] = None,
    ) -> None:
        self.root = Path(root)
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root {self.root} does not exist")
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file {self.annotation_file} does not exist")

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        self.categories = annotations.get("categories", [])
        self.category_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(self.categories, start=1)}
        self.idx_to_category_id = {idx: cat["id"] for idx, cat in enumerate(self.categories, start=1)}

        self.images = annotations.get("images", [])
        annotations_list = annotations.get("annotations", [])

        self.image_id_to_annotations: Dict[int, List[Dict[str, float]]] = {}
        for ann in annotations_list:
            image_id = ann["image_id"]
            category_id = ann["category_id"]
            bbox = ann["bbox"]  # [x, y, width, height]
            self.image_id_to_annotations.setdefault(image_id, []).append({
                "bbox": bbox,
                "category_id": category_id,
                "area": ann.get("area", bbox[2] * bbox[3]),
                "iscrowd": ann.get("iscrowd", 0),
                "id": ann.get("id"),
            })

        self.ids = [img["id"] for img in self.images]
        self.id_to_record = {img["id"]: img for img in self.images}

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def num_classes(self) -> int:
        """Number of classes including background."""
        return len(self.categories) + 1

    def _load_image(self, record: Dict[str, object]) -> Image.Image:
        path = self.root / record["file_name"]
        if not path.exists():
            raise FileNotFoundError(f"Image file {path} referenced in annotations does not exist")
        return Image.open(path).convert("RGB")

    def _format_annotations(self, image_id: int, height: int, width: int) -> Dict[str, torch.Tensor]:
        anns = self.image_id_to_annotations.get(image_id, [])
        boxes: List[List[float]] = []
        labels: List[int] = []
        areas: List[float] = []
        iscrowd: List[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_id_to_idx[ann["category_id"]])
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas_tensor = torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)

        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.ids[idx]
        record = self.id_to_record[image_id]
        image = self._load_image(record)
        width, height = image.size

        target = self._format_annotations(image_id, height, width)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))
