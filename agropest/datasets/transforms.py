"""Detection-specific data augmentation utilities."""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from ..utils.box_ops import clip_boxes_to_image


class Compose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, torch.Tensor]):
        image = TF.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: Image.Image | torch.Tensor, target: Dict[str, torch.Tensor]):
        if random.random() < self.prob:
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                width, _ = image.size
            else:
                image = torch.flip(image, dims=[2])
                width = image.shape[2]
            boxes = target["boxes"].clone()
            if boxes.numel() > 0:
                xmin = boxes[:, 0].clone()
                xmax = boxes[:, 2].clone()
                boxes[:, 0] = width - xmax
                boxes[:, 2] = width - xmin
                target["boxes"] = boxes
        return image, target


class Resize:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image: Image.Image | torch.Tensor, target: Dict[str, torch.Tensor]):
        width, height = self.size
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size
            image = image.resize((width, height), resample=Image.BILINEAR)
        else:
            orig_height, orig_width = image.shape[1:]
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
            ).squeeze(0)
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            scale_x = width / orig_width
            scale_y = height / orig_height
            boxes[:, 0::2] *= scale_x
            boxes[:, 1::2] *= scale_y
            boxes = clip_boxes_to_image(boxes, (height, width))
            target["boxes"] = boxes
        target["size"] = torch.tensor([height, width], dtype=torch.int64)
        return image, target


class Normalize:
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target: Dict[str, torch.Tensor]):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target


def build_transforms(train: bool = True, image_size: Tuple[int, int] | None = None):
    transforms: List[Callable] = []
    if image_size is not None:
        transforms.append(Resize(image_size))
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip())
    transforms.append(Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return Compose(transforms)
