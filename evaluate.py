"""CLI entrypoint for evaluating trained models."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agropest.datasets.agropest import AgroPestDetection, collate_fn
from agropest.datasets.transforms import build_transforms
from agropest.engine.evaluate import evaluate_model
from agropest.models.factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AgroPest detection models")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--method", type=str, default="faster_rcnn", choices=["faster_rcnn", "retinanet"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    transforms = build_transforms(train=False, image_size=tuple(args.image_size) if args.image_size else None)
    dataset = AgroPestDetection(args.data_root, args.annotations, transforms=transforms)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(args.method, dataset.num_classes)
    device = torch.device(args.device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = evaluate_model(model, data_loader, device, dataset.num_classes)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
