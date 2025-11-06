"""CLI entrypoint for training AgroPest detection models."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from agropest.datasets.agropest import AgroPestDetection, collate_fn
from agropest.datasets.transforms import build_transforms
from agropest.engine.train import train
from agropest.models.factory import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AgroPest detection models")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory with dataset images")
    parser.add_argument("--train-annotations", type=Path, required=True, help="COCO-style annotations for training")
    parser.add_argument("--val-annotations", type=Path, help="COCO-style annotations for validation")
    parser.add_argument("--method", type=str, default="faster_rcnn", choices=["faster_rcnn", "retinanet"], help="Detection method")
    parser.add_argument("--output", type=Path, default=Path("runs/faster_rcnn"), help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "multistep", "none"], help="LR scheduler type")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), help="Optional resize (width height)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path, help="Path to checkpoint to resume from")
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--adamw", action="store_true", help="Use AdamW optimizer instead of SGD")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_transforms = build_transforms(train=True, image_size=tuple(args.image_size) if args.image_size else None)
    val_transforms = build_transforms(train=False, image_size=tuple(args.image_size) if args.image_size else None)

    train_dataset = AgroPestDetection(args.data_root, args.train_annotations, transforms=train_transforms)
    val_dataset = None
    if args.val_annotations:
        val_dataset = AgroPestDetection(args.data_root, args.val_annotations, transforms=val_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    num_classes = train_dataset.num_classes
    model = build_model(args.method, num_classes)
    device = torch.device(args.device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.adamw:
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint)

    train(
        model,
        optimizer,
        train_loader,
        val_loader,
        device=device,
        num_epochs=args.epochs,
        num_classes=num_classes,
        scheduler_type=args.scheduler,
        output_dir=args.output,
        gradient_clip=args.gradient_clip,
        print_freq=args.print_freq,
    )


if __name__ == "__main__":
    main()
