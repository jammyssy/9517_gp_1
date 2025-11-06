"""Utility script to convert bounding-box annotations stored in CSV to COCO JSON."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AgroPest CSV annotations to COCO format")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing images referenced in the CSV")
    parser.add_argument("--annotations", type=Path, required=True, help="CSV file with bounding box annotations")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the generated COCO JSON")
    parser.add_argument("--image-column", type=str, default="image_id", help="Column containing image file names")
    parser.add_argument("--label-column", type=str, default="class_name", help="Column containing class names")
    parser.add_argument("--xmin-column", type=str, default="xmin")
    parser.add_argument("--ymin-column", type=str, default="ymin")
    parser.add_argument("--xmax-column", type=str, default="xmax")
    parser.add_argument("--ymax-column", type=str, default="ymax")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory {images_dir} does not exist")

    with open(args.annotations, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("Annotation CSV is empty")

    class_to_id: Dict[str, int] = {}
    categories: List[Dict[str, object]] = []

    for row in rows:
        label = row[args.label_column]
        if label not in class_to_id:
            class_id = len(class_to_id) + 1
            class_to_id[label] = class_id
            categories.append({"id": class_id, "name": label})

    images: Dict[str, Dict[str, object]] = {}
    annotations: List[Dict[str, object]] = []
    annotation_id = 1

    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[row[args.image_column]].append(row)

    for image_name, group in grouped_rows.items():
        image_path = images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_path} referenced in CSV not found")
        with Image.open(image_path) as img:
            width, height = img.size
        image_id = len(images) + 1
        images[image_name] = {
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height,
        }
        for row in group:
            xmin = float(row[args.xmin_column])
            ymin = float(row[args.ymin_column])
            xmax = float(row[args.xmax_column])
            ymax = float(row[args.ymax_column])
            width_box = max(0.0, xmax - xmin)
            height_box = max(0.0, ymax - ymin)
            area = width_box * height_box
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_to_id[row[args.label_column]],
                    "bbox": [xmin, ymin, width_box, height_box],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    coco_dict = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": categories,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)
    print(f"Saved COCO annotations to {args.output}")


if __name__ == "__main__":
    main()
