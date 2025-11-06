# COMP9517 Group Project

## Introduction

The goal of this group project is to collaborate in a team of five students to
develop computer vision solutions for detecting and classifying agricultural
insect pests. Teams are expected to present their findings in both oral and
written formats and can meet with tutors once per week during Weeks 6–10 for
feedback on progress.

## Project Description

Healthy and productive agriculture underpins human life and national economic
development. Insect pests can severely damage crops, reduce yields, and
threaten food availability. Monitoring pest activity throughout the cropping
season and deploying effective management strategies therefore requires
advanced tools. Computer vision techniques offer a way to rapidly and reliably
detect and identify insects from imagery captured in natural agricultural
environments. This project focuses on creating such tools.

## Task Statement

The overarching task is to develop and compare different computer vision
methods capable of detecting and classifying insects in natural-scene images.
Each complete method combines an object detector with a classifier, and the
group must implement and evaluate at least two distinct end-to-end methods.

## Dataset

The project uses the public AgroPest-12 dataset from Kaggle, which comprises
11,502 training images, 1,095 validation images, and 546 test images covering
12 insect classes. The dataset includes ground-truth class labels and bounding
boxes for every sample.

Dataset reference:

> Rupankar Majumdar. *AgroPest-12: Image Dataset for Crop Pest Detection*.
> Kaggle, September 2025.
> <https://www.kaggle.com/datasets/rupankarmajumdar/crop-pests-dataset>

## Method Development Guidelines

- Explore a variety of traditional, machine learning, and deep learning-based
  computer vision approaches. Concepts taught in the course can be combined
  with ideas from the literature.
- At least two complete detection-and-classification pipelines must be
  implemented and assessed.
- Coding custom solutions or modifying existing approaches is encouraged; cite
  all external sources, papers, tools, and repositories used.

### Comprehensive Method Development (Targeting 33–36 Marks)

To aim for higher marks, ensure methodological diversity, for example by:

- Implementing machine learning pipelines that compare handcrafted feature
  descriptors and classifiers.
- Developing deep learning pipelines that explore different neural network
  architectures and data augmentation strategies.

### Advanced Method Development (Targeting 37+ Marks)

To pursue top marks, incorporate original ideas and research-oriented
investigations, such as:

- Resampling the dataset to create imbalanced classification scenarios and
  evaluating mitigation strategies (reweighting, resampling, augmentation).
- Studying robustness to practical distortions like noise, blur, low
  brightness/contrast, or occlusion, akin to adversarial robustness analyses.
- Extending methods with explainable AI techniques (e.g., attention maps) and
  analyzing the resulting explanations.

## Training and Evaluation Protocol

- Respect the provided train/validation/test split to avoid data leakage.
- Use the training and validation sets for learning and hyperparameter tuning;
  reserve the test set strictly for final evaluation.
- Report detection performance using mean Average Precision (mAP).
- Report classification performance using precision, recall, F1 score,
  accuracy, and area under the curve (AUC).
- Compare training and inference times across methods.

## Results and Discussion Expectations

- Present quantitative metrics, representative successes, and failure cases in
  both the video presentation and written report.
- Analyze failure cases, discuss why certain methods outperform others, and
  propose future research directions to enhance performance.

## Repository Layout

This repository now contains executable code for preparing the AgroPest-12
dataset, training two baseline detection-and-classification pipelines, and
evaluating trained checkpoints.

```
├── agropest/                 # Python package with datasets, models, utilities
│   ├── datasets/             # Dataset + augmentations
│   ├── engine/               # Training and evaluation loops
│   ├── models/               # Model factory (Faster R-CNN, RetinaNet)
│   └── utils/                # Metric computation helpers
├── scripts/
│   └── convert_to_coco.py    # CSV → COCO annotation conversion helper
├── train.py                  # CLI for training a method end-to-end
├── evaluate.py               # CLI for offline evaluation of checkpoints
├── requirements.txt          # Python dependencies
└── README.md
```

Install the dependencies with

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparing the Dataset

The code expects COCO-style JSON annotations. If you download the Kaggle CSV
annotations, convert them using `scripts/convert_to_coco.py`:

```bash
python scripts/convert_to_coco.py \
  --images-dir /path/to/AgroPest-12/train/images \
  --annotations /path/to/AgroPest-12/train_annotations.csv \
  --output data/train.json
```

Run the script once per split (`train`, `val`, `test`) to create three JSON
files. The converter accepts custom column names through optional flags to
match the Kaggle CSV schema.

## Training Pipelines

The repository implements two required detector-classifier pipelines:

1. **Faster R-CNN + ResNet-50 FPN** (`--method faster_rcnn`) fine-tunes the
   Region Proposal Network and ROI heads on AgroPest-12.
2. **RetinaNet + ResNet-50 FPN** (`--method retinanet`) trains a one-stage
   detector with focal loss.

Launch training with, for example,

```bash
python train.py \
  --data-root /path/to/AgroPest-12/images \
  --train-annotations data/train.json \
  --val-annotations data/val.json \
  --method faster_rcnn \
  --output runs/faster_rcnn \
  --epochs 30 \
  --batch-size 4 \
  --learning-rate 0.0005
```

Key flags:

- `--adamw` switches to the AdamW optimizer (default is momentum SGD).
- `--image-size WIDTH HEIGHT` resizes images before training/evaluation.
- `--resume` resumes training from an existing checkpoint.

During training the script stores `last_model.pt`, `best_model.pt`, and a JSON
history with validation metrics in the chosen `--output` directory.

## Evaluating Checkpoints

Evaluate a trained model on the validation or test split with

```bash
python evaluate.py \
  --data-root /path/to/AgroPest-12/images \
  --annotations data/test.json \
  --method retinanet \
  --checkpoint runs/retinanet/best_model.pt
```

The command prints mAP@[0.5:0.95], mAP@0.5, macro precision/recall/F1,
accuracy, and (when computable) macro-averaged ROC AUC.

## Practical Considerations

- The dataset is under 600 MB, making training feasible on standard desktop or
  laptop hardware. If computational resources are limited, document any subset
  usage (e.g., 50–75%) and acknowledge the impact on performance.
- Consider leveraging platforms like Google Colab if additional compute is
  required.

## Additional References

- Sourav Chakrabarty et al. *Application of artificial intelligence in insect pest
  identification – a review*. Artificial Intelligence in Agriculture 16(1):44–61,
  March 2026. <https://doi.org/10.1016/j.aiia.2025.06.005>
- Kaili Wang et al. *AP162: A large-scale dataset for agricultural pest
  recognition*. Computers and Electronics in Agriculture 237B:110520, October
   2025. <https://doi.org/10.1016/j.compag.2025.110520>
- Xiaoping Wu et al. *IP102: A large-scale benchmark dataset for insect pest
  recognition*. CVPR, June 2019. <https://doi.org/10.1109/CVPR.2019.00899>
