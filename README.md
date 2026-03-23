# Pharma Defect Detection (YOLO)

Computer vision project for defect detection in pharmaceutical products and packaging using YOLO-based object detection.

## Overview

This repository contains:
- YOLO-ready dataset (`final_dataset.v1i.yolov8/`)
- Training assets and experiments (`Training/`)
- Notebook-based workflow for model training and evaluation

Primary use case: automated visual quality inspection for pharmaceutical manufacturing.

## Repository Structure

- `final_dataset.v1i.yolov8/`
  - `data.yaml`
  - `train/images`, `train/labels`
  - `valid/images`, `valid/labels`
- `Training/`
  - `Detection.ipynb`
  - `dvc.yaml`
  - baseline weights (`yolo11n.pt`, `yolov8n.pt`)
  - experiment outputs (`runs/`, `dvclive/`)

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install ultralytics jupyter dvc dvclive
```

3. Open the notebook:

```bash
jupyter notebook Training/Detection.ipynb
```

## Training with YOLO

Use the dataset config:
- `final_dataset.v1i.yolov8/data.yaml`

Example command:

```bash
yolo task=detect mode=train model=Training/yolov8n.pt data=final_dataset.v1i.yolov8/data.yaml epochs=100 imgsz=640 project=Training/runs name=pharma_defect
```

## Validation / Inference Example

```bash
yolo task=detect mode=val model=Training/runs/pharma_defect/weights/best.pt data=final_dataset.v1i.yolov8/data.yaml
```

## DVC Tracking

`Training/dvc.yaml` and `Training/dvclive/` are used for experiment tracking. If you use remote storage, configure your DVC remote before pushing artifacts.

## Notes

- Keep only selected model artifacts in git; use DVC or external storage for large files.
- Confirm GMP-aligned QA requirements (traceability, reproducibility, and validation protocol) before production deployment.

## License

Add your project license information here.
