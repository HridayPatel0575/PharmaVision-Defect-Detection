#!/usr/bin/env python
"""Quick diagnostic script to test backend startup."""

import sys
from pathlib import Path

print("=" * 60)
print("PharmaVision Backend Startup Diagnostics")
print("=" * 60)

# Test 1: Check model weight files
print("\n[1] Checking model weight candidates...")
ROOT_DIR = Path(__file__).resolve().parent.parent

WEIGHT_CANDIDATES = [
    ROOT_DIR / "Training" / "runs" / "detect" / "train_augmented2" / "weights" / "best.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train_augmented2" / "weights" / "last.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train_augmented" / "weights" / "best.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train2" / "weights" / "best.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train" / "weights" / "best.pt",
    ROOT_DIR / "yolov8n.pt",
]

chosen = None
for p in WEIGHT_CANDIDATES:
    exists = "✓" if p.exists() else "✗"
    print(f"  {exists} {p}")
    if p.exists() and chosen is None:
        chosen = p

if chosen:
    print(f"\n✓ Selected weights: {chosen}")
    print(f"  File size: {chosen.stat().st_size / 1024 / 1024:.1f} MB")
else:
    print("\n✗ ERROR: No model weights found!")
    sys.exit(1)

# Test 2: Try importing YOLO
print("\n[2] Testing YOLO import...")
try:
    from ultralytics import YOLO
    print("✓ YOLO imported successfully")
except ImportError as e:
    print(f"✗ ERROR: Failed to import YOLO: {e}")
    sys.exit(1)

# Test 3: Try loading the model
print(f"\n[3] Loading model from: {chosen}")
print("  (This may take 30-60 seconds...)")
try:
    model = YOLO(str(chosen))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ ERROR: Failed to load model: {e}")
    sys.exit(1)

# Test 4: Try a dummy prediction
print("\n[4] Testing inference with dummy image...")
try:
    import numpy as np
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(source=dummy_img, conf=0.5, verbose=False)
    print(f"✓ Inference successful ({len(results[0].boxes)} detections)")
except Exception as e:
    print(f"✗ ERROR: Inference failed: {e}")
    sys.exit(1)

# Test 5: Try starting FastAPI
print("\n[5] Testing FastAPI startup...")
try:
    from main import app
    print("✓ FastAPI app initialized successfully")
except Exception as e:
    print(f"✗ ERROR: Failed to initialize FastAPI: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All diagnostics passed!")
print("=" * 60)
print("\nTo start the backend, run:")
print("  python main.py")
print("\nOr with uvicorn directly:")
print("  uvicorn main:app --host 127.0.0.1 --port 8000 --reload")
