from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
# Use candidate weight paths (first-existing will be selected at runtime)
DEFAULT_WEIGHTS = ROOT_DIR / "Training" / "runs" / "detect" / "train_augmented2" / "weights" / "best.pt"

# Ordered candidates to try when locating model weights
WEIGHT_CANDIDATES = [
    ROOT_DIR / "Training" / "runs" / "detect" / "train_augmented2" / "weights" / "best.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train_augmented2" / "weights" / "last.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train_augmented" / "weights" / "best.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train2" / "weights" / "best.pt",
    ROOT_DIR / "Training" / "runs" / "detect" / "train" / "weights" / "best.pt",
    ROOT_DIR / "yolov8n.pt",
]

app = FastAPI(title="PharmaVision Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model, DEFAULT_WEIGHTS
    if _model is not None:
        return _model

    # Select the first existing candidate weight file
    chosen: Path | None = None
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            chosen = p
            break

    if chosen is None:
        tried = ", ".join(str(p) for p in WEIGHT_CANDIDATES)
        raise FileNotFoundError(f"Model weights not found. Tried: {tried}")

    # Persist chosen path for diagnostics and responses
    DEFAULT_WEIGHTS = chosen
    _model = YOLO(str(DEFAULT_WEIGHTS))
    return _model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _point_inside_box(cx: float, cy: float, box: dict[str, Any]) -> bool:
    return box["x1"] <= cx <= box["x2"] and box["y1"] <= cy <= box["y2"]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.75),
    strict_mode: bool = Form(True),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    image_np = np.array(image)

    user_conf = max(0.10, min(0.99, conf))
    model_conf = max(0.20, min(user_conf, 0.95))

    try:
        results = get_model().predict(source=image_np, conf=model_conf, verbose=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    result = results[0]
    boxes = result.boxes
    names = result.names

    raw_predictions: list[dict[str, Any]] = []
    image_area = float(image_np.shape[0] * image_np.shape[1])

    if boxes is not None and boxes.xyxy is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        box_conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            raw_predictions.append(
                {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(width),
                    "height": float(height),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(box_conf[i]),
                    "class_id": int(cls[i]),
                    "class": str(names.get(int(cls[i]), int(cls[i]))),
                    "area_ratio": float((width * height) / image_area) if image_area > 0 else 0.0,
                }
            )

    if strict_mode:
        strip_conf = max(user_conf, 0.70)
        strip_candidates = [
            p
            for p in raw_predictions
            if p["class"].lower() == "strip" and p["confidence"] >= strip_conf and 0.08 <= p["area_ratio"] <= 0.95
        ]

        filtered: list[dict[str, Any]] = []
        if strip_candidates:
            filtered.extend(strip_candidates)
            for p in raw_predictions:
                cls_name = p["class"].lower()
                if cls_name == "strip":
                    continue
                if p["confidence"] < user_conf:
                    continue
                if p["area_ratio"] > 0.20:
                    continue

                cx = p["x1"] + (p["width"] / 2.0)
                cy = p["y1"] + (p["height"] / 2.0)
                if any(_point_inside_box(cx, cy, s) for s in strip_candidates):
                    filtered.append(p)

        predictions = filtered
    else:
        predictions = [p for p in raw_predictions if p["confidence"] >= user_conf]

    return {
        "model": str(DEFAULT_WEIGHTS),
        "image_width": int(image_np.shape[1]),
        "image_height": int(image_np.shape[0]),
        "confidence_threshold": user_conf,
        "strict_mode": strict_mode,
        "num_raw_detections": len(raw_predictions),
        "num_detections": len(predictions),
        "predictions": predictions,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
