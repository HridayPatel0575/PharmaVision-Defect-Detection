from pathlib import Path
from ultralytics import YOLO, settings

settings.update({"dvc": False})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML    = PROJECT_ROOT / "Training" / "data.fixed.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"data.fixed.yaml not found at {DATA_YAML}.\n"
        "Run the path-fixing cell in Detection.ipynb first, or update DATA_YAML."
    )

model = YOLO("yolov8n.pt")

results = model.train(
    data    = str(DATA_YAML),
    epochs  = 100,
    imgsz   = 640,
    batch   = 8,
    workers = 0,

    project  = str(PROJECT_ROOT / "Training" / "runs" / "detect"),
    name     = "train_augmented",
    exist_ok = False,

    dropout         = 0.3,
    weight_decay    = 0.001,
    label_smoothing = 0.1,

    cos_lr        = True,
    lr0           = 0.005,
    lrf           = 0.005,
    warmup_epochs = 5,

    patience = 20,

    degrees     = 10.0,
    translate   = 0.15,
    scale       = 0.6,
    shear       = 5.0,
    perspective = 0.0005,
    fliplr      = 0.5,
    flipud      = 0.1,

    hsv_h   = 0.02,
    hsv_s   = 0.8,
    hsv_v   = 0.5,
    erasing = 0.4,

    mosaic       = 1.0,
    mixup        = 0.15,
    copy_paste   = 0.1,
    close_mosaic = 15,

    amp   = True,
    plots = True,
)

print(f"\nBest weights : {results.save_dir}/weights/best.pt")
print(f"Last weights : {results.save_dir}/weights/last.pt")
