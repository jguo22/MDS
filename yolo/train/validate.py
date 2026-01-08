from pathlib import Path
from ultralytics import YOLO

script_dir = Path(__file__).parent.absolute()
model_dir = script_dir / 'runs/train/exp7/weights/best.pt'
model = YOLO(model_dir)
metrics = model.val()  # Run validation

print(f"validating model {model_dir}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
