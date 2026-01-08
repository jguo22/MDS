from pathlib import Path
from ultralytics import YOLO

# Path to your pretrained model
SCRIPT_DIR = Path(__file__).parent.absolute()
model_path = SCRIPT_DIR / 'yolo11n.pt'

# Load the model
model = YOLO(model_path)

print("=" * 80)
print("YOLO11n Layer Freeze Status")
print("=" * 80)

# Get the PyTorch model
torch_model = model.model

frozen_count = 0
trainable_count = 0

print("\nLayers that will be TRAINED (requires_grad=True):")
print("-" * 80)
for name, param in torch_model.named_parameters():
    if param.requires_grad:
        print(f"  ✓ {name:60s} Shape: {str(tuple(param.shape)):20s}")
        trainable_count += 1

print("\n" + "=" * 80)
print(f"\nLayers that will be FROZEN (requires_grad=False):")
print("-" * 80)
for name, param in torch_model.named_parameters():
    if not param.requires_grad:
        print(f"  ✗ {name:60s} Shape: {str(tuple(param.shape)):20s}")
        frozen_count += 1

print("\n" + "=" * 80)
print(f"Summary:")
print(f"  Total trainable layers: {trainable_count}")
print(f"  Total frozen layers: {frozen_count}")
print(f"  Total parameters: {frozen_count + trainable_count}")
print("=" * 80)
