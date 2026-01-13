from pathlib import Path
from ultralytics import YOLO


def train_model():
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent.absolute()

    # Configuration
    runs_dir = SCRIPT_DIR / 'runs' / 'train'
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Track run number using a text file
    run_counter_file = runs_dir / 'run_counter.txt'
    if run_counter_file.exists():
        with open(run_counter_file, 'r') as f:
            run_number = int(f.read().strip())
        run_number += 1
    else:
        run_number = 1

    # Save the updated run number
    with open(run_counter_file, 'w') as f:
        f.write(str(run_number))

    run_name = f'exp{run_number}'

    config = {
        # Path to your data.yaml
        'data': str(SCRIPT_DIR / 'datasets/data1/data.yaml'),
        # Pretrained model
        'model': str(SCRIPT_DIR / 'yolo11n.pt'),
        'epochs': 30000,                   # Number of training epochs
        'imgsz': 640,                  # Image size
        # Use MPS (Metal Performance Shaders) for Apple Silicon
        'device': 'mps',
        'batch': 16,                   # Batch size
        'workers': 4,                  # Number of worker threads for data loading
        # Optimizer to use (auto, SGD, Adam, AdamW)
        'optimizer': 'auto',
        'lr0': 0.01,                   # Initial learning rate
        'lrf': 0.01,                   # Final learning rate (lr0 * lrf)
        'weight_decay': 0.0005,        # Weight decay
        'patience': 30000,                # Early stopping patience
        'project': str(runs_dir),      # Directory to save results
        'name': run_name,              # Auto-incremented run name
        'exist_ok': False,             # Don't overwrite existing experiments

        # Data Augmentation (conservative to avoid numerical warnings)
        'fliplr': 0.3,                 # Horizontal flip probability
        'hsv_h': 0.01,                 # Hue augmentation (minimal)
        'hsv_s': 0.3,                  # Saturation augmentation
        'hsv_v': 0.2,                  # Brightness augmentation
        'degrees': 10.0,               # Rotation range (+/- deg)
        'translate': 0.1,              # Translation - REDUCED to prevent overflow
        'scale': 0.3,                  # Scale variation - REDUCED to prevent overflow
        'shear': 0.0,                  # Shear disabled - can cause numerical issues
        'perspective': 0.0,            # Perspective disabled - causes divide by zero
        'mosaic': 1.0,                 # Mosaic augmentation
        'mixup': 0.0,                  # MixUp disabled
        'copy_paste': 0.0,             # Copy-paste disabled
        'erasing': 0.0,                # Erasing disabled - can cause issues with segmentation

        # Transfer Learning
        # Freeze first N layers (None=auto, 0=train all)
        'freeze': 11,
    }

    # Initialize model
    # model = YOLO(config['model'])
    model = YOLO("yolo11n-seg.pt")

    # Start training
    results = model.train(**config)

    # Get the path where YOLO saved the best model
    model_path = Path(config['project']) / \
        config['name'] / 'weights' / 'best.pt'
    print(f"Training completed. Best model saved to: {model_path}")


if __name__ == "__main__":
    train_model()
