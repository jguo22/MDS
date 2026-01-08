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
        'data': str(SCRIPT_DIR / 'datasets/combined1/data.yaml'),
        'model': str(SCRIPT_DIR / 'yolo11n.pt'),         # Pretrained model
        'epochs': 300,                   # Number of training epochs
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
        'patience': 50,                # Early stopping patience
        'project': str(runs_dir),      # Directory to save results
        'name': run_name,              # Auto-incremented run name
        'exist_ok': False,             # Don't overwrite existing experiments

        # Data Augmentation
        'fliplr': 0.5,                 # Horizontal flip probability
        'flipud': 0.1,                 # Vertical flip probability
        'hsv_h': 0.015,                # Hue augmentation (0-1)
        'hsv_s': 0.7,                  # Saturation augmentation (0-1)
        'hsv_v': 0.4,                  # Brightness/Value augmentation (0-1)
        'degrees': 10.0,               # Rotation range (+/- deg)
        'translate': 0.1,              # Translation (+/- fraction)
        'scale': 0.5,                  # Image scale (+/- gain)
        'shear': 2.0,                  # Shear angle (+/- deg)
        'perspective': 0.0001,         # Perspective distortion (0-0.001)
        'mosaic': 1.0,                 # Mosaic augmentation (probability)
        'mixup': 0.0,                  # MixUp augmentation (probability)
        'copy_paste': 0.0,             # Copy-paste augmentation (probability)

        # Transfer Learning
        # Freeze first N layers (None=auto, 0=train all)
        'freeze': 10,
    }

    # Initialize model
    model = YOLO(config['model'])

    # Start training
    results = model.train(**config)

    # Get the path where YOLO saved the best model
    model_path = Path(config['project']) / \
        config['name'] / 'weights' / 'best.pt'
    print(f"Training completed. Best model saved to: {model_path}")


if __name__ == "__main__":
    train_model()
