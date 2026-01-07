import os
from pathlib import Path
from ultralytics import YOLO

def train_model():
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent.absolute()
    
    # Configuration
    config = {
        'data': str(SCRIPT_DIR / 'datasets' / 'Pringles1' / 'data.yaml'),  # Path to your data.yaml
        'model': str(SCRIPT_DIR / 'yolo11n.pt'),         # Pretrained model
        'epochs': 5,                   # Number of training epochs
        'imgsz': 640,                  # Image size
        'device': 'mps',               # Use MPS (Metal Performance Shaders) for Apple Silicon
        'batch': 16,                   # Batch size
        'workers': 4,                  # Number of worker threads for data loading
        'project': str(SCRIPT_DIR / 'runs' / 'train'),  # Directory to save results
        'name': 'exp',                 # Experiment name
        'exist_ok': True,              # Overwrite existing experiment
        'optimizer': 'auto',           # Optimizer to use (auto, SGD, Adam, AdamW)
        'lr0': 0.01,                   # Initial learning rate
        'lrf': 0.01,                   # Final learning rate (lr0 * lrf)
        'weight_decay': 0.0005,        # Weight decay
        'patience': 50,                # Early stopping patience
    }

    # Initialize model
    model = YOLO(config['model'])
    
    # Start training
    results = model.train(
        data=config['data'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        device=config['device'],
        batch=config['batch'],
        workers=config['workers'],
        project=config['project'],
        name=config['name'],
        exist_ok=config['exist_ok'],
        optimizer=config['optimizer'],
        lr0=config['lr0'],
        lrf=config['lrf'],
        weight_decay=config['weight_decay'],
        patience=config['patience']
    )

    # Save the trained model
    save_path = Path(config['project']) / config['name'] / 'weights' / 'best.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    model.save(str(save_path))
    print(results)
    print(f"Training completed. Model saved to {save_path}")

if __name__ == "__main__":
    train_model()