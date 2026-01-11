# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the MASLAB 2026 team repository for building an autonomous robot. The project combines:
- **Raven motor controller**: Hardware interface for robot motor control
- **YOLO object detection**: Vision system using YOLOv11n for real-time object detection
- **Python-based control**: Main control logic in Python 3.11

## Hardware Components

### Raven Board
The Raven board is a motor controller accessed through the `raven` Python package. Key features:
- Motor control via `Raven.MotorChannel.CH1`, `CH2`, etc.
- Two control modes:
  - **DIRECT mode**: Set torque and speed factors directly
  - **Speed-controlled**: Use `set_motor_torque_factor()` to limit torque, `set_motor_speed_factor()` to set target speed
  - **Torque-controlled**: Set high speed factor, limit with torque factor
- Encoder access via `get_motor_encoder()` and `set_motor_encoder()`

Example workflow in `main.py`:
1. Initialize: `raven_board = Raven()`
2. Set mode: `raven_board.set_motor_mode(channel, Raven.MotorMode.DIRECT)`
3. Control: `raven_board.set_motor_speed_factor(channel, percentage, reverse=True/False)`

## Remote Communication System

### PiStreamer - TCP Video Streaming and Movement Control
The `connection/` module provides a bidirectional communication system between the Raspberry Pi and a computer. **Important:** This system uses **TCP sockets with a custom protocol**, not RTP/UDP.

**Architecture:**
- `connection/PiStreamer.py`: Raspberry Pi client that streams video and receives movement commands
- `connection/ComputerReceiver.py`: Computer server that receives video and sends movement commands
- `connection/protocol.py`: Custom TCP-based messaging protocol
- `connection/config.py`: Configuration (ports, timeouts, video settings)
- `connection/CameraCapture.py`: Unified camera interface (USB, PiCamera)

**Protocol Details:**
- **Transport**: TCP (`socket.SOCK_STREAM`) for reliable delivery
- **Message Format**: Custom framing with 8-byte length headers (`struct.pack('!Q', len(data))`)
- **Video Frames**: 4-byte frame_id + JPEG-encoded data
- **Movement Commands**: 12 bytes with three 4-byte floats (left_coef, right_coef, distance)
- **Why TCP not RTP**: Provides reliable, ordered delivery with automatic retransmission. RTP/UDP would offer lower latency but no delivery guarantees.

**Running on Raspberry Pi:**
```bash
# Run PiStreamer to send video and receive movement commands
python3 -m connection.pi_streamer --camera usb0 --host 192.168.1.101

# Options:
#   --camera: usb0, usb1, picamera0, etc.
#   --host: Computer IP address to connect to
#   --video-port: Video streaming port (default from config)
#   --movement-port: Movement command port (default from config)
```

**Running on Computer:**
```bash
# Run ComputerReceiver to receive video and send movement commands
python3 -m connection.computer_receiver

# The computer acts as a server listening for Pi connections
```

**PiStreamer API:**
```python
from connection import PiStreamer

# Initialize and configure
streamer = PiStreamer(host="192.168.1.101")
streamer.start_camera("usb0")

# Set callback for movement commands
def handle_movement(left_coef, right_coef, distance):
    print(f"Move: L={left_coef}, R={right_coef}, D={distance}")
    # Control motors here

streamer.set_movement_callback(handle_movement)

# Connect and stream
if streamer.connect():
    streamer.stream(max_fps=30.0)
```

**Connection Features:**
- Automatic reconnection with exponential backoff
- Camera auto-recovery on frame capture failures
- Configurable socket timeouts and buffer sizes
- Frame rate limiting
- Thread-safe movement command reception

**Configuration (connection/config.py):**
- `VIDEO_PORT`: Default 5000
- `MOVEMENT_PORT`: Default 5001
- `FRAME_WIDTH`, `FRAME_HEIGHT`: Video resolution (default 640x480)
- `JPEG_QUALITY`: Compression quality (default 80)
- `SOCKET_TIMEOUT`: Network timeout in seconds
- `BUFFER_SIZE`: Socket buffer size

## Vision System

### YOLO Detection Pipeline
Located in `yolo/` directory with the following structure:
- `yolo11n.pt`: YOLOv11n PyTorch model trained on COCO dataset
- `yolo11n_ncnn_model/`: NCNN-optimized model for embedded deployment
  - `model.ncnn.bin`: Binary model weights
  - `model.ncnn.param`: Model architecture parameters
  - `model_ncnn.py`: NCNN inference test script
  - `metadata.yaml`: Model configuration (80 COCO classes, 640x640 input)
- `yolo_detect.py`: Main detection script with full inference pipeline
- `runs/detect/`: Output directory for detection results

### Running YOLO Detection

The `yolo_detect.py` script supports multiple input sources:

```bash
# Image file
python3 yolo/yolo_detect.py --model yolo/yolo11n.pt --source image.jpg --thresh 0.5

# Image folder
python3 yolo/yolo_detect.py --model yolo/yolo11n.pt --source test_images/ --thresh 0.5

# Video file
python3 yolo/yolo_detect.py --model yolo/yolo11n.pt --source video.mp4 --thresh 0.5

# USB camera (index 0)
python3 yolo/yolo_detect.py --model yolo/yolo11n.pt --source usb0 --thresh 0.5

# PiCamera (index 0) - Raspberry Pi only
python3 yolo/yolo_detect.py --model yolo/yolo11n.pt --source picamera0 --thresh 0.5 --resolution 640x480

# Record video output
python3 yolo/yolo_detect.py --model yolo/yolo11n.pt --source usb0 --resolution 640x480 --record
```

**Key arguments:**
- `--model`: Path to YOLO model (.pt file)
- `--source`: Input source (image file, folder, video, usb0, picamera0)
- `--thresh`: Confidence threshold (default: 0.5)
- `--resolution`: Display resolution in WxH format (e.g., "640x480")
- `--record`: Record output video as "demo1.avi" (requires --resolution)

**Interactive controls during inference:**
- `q`: Quit
- `s`: Pause/resume
- `p`: Save current frame as "capture.png"

### YOLO Detection Pipeline Details

The `yolo_detect.py` script performs the following:
1. Loads YOLO model using Ultralytics (`model = YOLO(model_path, task='detect')`)
2. Reads frames from specified source
3. Runs inference: `results = model(frame, verbose=False)`
4. Extracts detections: `detections = results[0].boxes`
5. For each detection:
   - Extracts bounding box coordinates (xyxy format)
   - Gets class ID and name from `model.names`
   - Gets confidence score
   - Draws boxes and labels if confidence > threshold
6. Displays FPS for video/camera sources
7. Shows object count and visualization

**Model Format Notes:**
- PyTorch model (`yolo11n.pt`): Used with Ultralytics library for standard inference
- NCNN model (`yolo11n_ncnn_model/`): Optimized for embedded/mobile deployment with NCNN framework
  - NCNN provides faster inference on ARM/embedded devices
  - Test with: `python3 yolo/yolo11n_ncnn_model/model_ncnn.py`

## Python Environment

### Dependencies
Python 3.11 virtual environment in `venv/`. Key packages:
- `ultralytics==8.3.248`: YOLO training and inference
- `opencv-python==4.12.0.88`: Image processing and camera interface
- `torch==2.2.0`: PyTorch deep learning framework
- `ncnn==1.0.20250916`: NCNN inference framework
- `numpy==2.2.6`: Array operations
- `raven`: Motor controller interface (custom package from maslab-lib)

### Setup
Activate the virtual environment and install dependencies:
```bash
source venv/bin/activate
pip3 install -r requirements.txt
```

Or install manually:
```bash
source venv/bin/activate
pip3 install ultralytics ncnn numpy
pip3 install git+https://github.com/MASLAB/maslab-lib.git
```

The `raven` package is installed via maslab-lib and provides motor controller access.

**Important:** Always activate the venv before running scripts: `source venv/bin/activate`

## Custom Model Training

### Training Pipeline
The `yolo/train/` directory contains the complete training infrastructure:

**Training a Model:**
```bash
cd yolo/train
python3 train.py
```

The training script (`train.py`) automatically:
- Manages experiment numbering via `runs/train/exp{N}` directories
- Tracks run numbers in `runs/train/run_counter.txt`
- Saves best and last models to `runs/train/exp{N}/weights/`
- Uses transfer learning with `freeze: 10` (freezes first 10 layers)
- Configured for Apple Silicon with `device: 'mps'`

**Key Training Parameters in `train.py`:**
- Dataset: `datasets/combined1/data.yaml` (customizable)
- Base model: `yolo11n.pt` (pretrained YOLOv11n)
- Epochs: 300 with early stopping (`patience: 50`)
- Batch size: 16
- Image size: 640x640
- Learning rate: 0.01 → 0.0001 (linear decay)
- Data augmentation: flip, HSV, rotation, translation, scale, mosaic

**Validating a Model:**
```bash
cd yolo/train
python3 validate.py
```
Edit `validate.py` to change the model path (currently points to `exp2/weights/best.pt`).

**Checking Frozen Layers:**
```bash
cd yolo/train
python3 check_frozen_layers.py
```
This script shows which layers are trainable vs frozen in the base model.

### Dataset Structure
Datasets are organized in `yolo/train/datasets/` following YOLO format:
```
datasets/
├── combined1/          # Combined dataset
├── Pringles1/          # Individual Pringles dataset
└── Cheetos/            # Individual Cheetos dataset

Each dataset contains:
dataset_name/
├── data.yaml           # Dataset config (paths, class names, nc)
├── train/
│   ├── images/        # Training images
│   └── labels/        # Training labels (.txt, YOLO format)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**YOLO Label Format:**
Each `.txt` file contains one line per object:
```
class_id center_x center_y width height
```
All coordinates are normalized to [0, 1]. Empty `.txt` files indicate background images with no objects.

**Important:** To prevent false positives on backgrounds, include background-only images with empty label files (10-20% of dataset).

### Testing Custom Models
Test trained models on various sources:

```bash
# Test on dataset test images
python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp2/weights/best.pt --source yolo/train/datasets/combined1/test/images --thresh 0.2

# Test on live camera
python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp2/weights/best.pt --source usb0 --thresh 0.2

# Capture frame for debugging (press 'p' during inference)
python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp2/weights/best.pt --source usb0 --thresh 0.2
# Then test saved frame
python3 yolo/yolo_detect.py --model yolo/train/runs/train/exp2/weights/best.pt --source capture.png --thresh 0.2
```

**Threshold Tuning:** Custom models often need lower thresholds (0.2-0.4) compared to pretrained models (0.5+).

## Development Workflow

### Working with YOLO Models
1. **Training custom models**: Edit `yolo/train/train.py` to configure dataset and parameters, then run
2. **Model export**: Models can be exported to NCNN format for embedded deployment
3. **Testing**: Run detection on test images/videos before deploying to robot
4. **Tuning**: Adjust confidence threshold (`--thresh`) based on detection requirements
5. **Debugging**: Use profiler (`yolo/profiler.py`) to measure inference performance

### Motor Control Integration
When integrating vision with motor control:
1. Use YOLO to detect objects and get their positions
2. Calculate control signals based on detection results (e.g., bounding box centers)
3. Send commands to Raven board to actuate motors
4. Main control loop should handle both vision processing and motor updates

### Remote Operation Workflow
For remote control operation (computer controlling Pi robot):
1. **On Raspberry Pi**: Run PiStreamer to stream camera feed and receive movement commands
2. **On Computer**: Run ComputerReceiver to display video and send movement commands
3. **Integration**: PiStreamer movement callback translates commands to Raven motor control
4. **Example**: `main_pi.py` shows full integration of PiStreamer + Raven + optional YOLO

**Typical deployment:**
- Pi runs `main_pi.py` with PiStreamer for video streaming and movement callback for motor control
- Computer runs receiver to view video stream and issue movement commands (manual or YOLO-based)
- TCP protocol ensures reliable command delivery but adds ~10-50ms latency vs UDP/RTP

### Common Pitfalls and Solutions

**Detection Issues:**
- **False positives on backgrounds**: Add 10-20% background-only images with empty label files to training dataset
- **Low confidence scores**: Custom models often perform best with lower thresholds (0.2-0.4 vs 0.5 default)
- **Model not detecting objects**: Check if input resolution matches training size (640x640), verify model path

**Camera and Display:**
- **Camera resolution**: For PiCamera, always specify `--resolution` to avoid configuration issues
- **Resolution mismatch**: Match inference resolution to training resolution for best results (640x480 or 1280x720)
- **Interactive controls**: Use 'p' to capture frames, 's' to pause/resume, 'q' to quit

**Training:**
- **Model paths**: Use absolute paths or paths relative to project root
- **Dataset paths in data.yaml**: Use relative paths (`../train/images`) from dataset directory
- **Empty training runs**: Verify dataset has matching image/label file pairs with correct naming
- **Overfitting**: Increase data augmentation, add more training data, reduce `freeze` parameter

**Performance:**
- **Frame rate**: YOLO inference takes ~30-100ms per frame on typical hardware; factor this into control loops
- **Coordinate systems**: YOLO returns pixel coordinates in xyxy format; convert to robot coordinates for navigation
- **MPS acceleration**: Training uses Apple Silicon GPU (`device: 'mps'`); inference auto-detects available hardware

**Connection and Networking:**
- **Connection failures**: Verify Pi and computer are on same network, check firewall settings, verify correct IP addresses
- **Video lag**: Reduce JPEG_QUALITY in config.py, lower resolution, or reduce max_fps in stream() call
- **Dropped frames**: TCP guarantees delivery but can cause frame buildup under poor network conditions; monitor frame_id gaps
- **Movement command latency**: TCP adds 10-50ms vs UDP; factor this into control loops for time-sensitive operations
- **Socket timeout errors**: Increase SOCKET_TIMEOUT in config.py for unreliable networks

## File Organization

```
MDS/
├── main.py                    # Raven motor controller example
├── main_pi.py                 # Raspberry Pi main script with PiStreamer integration
├── requirements.txt           # Python dependencies (ultralytics, ncnn, numpy, maslab-lib)
├── out.jpg                   # Example output image
├── runs/                     # Root-level runs directory (generated)
├── connection/               # Remote communication system (TCP-based)
│   ├── __init__.py
│   ├── PiStreamer.py         # Pi-side video streamer and movement receiver
│   ├── ComputerReceiver.py   # Computer-side video receiver and movement sender
│   ├── protocol.py           # Custom TCP messaging protocol
│   ├── config.py             # Network and video configuration
│   └── CameraCapture.py      # Unified camera interface (USB/PiCamera)
└── yolo/
    ├── yolo11n.pt            # Pretrained YOLOv11n model (COCO dataset, 80 classes)
    ├── yolo_detect.py        # Main detection script for inference
    ├── profiler.py           # Performance profiling utility
    ├── turn.py               # (Placeholder)
    ├── yolo11n_ncnn_model/   # NCNN optimized model for embedded deployment
    │   ├── model.ncnn.bin
    │   ├── model.ncnn.param
    │   ├── model_ncnn.py
    │   └── metadata.yaml
    ├── runs/detect/          # Detection output directory (generated)
    └── train/
        ├── yolo11n.pt        # Base model for training
        ├── train.py          # Training script with auto-incrementing experiments
        ├── validate.py       # Model validation script
        ├── check_frozen_layers.py  # Utility to inspect layer freeze status
        ├── datasets/         # Training datasets
        │   ├── combined1/    # Primary combined dataset
        │   ├── Pringles1/
        │   └── Cheetos/
        └── runs/train/       # Training outputs (generated)
            ├── run_counter.txt
            ├── exp1/
            ├── exp2/
            └── exp{N}/
                └── weights/
                    ├── best.pt   # Best model checkpoint
                    └── last.pt   # Last epoch checkpoint
```

## Model Information

The YOLO11n model is trained on COCO dataset with 80 object classes including:
- People and vehicles: person, bicycle, car, motorcycle, bus, truck
- Animals: cat, dog, bird, horse, cow, sheep, etc.
- Common objects: chair, bottle, book, cell phone, laptop, etc.

Full class list available in `yolo/yolo11n_ncnn_model/metadata.yaml`.

Input size: 640x640 pixels
Stride: 32
Task: Object detection (bounding boxes)
