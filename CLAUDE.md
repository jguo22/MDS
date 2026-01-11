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
- **Message Framing**: Custom framing with 8-byte length headers (`struct.pack('!Q', len(data))`)
- **Video Frames**: 4-byte frame_id + JPEG-encoded data
- **Command Messages**: Generic message protocol with 1-byte message type + fixed-length float arguments
  - Message Type 0 (Close): No arguments - gracefully closes connection
  - Message Type 1 (Movement): `[left_coef, right_coef, distance]` (3 floats)
  - Format: `struct.pack('!B', msg_type) + struct.pack('!Nf', *args)` where N is defined per message type
  - Argument counts are validated based on message type in `config.MESSAGE_ARG_COUNTS`
  - Extensible design allows adding new message types (0-255) with different argument counts
- **Why TCP not RTP**: Provides reliable, ordered delivery with automatic retransmission. RTP/UDP would offer lower latency but no delivery guarantees.

**Running on Raspberry Pi:**
```bash
# Run main_pi.py to send video and receive commands
python3 main_pi.py --camera usb0

# Options:
#   --camera: usb0, usb1, picamera0, etc. (default: usb0)
#
# All other settings (host, ports, FPS, reconnect delay) are configured in connection/config.py
```

**Running on Computer:**
```bash
# Run main_comp.py for full control interface with interactive input
python3 main_comp.py

# Or run basic ComputerReceiver module
python3 -m connection.computer_receiver

# The computer acts as a server listening for Pi connections
```

**main_comp.py features:**
- Interactive terminal input for manual movement commands
- Click-to-move on video window
- Movement command queueing and timing
- Automatic reconnection handling

### Development Workflow

**Typical startup sequence:**

1. **Start the Raspberry Pi first:**
   ```bash
   # On Raspberry Pi
   python3 main_pi.py --camera usb0
   ```
   The Pi will continuously attempt to connect to the computer (configured in `connection/config.py`).
   It will retry every 5 seconds (default `RECONNECT_DELAY`) until successful.

2. **Start the computer receiver:**
   ```bash
   # On Computer
   python3 main_comp.py
   ```
   The Pi will automatically connect within a few seconds and begin streaming video.
   You can now send movement commands by typing in the terminal or clicking on the video window.

3. **Making changes to computer code:**
   - Press `Ctrl+C` on the computer to stop the receiver
   - Make your code changes
   - Restart: `python3 main_comp.py`
   - **The Pi will automatically reconnect** (no need to restart it)

4. **Making changes to Pi code:**
   - Press `Ctrl+C` on the Raspberry Pi
   - Make your code changes
   - Restart: `python3 main_pi.py --camera usb0`
   - The Pi will reconnect to the computer

**Key benefits:**
- **Pi-initiated reconnection**: The Pi actively tries to connect, so you can restart the computer at any time
- **No manual reconnection**: After stopping either side, just restart and they'll automatically reconnect
- **Rapid iteration**: Modify computer vision code on the computer, restart receiver, and the Pi immediately reconnects
- **Persistent camera**: The Pi keeps the camera open across reconnections, avoiding reinitialization delays

**Interactive manual control:**

When running `main_comp.py`, you can manually send movement commands by typing in the terminal:
```bash
# Type two numbers (left_coef right_coef) and press Enter
Enter movement (left right): 0.5 0.5    # Move forward
Enter movement (left right): -0.5 0.5   # Turn left
Enter movement (left right): 0.5 -0.5   # Turn right
Enter movement (left right): 0 0        # Stop

# Optional: specify distance as third parameter
Enter movement (left right): 0.5 0.5 200.0
```

This allows you to test movement commands interactively without modifying code or clicking on the video window.

**Common scenarios:**

| Scenario | Action |
|----------|--------|
| Testing movement commands interactively | Run `main_comp.py`, type coefficients in terminal (e.g., `0.5 0.5`) |
| Testing movement via clicking | Run `main_comp.py`, click on video window to send robot to that position |
| Modifying movement logic | Modify computer code → Ctrl+C → Restart → Pi auto-reconnects |
| Adjusting camera settings | Modify Pi code → Ctrl+C on Pi → Restart Pi |
| Network disconnection | Both sides handle gracefully → Auto-reconnect when network restored |
| Changing config (IP, ports, FPS) | Edit `connection/config.py` → Restart both sides |

**PiStreamer API (Single-Use Pattern):**
```python
from connection.PiStreamer import PiStreamer
from connection.CameraCapture import CameraCapture
import time

# Create camera once (reused across reconnections)
camera = CameraCapture("usb0", 640, 480)
camera.open()

# Movement callback
def handle_movement(left_coef, right_coef, distance):
    print(f"Move: L={left_coef}, R={right_coef}, D={distance}")
    # Control motors here

# Reconnection loop - create new PiStreamer for each connection
while True:
    # Create new streamer instance for this connection
    streamer = PiStreamer(camera=camera, host="192.168.1.101")
    streamer.set_movement_callback(handle_movement)

    # Connect and stream (blocks until disconnected)
    if streamer.connect():
        streamer.stream()  # Uses DEFAULT_MAX_FPS from config

    # Brief pause before reconnecting
    time.sleep(2.0)
```

**IMPORTANT:** PiStreamer is designed for single-use. Each instance handles one connection lifecycle.
For reconnection, create a new PiStreamer instance while reusing the same camera.

**ComputerReceiver API:**
```python
from connection import ComputerReceiver

receiver = ComputerReceiver()
receiver.start_servers()
receiver.wait_for_connection()

# Send movement command
receiver.send_movement(left_coef=0.5, right_coef=0.5, distance=100.0)

# Gracefully close Pi connection
receiver.send_close()
```

**Connection Features:**
- Single-use connection instances (no race conditions)
- Clean connection lifecycle (create → connect → stream → done)
- Camera management external to streamer (reuse across connections)
- Configurable socket timeouts and buffer sizes
- Frame rate limiting
- Thread-safe command message reception
- Extensible message protocol for future command types

**Configuration (connection/config.py):**
- `VIDEO_PORT`: Default 5000
- `COMMAND_PORT`: Default 5001
- `FRAME_WIDTH`, `FRAME_HEIGHT`: Video resolution (default 640x480)
- `JPEG_QUALITY`: Compression quality (default 80)
- `DEFAULT_MAX_FPS`: Default maximum FPS for streaming (default 30.0)
- `SOCKET_TIMEOUT`: Network timeout in seconds (default 180.0)
- `RECONNECT_DELAY`: Delay between reconnection attempts (default 5.0s)
- `BUFFER_SIZE`: Socket buffer size
- `MSG_TYPE_CLOSE`: Message type constant (0) for closing connection
- `MSG_TYPE_MOVEMENT`: Message type constant (1) for movement commands
- `MESSAGE_ARG_COUNTS`: Dict mapping message types to their expected argument counts

**Protocol API (connection/protocol.py):**
```python
# Send generic command message
protocol.send_command(socket, msg_type: int, args: list[float]) -> bool

# Receive generic command message (validates arg count based on message type)
msg_type, args = protocol.recv_command(socket) -> tuple[int, list[float]] | None

# Safely close socket with proper shutdown and error handling
protocol.close_socket(socket: Optional[socket.socket]) -> None

# Example: Send close command (type 0, no args)
protocol.send_command(sock, config.MSG_TYPE_CLOSE, [])

# Example: Send movement command (type 1, 3 args)
protocol.send_command(sock, config.MSG_TYPE_MOVEMENT, [0.5, 0.5, 100.0])

# Example: Receive and handle commands
result = protocol.recv_command(sock)
if result:
    msg_type, args = result
    if msg_type == config.MSG_TYPE_CLOSE:
        print("Connection closing...")
        protocol.close_socket(sock)
        break
    elif msg_type == config.MSG_TYPE_MOVEMENT:
        left_coef, right_coef, distance = args
        # Handle movement...

# Adding new message types:
# 1. Add to config.py: MSG_TYPE_CUSTOM = 2
# 2. Add to MESSAGE_ARG_COUNTS: MSG_TYPE_CUSTOM: 4
# 3. Handle in receiver code
```

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
See the **Development Workflow** section under "Remote Communication System" for detailed startup and reconnection procedures.

**Integration with motor control:**
1. **On Raspberry Pi**: `main_pi.py` runs PiStreamer to stream camera feed and receive movement commands
2. **On Computer**: Run ComputerReceiver to display video and send movement commands
3. **Movement callback**: PiStreamer callback translates commands to Raven motor control via `nav.startPath()`
4. **Automatic reconnection**: Pi continuously attempts to reconnect, enabling rapid development iteration

**Key characteristics:**
- TCP protocol ensures reliable command delivery but adds ~10-50ms latency vs UDP/RTP
- Pi-initiated reconnection allows restarting computer code without touching the robot
- Persistent camera across reconnections avoids reinitialization delays

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
- **Pi won't connect**: Verify Pi and computer are on same network, check firewall settings, verify `COMPUTER_IP` in config.py matches your computer's IP
- **Automatic reconnection**: Connection failures are expected and handled automatically - Pi retries every 5 seconds (configurable via `RECONNECT_DELAY`)
- **Restarting computer code**: Just Ctrl+C and restart - Pi will automatically reconnect within seconds (no need to restart Pi)
- **Video lag**: Reduce `JPEG_QUALITY` in config.py, lower resolution, or reduce `DEFAULT_MAX_FPS`
- **Dropped frames**: TCP guarantees delivery but can cause frame buildup under poor network conditions; monitor frame_id gaps
- **Command latency**: TCP adds 10-50ms vs UDP; factor this into control loops for time-sensitive operations
- **Socket timeout errors**: Increase `SOCKET_TIMEOUT` in config.py for unreliable networks (default is 180s)

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
