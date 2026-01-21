# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the MASLAB 2026 team repository for building an autonomous robot. The project combines:
- **Raven motor controller**: Hardware interface for robot motor control
- **Navigation system**: IMU-based odometry and path planning with smooth movement profiles
- **YOLO object detection**: Vision system using YOLOv11n for real-time object detection
- **Remote operation**: TCP-based video streaming with click-to-move interface
- **Pixel-to-3D transformation**: Camera calibration and homography for ground plane mapping
- **Python-based control**: Main control logic in Python 3.11

## Hardware Components

### Raven Board
The Raven board is a motor controller accessed through the local `raven.py` module. This is a custom serial communication interface that implements the Raven protocol over USB at 460800 baud.

**Key Features:**
- **Motor Channels**: 5 channels (CH1-CH5) via `Raven.MotorChannel`
- **Servo Channels**: 4 channels (CH1-CH4) via `Raven.ServoChannel`
- **Control Modes**:
  - **DISABLE**: Motors disabled
  - **DIRECT**: Set torque and speed factors directly
  - **POSITION**: PID position control using encoder counts
  - **VELOCITY**: PID velocity control using encoder counts/sec
- **PID Control**: Configurable P, I, D gains with effort limiting
- **Encoder Access**: Read/write encoder counts via `get_motor_encoder()` and `set_motor_encoder()`
- **Odometry Support**: Built-in odometry tracking with `get_odometry()` and `set_odometry()`
- **Angle Tracking**: IMU angle integration with `get_angle()` and `set_angle()`
- **Base Configuration**: Configurable wheel diameter and base width via `set_base()`

**Communication Protocol:**
- Serial interface with custom framing (0xAA start byte)
- CRC8 checksums for reliable communication
- Automatic retry mechanism for failed commands
- Message types for motors, servos, encoders, odometry, and configuration

**Example Workflow:**
```python
from raven import Raven

# Initialize (auto-detects serial port or specify manually)
raven = Raven()  # or Raven(port="/dev/ttyUSB0")

# DIRECT mode - manual speed control
raven.set_motor_mode(Raven.MotorChannel.CH1, Raven.MotorMode.DIRECT)
raven.set_motor_speed_factor(Raven.MotorChannel.CH1, 50, reverse=False)  # 50% speed

# POSITION mode - move to encoder position
raven.set_motor_mode(Raven.MotorChannel.CH2, Raven.MotorMode.POSITION)
raven.set_motor_pid(Raven.MotorChannel.CH2, p_gain=30, i_gain=10, d_gain=2, percent=50)
raven.set_motor_target(Raven.MotorChannel.CH2, 640.0)  # Target encoder count

# VELOCITY mode - maintain speed
raven.set_motor_mode(Raven.MotorChannel.CH3, Raven.MotorMode.VELOCITY)
raven.set_motor_pid(Raven.MotorChannel.CH3, p_gain=0, i_gain=5, d_gain=1, percent=25)
raven.set_motor_target(Raven.MotorChannel.CH3, 2300.0)  # Target encoder counts/sec

# Read encoder and velocity
encoder = raven.get_motor_encoder(Raven.MotorChannel.CH1)
velocity = raven.get_motor_velocity(Raven.MotorChannel.CH1)

# Configure base for odometry
raven.set_base(wheel_d=95.0, base_d=209.0)  # wheel diameter, base width in mm

# Read odometry
x, y = raven.get_odometry()  # Position in mm
angle = raven.get_angle()     # Heading in radians
```

## Navigation System

### Nav Class (nav.py)
The `Nav` class provides high-level navigation and odometry for the robot. It integrates the Raven motor controller with an IMU (BNO08x) for precise positioning and path following.

**Key Features:**
- **IMU Integration**: BNO08x sensor via I2C for heading tracking
- **Odometry**: Position tracking using motor encoders and IMU fusion
- **Path Planning**: Queue-based movement system with smooth transitions
- **Position Control**: Uses Raven's POSITION mode with PID control
- **Thread-Safe**: Movement queue can be safely updated from multiple threads

**Motor Configuration:**
- Left Motor: `Raven.MotorChannel.CH2`
- Right Motor: `Raven.MotorChannel.CH3`
- Wheel Diameter: 95mm
- Base Width: 209mm
- Encoder Ticks per Rotation: 64 × 50 = 3200

**Basic Usage:**
```python
from nav import Nav, NavMove

# Initialize navigation (must run on Raspberry Pi)
nav = Nav()

# Add single movement to queue
nav.addPath(NavMove(left=1.0, right=1.0, dist=1000, smooth=True))

# Override entire queue with new path
movements = [
    NavMove(left=1.0, right=1.0, dist=500, smooth=True),
    NavMove(left=-1.0, right=1.0, dist=800, smooth=False)
]
nav.overridePaths(movements)

# Start navigation loop (blocking)
nav.startLoop()
```

**NavMove Parameters:**
- `left`: Left motor coefficient (-1.0 to 1.0, negative = reverse)
- `right`: Right motor coefficient (-1.0 to 1.0, positive = forward)
- `dist`: Distance in encoder ticks
- `smooth`: If True, maintain velocity when transitioning to next move

**Helper Functions:**
```python
from nav import get_forward_mm, get_rotate

# Get movement tuple for forward motion
left_coef, right_coef, distance = get_forward_mm(200.0)  # 200mm forward

# Get movement tuple for rotation
left_coef, right_coef, distance = get_rotate(math.pi / 2)  # 90° CCW

# Example: Move forward 300mm then turn 180°
nav.overridePaths([
    NavMove(*get_forward_mm(300.0), smooth=True),
    NavMove(*get_rotate(math.pi), smooth=False)
])
```

**Odometry and Position:**
The Nav class automatically updates the Raven board's odometry system:
```python
# Get current position (x, y) in mm
x, y = nav.raven.get_odometry()

# Get current heading in radians
angle = nav.raven.get_angle()
```

**Control Loop Details:**
- Update Rate: 20 Hz (50ms frame time)
- Acceleration: 5.0 rotations/s² (reach max speed in 1s)
- Max Velocity: 3.0 rotations/s
- PID Gains:
  - Right Motor: P=25, I=5, D=0.13
  - Left Motor: P=20, I=5, D=0.1

**Important Notes:**
- IMU must be initialized **before** Raven board
- The `startLoop()` method is blocking and runs indefinitely
- Path smoothing uses velocity profiling for acceleration/deceleration
- Angle correction uses proportional and derivative control (ANGLE_PROP=5000, ANGLE_D=5000)

## Remote Communication System

### PiStreamer - TCP Video Streaming and Movement Control
The `connection/` module provides a bidirectional communication system between the Raspberry Pi and a computer. **Important:** This system uses **TCP sockets with a custom protocol**, not RTP/UDP.

**Architecture:**
- `connection/PiStreamer.py`: Raspberry Pi client that streams video and receives movement commands
- `connection/ComputerReceiver.py`: Computer server that receives video and sends movement commands
- `connection/protocol.py`: Custom TCP-based messaging protocol
- `connection/config.py`: Configuration (ports, timeouts, video settings)
- `connection/CameraCapture.py`: Unified camera interface (USB, PiCamera)
- `connection/message_types.py`: Message type constants and definitions
- `connection/frame_processor/`: Frame processing pipeline for computer-side video handling
  - `FrameProcessor.py`: Abstract base class for frame processors
  - `ClickProcessor.py`: Handles mouse clicks on video to send navigation commands
  - `SaveImageProcessor.py`: Saves frames to disk with cooldown

**Protocol Details:**
- **Transport**: TCP (`socket.SOCK_STREAM`) for reliable delivery
- **Message Framing**: Custom framing with 8-byte length headers (`struct.pack('!Q', len(data))`)
- **Video Frames**: 4-byte frame_id + JPEG-encoded data
- **Command Messages**: Generic message protocol with 1-byte message type + variable-length float arguments
  - Message Type 0 (CLOSE): No arguments - gracefully closes connection
  - Message Type 1 (ADD_MOVEMENT): `[left_coef, right_coef, distance]` (3 floats) - adds one movement to queue
  - Message Type 2 (OVERRIDE_MOVEMENTS): `[l1, r1, d1, l2, r2, d2, ...]` (multiples of 3 floats) - replaces entire movement queue
  - Format: `struct.pack('!B', msg_type) + struct.pack('!Nf', *args)` where N is the number of floats
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

# Send single movement command (adds to queue)
receiver.add_movement(left_coef=0.5, right_coef=0.5, distance=100.0)

# Send X/Y coordinate (automatically plans path: rotate + forward)
receiver.send_xy(x=200.0, y=150.0)  # x, y in mm

# Override entire movement queue
movements = [1.0, 1.0, 500.0, -1.0, 1.0, 800.0]  # [l1, r1, d1, l2, r2, d2]
receiver.override_movement(movements)

# Gracefully close Pi connection
receiver.send_close()
```

### Frame Processor Architecture

The `connection/frame_processor/` module provides an extensible architecture for processing video frames on the computer side.

**FrameProcessor Interface:**
```python
from connection.frame_processor.FrameProcessor import FrameProcessor
import numpy as np
from typing import Optional, Tuple

class CustomProcessor(FrameProcessor):
    def process(self, frame: np.ndarray, frame_id: int) -> Optional[Tuple[float, float, float]]:
        # Process the frame
        # Return movement command tuple (left_coef, right_coef, distance) or None
        return None
```

**Built-in Processors:**

1. **ClickProcessor**: Click-to-move interface
   ```python
   from connection.frame_processor.ClickProcessor import ClickProcessor
   from connection.ComputerReceiver import ComputerReceiver

   receiver = ComputerReceiver()
   processor = ClickProcessor(receiver, window_name="Pi Camera")

   # Now clicks on the video window will:
   # 1. Convert pixel coordinates to 3D ground plane coordinates
   # 2. Send X/Y navigation command via receiver.send_xy()
   # 3. Robot automatically plans and executes: rotate → forward
   ```

2. **SaveImageProcessor**: Automatic frame capture
   ```python
   from connection.frame_processor.SaveImageProcessor import SaveImageProcessor

   processor = SaveImageProcessor(
       cooldown_seconds=1.0,  # Minimum time between saves
       output_dir="images"     # Output directory
   )
   # Frames saved as: frame_YYYYMMDD_HHMMSS_timestamp.jpg
   ```

**Integrating with ComputerReceiver:**
Frame processors are integrated into `main_comp.py` to process incoming video frames and generate navigation commands.

### Pixel-to-3D Transformation (pixelTo3D.py)

The `pixelTo3D` module converts pixel coordinates from the camera to real-world coordinates on the ground plane.

**Key Components:**
- **Camera Calibration**: Intrinsic matrix and distortion coefficients from camera calibration
- **Homography Matrix**: 3x3 transformation from image plane to ground plane (in mm)
- **Coordinate Systems**:
  - Pixel coordinates: Origin at top-left, u increases right, v increases down
  - Robot coordinates: Origin at camera, x increases forward, y increases left

**Main Function:**
```python
from pixelTo3D import transform_uv_to_xy

# Convert pixel click to ground plane coordinates
x_mm, y_mm = transform_uv_to_xy(u=320, v=240)  # Center of 640x480 frame

# x: forward distance from camera (mm)
# y: lateral distance from camera, positive = left (mm)
```

**Camera Calibration Data:**
```python
CAMERA_MATRIX = np.array([
    [900.83, 0, 319.14],
    [0, 905.18, 236.54],
    [0, 0, 1]
])

DISTORTION = np.array([
    [0.0963, 0.7159, 0.0037, 0.0118, -6.7639]
])
```

**Homography Matrix:**
The homography matrix `h` is precalculated from camera calibration and transforms image points to ground plane points:
```python
h = np.array([
    [-6.09741811e-01, -2.09501156e-02, 1.57159029e+02],
    [2.85708157e-02, 9.39073240e-03, -5.18950601e+02],
    [6.09393965e-04, -7.78799171e-03, 1.00000000e+00]
])
```

**Usage in Click-to-Move:**
1. User clicks on video frame at pixel (u, v)
2. `transform_uv_to_xy(u, v)` converts to ground plane (x, y) in mm
3. `ComputerReceiver.send_xy(x, y)` calculates rotation angle and distance
4. Movement commands sent: rotate to face target, then drive forward
5. Robot executes smooth path to clicked location

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

**Message Types (connection/message_types.py):**
- `CLOSE` (0): Close connection gracefully
- `ADD_MOVEMENT` (1): Add single movement to queue (3 floats: left_coef, right_coef, distance)
- `OVERRIDE_MOVEMENTS` (2): Replace entire movement queue (multiples of 3 floats)

**Protocol API (connection/protocol.py):**
```python
from connection import protocol, message_types

# Send generic command message
protocol.send_command(socket, msg_type: int, args: list[float]) -> bool

# Receive generic command message
msg_type, args = protocol.recv_command(socket) -> tuple[int, list[float]] | None

# Safely close socket with proper shutdown and error handling
protocol.close_socket(socket: Optional[socket.socket]) -> None

# Example: Send close command
protocol.send_command(sock, message_types.CLOSE, [])

# Example: Send single movement (add to queue)
protocol.send_command(sock, message_types.ADD_MOVEMENT, [0.5, 0.5, 100.0])

# Example: Override movement queue
movements = [1.0, 1.0, 500.0, -1.0, 1.0, 800.0]  # Two movements
protocol.send_command(sock, message_types.OVERRIDE_MOVEMENTS, movements)

# Example: Receive and handle commands
result = protocol.recv_command(sock)
if result:
    msg_type, args = result
    if msg_type == message_types.CLOSE:
        print("Connection closing...")
        protocol.close_socket(sock)
        break
    elif msg_type == message_types.ADD_MOVEMENT:
        left_coef, right_coef, distance = args
        nav.addPath(NavMove(left_coef, right_coef, distance, smooth=True))
    elif msg_type == message_types.OVERRIDE_MOVEMENTS:
        # Parse movements in groups of 3
        movements = []
        for i in range(0, len(args), 3):
            movements.append(NavMove(args[i], args[i+1], args[i+2], smooth=True))
        nav.overridePaths(movements)

# Adding new message types:
# 1. Add to message_types.py: NEW_TYPE = 3
# 2. Add to messageTypes list
# 3. Handle in receiver code (PiStreamer or ComputerReceiver)
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
Python 3.11 project using `pyproject.toml` for dependency management. Key packages:
- `ultralytics`: YOLO training and inference
- `ncnn`: NCNN inference framework for embedded deployment
- `numpy`: Array operations and numerical computing
- `opencv-python`: Computer vision and image processing
- `spatialmath-python`: Spatial mathematics library (rotation matrices, transformations used in pixelTo3D.py)
- `adafruit-circuitpython-bno08x`: BNO08x IMU sensor library (I2C communication)
- `pyserial`: Serial port communication (required by `raven.py`)

**Local Modules:**
- `raven.py`: Motor controller interface implementing Raven serial protocol
- `nav.py`: Navigation system with IMU integration and path planning
- `connection/`: Remote communication system (TCP video streaming)
- `yolo/`: YOLO detection pipeline

### Setup
Install the project and dependencies:
```bash
# Install project in development mode
pip install -e .

# Or install from requirements.txt (legacy)
pip install -r requirements.txt
```

**Important Notes:**
- The `raven` module is implemented locally in `raven.py` (not an external package)
- The project uses `pyproject.toml` for modern Python packaging
- Always activate your virtual environment before running scripts
- On Raspberry Pi, ensure I2C is enabled for IMU communication (`sudo raspi-config`)

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
1. **On Raspberry Pi**: `main_pi.py` initializes Nav system and runs PiStreamer to stream camera feed
2. **On Computer**: `main_comp.py` runs ComputerReceiver with ClickProcessor for interactive control
3. **Movement Flow**:
   - User clicks on video window at pixel (u, v)
   - ClickProcessor converts to ground plane coordinates (x, y) via `transform_uv_to_xy()`
   - `ComputerReceiver.send_xy(x, y)` calculates rotation and forward movement using `nav.get_rotate()` and `nav.get_forward_mm()`
   - OVERRIDE_MOVEMENTS message sent with both commands (rotate + forward)
   - PiStreamer receives commands and calls `nav.overridePaths()` to update movement queue
   - Nav system executes smooth path: rotate to face target, then drive forward
4. **Automatic reconnection**: Pi continuously attempts to reconnect, enabling rapid development iteration

**Key characteristics:**
- TCP protocol ensures reliable command delivery but adds ~10-50ms latency vs UDP/RTP
- Pi-initiated reconnection allows restarting computer code without touching the robot
- Persistent camera across reconnections avoids reinitialization delays
- Click-to-move provides intuitive visual navigation interface
- Movement queue allows complex multi-step paths

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

**Navigation and Movement:**
- **Robot doesn't move**: Check that Nav instance is running `startLoop()` in a thread, verify motors are in POSITION mode, check encoder connections
- **IMU initialization fails**: Ensure IMU is initialized **before** Raven board, check I2C connections and address, verify BNO08x library installation
- **Inaccurate odometry**: Recalibrate wheel diameter and base width measurements, check for wheel slippage, verify encoder tick counts
- **Click-to-move goes wrong direction**: Verify homography matrix is calibrated for current camera mounting, check coordinate system orientation (x forward, y left)
- **Robot overshoots target**: Reduce `max_velocity` or `acceleration` in Nav class, tune PID gains, check for motor saturation
- **Movement queue not updating**: Verify thread-safe access with `_lock`, check that movement commands are formatted correctly (multiples of 3 floats for OVERRIDE_MOVEMENTS)
- **Rotation errors**: Recalibrate IMU, check `BASE_D` and `WHEEL_D` constants, verify `TURN_CONSTANT` calculation matches physical robot
- **Homography transformation incorrect**: Recalibrate camera using checkerboard pattern, verify camera matrix and distortion coefficients, ensure camera mounting hasn't changed

## File Organization

```
MDS/
├── main.py                    # Raven motor controller example
├── main_pi.py                 # Raspberry Pi main script with Nav + PiStreamer
├── main_comp.py               # Computer main script with ComputerReceiver + ClickProcessor
├── raven.py                   # Raven motor controller serial interface (local implementation)
├── nav.py                     # Navigation system with IMU, odometry, and path planning
├── pixelTo3D.py              # Pixel-to-3D transformation (camera calibration + homography)
├── robot.py                   # Robot control script
├── requirements.txt           # Python dependencies (legacy, use pyproject.toml)
├── pyproject.toml             # Modern Python project configuration and dependencies
├── out.jpg                   # Example output image
├── runs/                     # Root-level runs directory (generated)
├── images/                   # Saved frames from SaveImageProcessor (generated)
├── connection/               # Remote communication system (TCP-based)
│   ├── __init__.py
│   ├── PiStreamer.py         # Pi-side video streamer and movement receiver
│   ├── ComputerReceiver.py   # Computer-side video receiver and movement sender
│   ├── protocol.py           # Custom TCP messaging protocol
│   ├── config.py             # Network and video configuration
│   ├── message_types.py      # Message type constants (CLOSE, ADD_MOVEMENT, OVERRIDE_MOVEMENTS)
│   ├── CameraCapture.py      # Unified camera interface (USB/PiCamera)
│   └── frame_processor/      # Frame processing architecture
│       ├── FrameProcessor.py      # Abstract base class
│       ├── ClickProcessor.py      # Click-to-move interface
│       └── SaveImageProcessor.py  # Frame capture to disk
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
