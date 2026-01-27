# CLAUDE.md

Project guidance for Claude Code working with the MASLAB 2026 autonomous robot codebase.

## Project Overview

Autonomous robot combining:
- **Raven motor controller**: Custom serial interface (460800 baud) for motor/servo control
- **Navigation**: IMU-based (BNO08x) odometry with queue-based path planning
- **Vision**: Custom YOLOv11 segmentation for game objects (zones, cans, robots)
- **Remote operation**: TCP video streaming (640x480) with click-to-move
- **Pixel-to-3D**: Camera calibration for ground plane mapping
- **Python 3.11**: Main control logic

## Hardware Components

### Raven Board (`raven.py`)
Custom serial motor controller (460800 baud, 0xAA framing, CRC8 checksums).

**Features:**
- 5 motor channels, 4 servo channels
- Control modes: DISABLE, DIRECT (speed), POSITION (PID encoder), VELOCITY (PID speed)
- Built-in odometry: `get_odometry()` → (x, y) in mm, `get_angle()` → heading in radians
- Base config: `set_base(wheel_d=95.0, base_d=209.0)`

**Quick Example:**
```python
from raven import Raven
raven = Raven()  # Auto-detect serial port
raven.set_motor_mode(Raven.MotorChannel.CH2, Raven.MotorMode.POSITION)
raven.set_motor_pid(Raven.MotorChannel.CH2, p_gain=30, i_gain=10, d_gain=2, percent=50)
raven.set_motor_target(Raven.MotorChannel.CH2, 640.0)
x, y = raven.get_odometry()
angle = raven.get_angle()
```

## Navigation System

### IMUWrapper (`IMUWrapper.py`)
BNO08x IMU wrapper with automatic offset calibration (I2C 800kHz).
- **CRITICAL**: Initialize IMU **before** Raven board
- Heading relative to startup (not magnetic north)
- `imu.get_heading()` → angle in radians

### Nav Class (`nav.py`)
High-level navigation integrating Raven + IMU with queue-based path planning.

**Configuration:**
- Motors: CH2 (left), CH3 (right) | Wheel: 95mm | Base: 209mm | Ticks: 3200/rotation
- Control: 20 Hz, 5.0 rot/s² accel, 3.0 rot/s max velocity
- PID: Right (P=25, I=5, D=0.13), Left (P=20, I=5, D=0.1)

**Usage:**
```python
from nav import Nav, NavMove, get_forward_mm, get_rotate

nav = Nav()
nav.overridePaths([
    NavMove(*get_forward_mm(300.0), smooth=True),  # Forward 300mm
    NavMove(*get_rotate(math.pi), smooth=False)    # Turn 180°
])
nav.startLoop()  # Blocking

# NavMove(left_coef, right_coef, dist_ticks, smooth)
# left/right: -1.0 to 1.0, dist: encoder ticks, smooth: maintain velocity
```

## Remote Communication System

TCP-based bidirectional communication between Pi (client) and computer (server).

**Architecture:**
- `PiStreamer.py`: Pi video streamer + movement receiver
- `ComputerReceiver.py`: Computer video receiver + movement sender
- `protocol.py`: TCP messaging (8-byte headers, JPEG frames)
- `message_types.py`: CLOSE (0), ADD_MOVEMENT (1), OVERRIDE_MOVEMENTS (2)
- `FrameSaver.py`: Frame saver (auto-save frames with cooldown)

**Protocol:** TCP SOCK_STREAM, custom framing, 1-byte msg type + N floats
- ADD_MOVEMENT: [left, right, dist] (3 floats)
- OVERRIDE_MOVEMENTS: [l1, r1, d1, l2, r2, d2, ...] (multiples of 3)
- TCP chosen for reliability vs UDP/RTP (10-50ms latency trade-off)

**Quick Start:**
```bash
# Pi (starts first, auto-reconnects every 5s)
python3 main_pi.py --camera usb0

# Computer (server, click video or type "0.5 0.5" for movement)
python3 main_comp.py
```

**Config:** `connection/config.py` - VIDEO_PORT (5000), COMMAND_PORT (5001), 640x480, JPEG_QUALITY (80), 30 FPS, 180s timeout

**Development Workflow:**
1. Start Pi first: `python3 main_pi.py --camera usb0` (retries every 5s)
2. Start computer: `python3 main_comp.py` (auto-connects)
3. Modify computer code → Ctrl+C → restart → Pi auto-reconnects (no Pi restart needed)
4. Modify Pi code → restart Pi → auto-reconnects
5. Type commands: `0.5 0.5` or click video for movement

**Key APIs:**
```python
# PiStreamer (single-use, recreate per connection)
camera = CameraCapture("usb0", 640, 480)
streamer = PiStreamer(camera=camera, host="192.168.1.101")
streamer.set_movement_callback(lambda l, r, d: nav.addPath(NavMove(l, r, d, True)))
streamer.connect() and streamer.stream()

# ComputerReceiver
receiver = ComputerReceiver()
receiver.add_movement(0.5, 0.5, 100.0)  # Add to queue
receiver.send_xy(200.0, 150.0)          # Auto-plan: rotate + forward
receiver.override_movement([1.0, 1.0, 500.0, -1.0, 1.0, 800.0])

# Frame Processors
# - InputProcessor: Click-to-move (pixel → ground plane → send_xy())
# - FrameSaver: Auto-save frames with cooldown
```

### Pixel-to-3D Transformation (`yolo/pixelTo3D.py`)

Converts pixel coordinates to ground plane (mm) using camera calibration.

**Key Function:** `transform_uv_to_xy(pixel_x, pixel_y)` → `(x, y)` in mm or `None`
- Undistorts → normalizes → applies camera rotation (SO3.Rx(-1)) → projects to z=0 plane
- **640x480 resolution required** (match camera/inference resolution)
- Coordinates: x forward, y left (positive=left, negative=right)

**Calibration Data:** CAMERA_MATRIX (fx, fy, cx, cy), DISTORTION (5 coeffs), ANGLE_MATRIX (tilt)

**Usage:** Click-to-move: pixel → `transform_uv_to_xy()` → `send_xy()` → rotate + drive

### Coordinate Transformations (`coordinates/relativeCoordinates.py`)

**CRITICAL**: Always use this module. DO NOT manually implement rotation math.

SE(2) transformations using `spatialmath` library:
- `world_to_relative(world_point, robot_pose)` → (local_x, local_y)
- `relative_to_world(rel_point, robot_pose)` → (world_x, world_y)
- `angle_between_points(p1, p2)` → angle in radians

```python
from spatialmath import SE2
from coordinates.relativeCoordinates import world_to_relative

robot_pose = SE2(robot_x, robot_y, robot_heading)
local_x, local_y = world_to_relative(target_point, robot_pose)
if 0 <= local_x <= 500 and -150 <= local_y <= 150:
    # Point is in front of robot
```

### Camera Calibration (`calibration/`)

- `distortion_calibration.py`: Checkerboard → CAMERA_MATRIX + DISTORTION
- `manual_calibration.py`: Ground plane homography → ANGLE_MATRIX
- Recalibrate when: camera changes, mount changes, coordinates drift

### Protocol API (`connection/protocol.py`)
```python
from connection import protocol, message_types

protocol.send_command(sock, message_types.ADD_MOVEMENT, [0.5, 0.5, 100.0])
protocol.send_command(sock, message_types.OVERRIDE_MOVEMENTS, [l1, r1, d1, ...])
msg_type, args = protocol.recv_command(sock)
protocol.close_socket(sock)
```

## Vision System

### YOLO Segmentation (Custom Model)

**Primary model:** `yolo/last.pt` (YOLOv11n, 8.1MB) - 8 classes for MASLAB 2026

**Classes:** Boundary, Golden Can/Zone, Green Can/Zone, Red Can/Zone, Robot

**Pipeline:**
```python
from yolo.segment import segmentImage
result = segmentImage(image)  # thresh=0.25
# result.boxes, result.masks, result.names
```

**Mask Refinement** (`mask_utils.py`):
1. YOLO segmentation → rough masks
2. `fixSegmentation()`: HSV-based colored tape detection (dom_sat_min=70, hue_tolerance=15)
3. `calculateQuadFromMask()`: Douglas-Peucker quad fitting with area preservation

**Detection:** `yolo_detect.py` supports images/folders/video/usb0/picamera0
```bash
python3 yolo/yolo_detect.py --model yolo/last.pt --source usb0 --thresh 0.25 --resolution 640x480
# Controls: q (quit), s (pause), p (save frame)
```

**Models:**
- `last.pt`/`best.pt`: Custom segmentation (8 classes)
- `yolo11n.pt`: Pretrained detection (COCO, 80 classes)
- `yolo11n_ncnn_model/`: NCNN for embedded (faster ARM inference)

## Python Environment

**Python 3.11** with `pyproject.toml` dependencies:
- ultralytics, ncnn, numpy, opencv-python, spatialmath-python, adafruit-circuitpython-bno08x, pyserial

**Setup:** `pip install -e .` (or `pip install -r requirements.txt`)
**Pi setup:** Enable I2C via `sudo raspi-config`

## Custom Model Training

**Train:** `cd yolo/train && python3 train.py`
- Auto-increments `runs/train/exp{N}` directories
- Base: `yolo11n.pt`, freeze: 10 layers, device: 'mps' (Apple Silicon)
- 300 epochs (patience: 50), batch: 16, img: 640x640, LR: 0.01→0.0001
- Outputs: `runs/train/exp{N}/weights/{best,last}.pt`

**Validate:** `python3 validate.py`
**Check layers:** `python3 check_frozen_layers.py`

**Dataset Structure:** `datasets/{name}/` with `data.yaml`, `train/valid/test/{images,labels}`
- YOLO format: `class_id center_x center_y width height` (normalized [0,1])
- Include 10-20% background images with empty `.txt` files

**Test:** `python3 yolo/yolo_detect.py --model runs/train/exp2/weights/best.pt --source usb0 --thresh 0.2`
- Custom models use lower thresholds (0.2-0.4) vs pretrained (0.5+)

## Game-Specific Detection

### Zone Detection (`zone_utils.py`)
`getZones(result, image)` → `(quads_xy, class_names)` - 6 zones (3 ours, 3 opponent)
- Extracts quads from masks → `fixSegmentation()` → `calculateQuadFromMask()`
- Transforms pixels to ground plane (mm) via `transform_uv_to_xy()`
- Filters: Green/Golden/Red zones only

**Field Config** (`config.py`):
```python
BACK_BORDER_X = -304.8 | LEFT_BORDER_Y = 1219.2 | RIGHT_BORDER_Y = -1219.2
CAN_DIAMETER = 76.2
```

### Can Detection (`can_utils.py`)
`getCans(result, image)` → `(can_locations_xy, class_names)` in mm
- `getBottomCenterPixel(mask)`: Mean u-coord (horizontal center) + max v-coord (bottom)
- Bottom-center ensures accurate ground plane mapping for navigation

### RobotHandler State Machine (`RobotHandler.py`)

**States:** StartScan (1) → StartGather (2) → MoveToZone (3)
**Zones:** GREEN_ZONE (0), RED_ZONE (1), GOLDEN_ZONE (2), *_OPP (3-5)

**Key Methods:**
- `handleFrame(frame, frame_id)`: Runs `segmentImage()` → `getZones()` → state logic
- `getOurZones()`: Assigns 6 zones 


```python
handler = RobotHandler(computer_receiver)
handler.handleFrame(frame, frame_id)
for x, y in handler.planned_path:
    computer_receiver.send_xy(x, y)
```

## Development Workflow

**Test Segmentation:** `cd yolo && python3 segment.py` (loads last.pt, shows masks/quads on test.jpg)

**Game Integration:**
```bash
# Pi
python3 main_pi.py --camera usb0

# Computer (main_comp.py)
handler = RobotHandler(receiver)
handler.handleFrame(frame, frame_id)  # StartScan → StartGather → MoveToZone
```

**Click-to-Move Flow:** Click (u,v) → `transform_uv_to_xy()` → `send_xy()` → rotate + forward commands → `nav.overridePaths()`

**Best Practices:**
- **ALWAYS** use `coordinates/relativeCoordinates.py` (SE2) - NEVER manual rotation math
- Game logic in `RobotHandler.py`, field params in `config.py`
- Test with `segment.py`, tune thresholds (0.25 segmentation, 0.2-0.4 custom models)
- Debug: Check console coords, `handler.planned_path`, tune `mask_utils.params`

### Common Pitfalls

**Vision:**
- False positives → Add 10-20% background images (empty labels)
- Low confidence → Custom models use 0.25 (not 0.5)
- No detections → Match resolution (640x640), verify model path
- Missing colored tape → Tune `mask_utils.params` (hue_tolerance, blur_size)
- Invalid zones → `transform_uv_to_xy()` returns None if behind camera

**Training:**
- Match image/label pairs, use relative paths in data.yaml
- Overfitting → More augmentation, reduce freeze parameter

**Connection:**
- Pi won't connect → Check network, firewall, `COMPUTER_IP` in config.py
- Auto-reconnect every 5s → Just restart computer, Pi reconnects
- Video lag → Lower JPEG_QUALITY, resolution, or FPS
- TCP latency ~10-50ms vs UDP

**Navigation:**
- No movement → `startLoop()` in thread, motors in POSITION mode
- **IMU before Raven** (critical initialization order)
- Inaccurate → Recalibrate wheel/base, check slippage
- Wrong direction → Recalibrate camera (checkerboard)
- Overshoots → Reduce velocity/accel, tune PID

## File Organization

```
MDS/
├── main_pi.py, main_comp.py       # Pi (Nav+PiStreamer), Computer (Receiver+Click)
├── raven.py, nav.py               # Motor controller, Navigation (IMU+odometry)
├── RobotHandler.py, config.py     # State machine, Field config
├── InputProcessor.py              # Click-to-move input handler
├── pyproject.toml                 # Dependencies
├── connection/                    # TCP streaming
│   ├── PiStreamer.py, ComputerReceiver.py, protocol.py, message_types.py
│   ├── CameraCapture.py, FrameSaver.py, config.py, frame_info.py
├── coordinates/relativeCoordinates.py  # **SE(2) transforms (ALWAYS use)**
├── calibration/ (distortion, manual, marking)
└── yolo/
    ├── last.pt, best.pt           # **PRIMARY segmentation models**
    ├── segment.py                 # Main segmentation script
    ├── mask_utils.py, zone_utils.py, can_utils.py, pixelTo3D.py
    ├── yolo_detect.py, yolo11n.pt, yolo11n_ncnn_model/
    └── train/ (train.py, validate.py, datasets/, runs/train/exp{N}/weights/)
```

## Model Information

**Custom Segmentation** (`yolo/last.pt`): YOLOv11n, 640x640, thresh=0.25, 8 classes
- 0: Boundary, 1: Golden Can, 2: Golden Zone, 3: Green Can, 4: Green Zone, 5: Red Can, 6: Red Zone, 7: Robot
- Outputs: `result.boxes`, `result.masks`, `result.names`

**Pretrained Detection** (`yolo11n.pt`): COCO 80 classes, 640x640, bounding boxes only
