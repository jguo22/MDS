# MASLAB 2026 Autonomous Robot

Autonomous can-collecting robot for the MIT MASLAB 2026 competition. Features dual-camera vision, YOLOv11 segmentation, path planning with Theta*, and a distributed architecture with Pi-side execution and computer-side control.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Computer                                    │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │ ComputerReceiver│◄───│  RobotHandler    │◄───│ Vision (YOLO)  │  │
│  │ (video in,      │    │  State Machine   │    │ Segmentation   │  │
│  │  commands out)  │    │  + Theta* Path   │    │ + Pixel-to-3D  │  │
│  └────────┬────────┘    └──────────────────┘    └────────────────┘  │
│           │                                                          │
│           │ TCP (video: 5000, commands: 5001)                        │
│           ▼                                                          │
│  ┌─────────────────┐                                                 │
│  │RemoteRobotCmd   │ ◄── IRobotCommander interface                   │
│  └─────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Network
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Raspberry Pi                                  │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  PiStreamer     │───►│ DirectRobotCmd   │───►│ Nav + Raven    │  │
│  │  (video out,    │    │ (local execution)│    │ Motor Control  │  │
│  │   commands in)  │    └──────────────────┘    └────────────────┘  │
│  └─────────────────┘                                                 │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐    ┌──────────────────┐                        │
│  │  Dual Cameras   │    │  IMU + Distance  │                        │
│  │  (top + bottom) │    │  Sensors         │                        │
│  └─────────────────┘    └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Computer Side
```bash
# Install dependencies
pip install -e .

# Run computer receiver (waits for Pi connection)
python main_comp.py
```

### Pi Side
```bash
# Install dependencies
pip install -e .

# Run Pi streamer (auto-reconnects to computer)
python main_pi.py --camera-top /dev/videoblacktop --camera-bottom /dev/videoblackbot
```

## Project Structure

```
MDS/
├── main_comp.py              # Computer entry point (receiver + vision + state machine)
├── main_pi.py                # Pi entry point (streamer + command execution)
│
├── IRobotCommander.py        # Abstract interface for robot commands
├── RemoteRobotCommander.py   # Network implementation (computer side)
├── DirectRobotCommander.py   # Direct execution (Pi side)
│
├── RobotHandler.py           # Full state machine with zone/can tracking
├── RobotHandler_Simple.py    # Simplified state machine for basic collection
│
├── thetaStar.py              # Theta* path planning with obstacle avoidance
├── nav.py                    # Navigation controller (IMU + odometry)
├── navHelpers.py             # Movement command generators
│
├── raven.py                  # Raven motor controller driver
├── RavenWrapper.py           # High-level Raven interface (gripper, elevator)
├── IMUWrapper.py             # BNO08x IMU wrapper
├── distanceSensorWrapper.py  # Distance sensor for can approach
│
├── config.py                 # Robot configuration constants
├── colors.py                 # YOLO class names and color constants
├── getPickedCanColor.py      # Detect color of can in gripper
│
├── connection/               # Network communication
│   ├── ComputerReceiver.py   # Computer-side video receiver + command sender
│   ├── PiStreamer.py         # Pi-side video streamer + command receiver
│   ├── RemoteRobotCommander.py # Network command implementation
│   ├── protocol.py           # TCP message framing
│   ├── message_types.py      # Command type constants
│   ├── CameraCapture.py      # Camera interface
│   ├── FrameSaver.py         # Auto-save frames for debugging
│   └── frame_info.py         # Frame metadata structure
│
└── vision/                   # Vision processing (refactored)
    ├── segment.py            # YOLO segmentation entry point
    ├── mask_utils.py         # Mask processing (convex hull, HSV refinement)
    ├── can_utils.py          # Can detection + pixel-to-world transform
    ├── zone_utils.py         # Zone detection + quad extraction
    ├── pixelTo3D.py          # Camera calibration + ground plane projection
    ├── relativeCoordinates.py # SE2 coordinate transformations
    ├── calibration/          # Camera calibration tools
    ├── train/                # YOLO training scripts and datasets
    ├── best.pt               # Trained YOLO model
    └── yolo11n.pt            # Pretrained YOLO model
```

## Vision System

### YOLO Segmentation

Custom YOLOv11n model trained on 8 classes:
- `Boundary` (0), `Golden Can` (1), `Golden Zone` (2)
- `Green Can` (3), `Green Zone` (4), `Red Can` (5), `Red Zone` (6), `Robot` (7)

```python
from vision.segment import segmentImage
from vision.can_utils import getCans
from vision.zone_utils import getZones

result = segmentImage(image)  # Run YOLO segmentation
cans, can_classes = getCans(result, image, is_top=True)  # Get can positions (mm)
zones, zone_classes = getZones(result, image, is_top=True)  # Get zone quads (mm)
```

### Pixel-to-World Transformation

Transforms pixel coordinates to ground plane (mm) using camera calibration:

```python
from vision.pixelTo3D import transform_uv_to_xy

# pixel (u, v) -> world (x, y) in mm
x, y = transform_uv_to_xy(pixel_x, pixel_y, is_top=True)
```

### Coordinate Transformations

SE2-based world/relative coordinate conversion:

```python
from spatialmath import SE2
from vision.relativeCoordinates import world_to_relative, relative_to_world

robot_pose = SE2(robot_x, robot_y, robot_heading)
local_x, local_y = world_to_relative(world_point, robot_pose)
world_x, world_y = relative_to_world(local_point, robot_pose)
```

### Gripper Can Detection

Detect which color can is currently in the gripper:

```python
from getPickedCanColor import getPickedUpCanColor

color = getPickedUpCanColor(result, image)  # Returns 'Green Can', 'Red Can', etc.
```

## Robot Commander Interface

All robot commands go through `IRobotCommander`, with two implementations:

| Method | Description |
|--------|-------------|
| `override_movement(args)` | Raw motor commands `[left, right, ticks, ...]` |
| `override_waypoints(args)` | Navigate through waypoints `[x1, y1, x2, y2, ...]` |
| `override_relative_xy(x, y)` | Move relative to robot (mm) |
| `override_world_xy(x, y)` | Navigate to world coordinates (mm) |
| `pickup_can()` | Close gripper + raise elevator |
| `set_down_can()` | Lower elevator + open gripper |
| `open_gripper()` | Open gripper only |
| `lower_elevator()` | Lower elevator only |
| `approach_can_with_ds()` | Approach can using distance sensor |
| `backup()` | Back up short distance |
| `stack(temp, stack, count)` | Stack can at position |
| `waitFinishedMoving()` | Block until movement complete |

## State Machines

### RobotHandler (Full)
Complex state machine with:
- Zone confidence tracking
- Can detection grid with thresholds
- Opponent zone avoidance
- Full early game strategy

### RobotHandler_Simple (Simplified)
Basic can collection:
```
SearchForCan -> MoveToCan -> ApproachingCan -> GrabCan -> MoveToZone -> Done
```

Configuration:
- `MAX_STACK`: Number of cans to collect before going to zone
- `target_zone`: Which zone to deliver to (GREEN_ZONE, RED_ZONE, GOLDEN_ZONE)
- `target_can_color`: Which color cans to collect

## Hardware

### Raven Motor Controller
Custom serial motor controller (460800 baud):
- 5 motor channels, 4 servo channels
- Position/velocity PID control
- Built-in odometry

### IMU (BNO08x)
- Must initialize before Raven board
- Heading relative to startup position

### Distance Sensor
- Used for precise can approach
- Target distance: 25mm, Max detection: 500mm

## Network Protocol

TCP-based with custom framing:
- Video port: 5000 (JPEG frames)
- Command port: 5001 (typed messages with float arrays)

Message types defined in `connection/message_types.py`:
```python
OVERRIDE_MOVEMENTS = 2      # [left, right, ticks, ...]
OVERRIDE_WAYPOINTS = 3      # [x1, y1, x2, y2, ...]
PICKUP_CAN = 4              # No args
RELEASE_CAN = 5             # No args
APPROACH_CAN_DS = 9         # No args
OPEN_GRIPPER = 15           # No args
LOWER_ELEVATOR = 16         # No args
# ... and more
```

## Configuration

Key constants in `config.py`:
```python
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

CAN_DIAMETER = 76.2         # mm
ROBOT_DIAMETER = 300        # mm
CLAW_OFFSET = 150           # mm from robot center
APPROACH_OFFSET = 200       # mm approach distance
```

## Development

### Testing Vision
```bash
# Test segmentation on image
python vision/segment.py

# Test can detection
python getPickedCanColor.py
```

### Training YOLO
```bash
cd vision/train
python train.py
```

### Debugging
- Frames auto-saved to `images/` with cooldown
- Telemetry streamed via `Streamer` class
- Profile data saved to `profiles/`

## Dependencies

See `pyproject.toml`:
- ultralytics (YOLO)
- opencv-python
- numpy
- spatialmath-python
- shapely
- adafruit-circuitpython-bno08x (Pi only)
- pyserial (Pi only)
