# TCP Object Detection System

Remote YOLO object detection system for the MASLAB 2026 robot. Sends images from Raspberry Pi to a computer for processing and receives detection results.

## Architecture

```
┌─────────────────────┐                    ┌─────────────────────┐
│  Raspberry Pi       │                    │  Computer           │
│  (Robot)            │◄──────TCP─────────►│  (Detection Server) │
│                     │                    │                     │
│  - Camera capture   │                    │  - YOLO detection   │
│  - Motor control    │   Base64 Image     │  - Model inference  │
│                     │   ────────────►    │                     │
│                     │   ◄────────────    │                     │
│                     │    Detections      │                     │
└─────────────────────┘                    └─────────────────────┘
```

## File Structure

```
tcp/
├── README.md              # This file
├── tcp_protocol.py        # Length-prefixed TCP protocol
├── yolo_detector.py       # YOLO model wrapper
├── server.py              # Detection server (runs on computer)
└── client.py              # Detection client (runs on Raspberry Pi)
```

## Quick Start

### 1. Setup

**On Computer:**
```bash
cd tcp/
pip3 install opencv-python numpy ultralytics
```

**On Raspberry Pi:**
```bash
cd tcp/
pip3 install opencv-python numpy
```

### 2. Find Your Computer's IP

```bash
# Mac/Linux
ifconfig | grep inet

# Windows
ipconfig
```

### 3. Configure Client

Edit `client.py` and update:
```python
SERVER_HOST = '192.168.1.100'  # Your computer's IP
```

### 4. Run

**On Computer (start server first):**
```bash
python3 server.py
```

**On Raspberry Pi:**
```bash
python3 client.py
```

## Usage

### Server (Computer)

```bash
# Use default model (yolo/yolo11n.pt)
python3 server.py

# Use custom trained model
python3 server.py yolo/train/runs/train/exp2/weights/best.pt
```

The server will:
1. Load the YOLO model
2. Listen on port 5000
3. Process incoming images
4. Return detection results

### Client (Raspberry Pi)

**Basic Usage:**
```python
from client import DetectionClient
import cv2

# Connect to server
client = DetectionClient(host='192.168.1.100', port=5000)
client.connect()

# Capture and detect
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    detections = client.detect(frame, thresh=0.5)

    for det in detections:
        print(f"{det['class']}: {det['confidence']:.2f} at {det['bbox']}")

client.disconnect()
```

**Continuous Detection Loop:**
```python
from client import DetectionClient
import cv2
import time

client = DetectionClient(host='192.168.1.100', port=5000)
client.connect()

cap = cv2.VideoCapture(0)
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect every 10 frames (~3 FPS detection)
        frame_count += 1
        if frame_count % 10 == 0:
            detections = client.detect(frame, thresh=0.5)

            if detections:
                print(f"Found {len(detections)} objects")
                # TODO: Use detections for motor control

        time.sleep(0.03)  # ~30 FPS capture

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    client.disconnect()
```

**Detect from File:**
```python
from client import DetectionClient

client = DetectionClient(host='192.168.1.100', port=5000)
client.connect()

# Detect from image file
detections = client.detect('photo.jpg', thresh=0.5)

if detections:
    for det in detections:
        print(f"{det['class']}: {det['confidence']:.2f}")

client.disconnect()
```

## API Reference

### DetectionClient

**Methods:**

- `connect()` - Connect to server. Returns `True` if successful.
- `is_connected()` - Check connection status.
- `detect(image, thresh=0.5)` - Send image for detection. Returns list of detections or `None`.
- `disconnect()` - Close connection.

**Detection Format:**

Each detection is a dictionary:
```python
{
    'class': 'person',           # Object class name
    'confidence': 0.95,          # Confidence score (0-1)
    'bbox': [x1, y1, x2, y2]    # Bounding box coordinates
}
```

### DetectionServer

**Methods:**

- `load_model()` - Load YOLO model. Returns `True` if successful.
- `start()` - Start the server (blocking).
- `stop()` - Stop the server.

## Protocol

### Request Format

```json
{
    "image": "<base64-encoded JPEG>",
    "thresh": 0.5
}
```

### Response Format

**Success:**
```json
{
    "status": "ok",
    "detections": [
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": [100, 150, 300, 400]
        }
    ],
    "image_size": [640, 480]
}
```

**Error:**
```json
{
    "status": "error",
    "message": "Error description"
}
```

## Modules

### tcp_protocol.py

Implements length-prefixed TCP protocol for reliable transmission of large messages.

**Key Classes:**
- `TCPProtocol` - Low-level send/receive with 4-byte length prefix
- `TCPConnection` - High-level connection wrapper

### yolo_detector.py

YOLO model wrapper for object detection.

**Key Functions:**
- `YOLODetector.load()` - Load model
- `YOLODetector.detect(image, thresh)` - Run detection
- `YOLODetector.detect_from_base64(image_b64, thresh)` - Detect from base64 image
- `encode_image_to_base64(image)` - Encode OpenCV image to base64

## Performance

### Typical Latency

- Image encoding (Pi): ~10-20ms
- Network transmission: ~50-100ms
- YOLO detection (Computer): ~30-100ms
- **Total round-trip: 100-300ms**

### Throughput

- Recommended detection rate: **3-10 FPS**
- Image size after compression: ~50-100 KB (640x480 JPEG)
- Network bandwidth needed: ~0.5-1 MB/s at 10 FPS

### Optimization Tips

1. **Lower detection frequency** - Detect every N frames instead of every frame
2. **Reduce image resolution** - Resize to 320x240 or 416x416 before sending
3. **Adjust JPEG quality** - Lower quality = smaller files (edit `yolo_detector.py`)
4. **Increase threshold** - Higher confidence threshold = faster processing

## Troubleshooting

### Connection Refused

- Check that server is running on computer
- Verify firewall allows port 5000
- Confirm computer and Pi are on same network
- Verify SERVER_HOST is correct in `client.py`

### "YOLO model not loaded"

- Check model path is correct
- Verify model file exists
- Ensure Ultralytics is installed: `pip3 install ultralytics`

### Slow Performance

- Reduce camera resolution
- Increase detection interval (detect every N frames)
- Use smaller YOLO model (yolo11n vs yolo11s/m/l)
- Check WiFi signal strength

### Camera Not Found

- Verify camera is connected: `ls /dev/video*`
- Try different camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
- For PiCamera: Use `libcamera` or picamera2 library

### No Detections

- Lower confidence threshold: `detect(frame, thresh=0.2)`
- Check that objects are in frame
- Verify lighting conditions
- Test with images that have known objects

## Example: Robot Navigation

```python
from client import DetectionClient
import cv2
import time

# Import your motor control library
# from raven import Raven, MotorChannel, MotorMode

client = DetectionClient(host='192.168.1.100', port=5000)
client.connect()

# Initialize motor controller
# raven = Raven()
# raven.set_motor_mode(MotorChannel.CH1, MotorMode.DIRECT)
# raven.set_motor_mode(MotorChannel.CH2, MotorMode.DIRECT)

cap = cv2.VideoCapture(0)
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        # Detect every 5 frames
        if frame_count % 5 == 0:
            detections = client.detect(frame, thresh=0.5)

            if detections:
                # Find target object (e.g., bottle)
                target = None
                for det in detections:
                    if det['class'] == 'bottle':
                        target = det
                        break

                if target:
                    # Get bounding box
                    x1, y1, x2, y2 = target['bbox']
                    center_x = (x1 + x2) / 2
                    image_width = frame.shape[1]

                    # Simple centering logic
                    error = (center_x - image_width / 2) / image_width

                    if abs(error) < 0.1:
                        # Centered - move forward
                        print("Moving forward")
                        # raven.set_motor_speed_factor(MotorChannel.CH1, 50)
                        # raven.set_motor_speed_factor(MotorChannel.CH2, 50)
                    elif error < 0:
                        # Target on left - turn left
                        print("Turning left")
                        # raven.set_motor_speed_factor(MotorChannel.CH1, 30)
                        # raven.set_motor_speed_factor(MotorChannel.CH2, 50)
                    else:
                        # Target on right - turn right
                        print("Turning right")
                        # raven.set_motor_speed_factor(MotorChannel.CH1, 50)
                        # raven.set_motor_speed_factor(MotorChannel.CH2, 30)
                else:
                    # No target - stop
                    print("No target found - stopping")
                    # raven.set_motor_speed_factor(MotorChannel.CH1, 0)
                    # raven.set_motor_speed_factor(MotorChannel.CH2, 0)

        time.sleep(0.03)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    client.disconnect()
    # raven.set_motor_speed_factor(MotorChannel.CH1, 0)
    # raven.set_motor_speed_factor(MotorChannel.CH2, 0)
```

## License

Part of MASLAB 2026 project.
