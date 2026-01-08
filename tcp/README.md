# TCP Communication System

This directory contains a TCP-based client-server communication system for the MASLAB 2026 robot. The system enables reliable communication between a Raspberry Pi (robot) and a computer (control station) for remote object detection, sensor data transmission, and motor command coordination.

## Architecture Overview

```
┌─────────────────────┐                    ┌─────────────────────┐
│  Raspberry Pi       │                    │  Computer           │
│  (Robot)            │◄──────TCP─────────►│  (Control Station)  │
│                     │                    │                     │
│  - Camera capture   │                    │  - YOLO detection   │
│  - Sensor reading   │                    │  - Path planning    │
│  - Motor control    │   Image + Data     │  - Decision making  │
│                     │   ────────────►    │                     │
│                     │   ◄────────────    │                     │
│                     │  Motor Commands    │                     │
└─────────────────────┘                    └─────────────────────┘
```

## File Structure

```
tcp/
├── README.md              # This file
├── tcp_protocol.py        # Length-prefixed TCP protocol implementation
├── yolo_detector.py       # YOLO model wrapper for object detection
├── server.py              # Server (runs on computer)
└── client.py              # Client (runs on Raspberry Pi)
```

## Modules

### `tcp_protocol.py`

Implements a length-prefixed TCP protocol for reliable message transmission. This module solves the problem of sending variable-length messages (especially large images) over TCP.

**Key Classes:**
- `TCPProtocol`: Static methods for sending/receiving length-prefixed messages
- `TCPConnection`: Wrapper for socket connections with protocol support

**Protocol Format:**
```
[4 bytes: message length][N bytes: UTF-8 encoded message]
```

**Usage:**
```python
from tcp_protocol import TCPProtocol, TCPConnection

# Low-level usage
TCPProtocol.send_message(socket, "Hello")
message = TCPProtocol.recv_message(socket)

# High-level usage
conn = TCPConnection()
conn.connect("192.168.1.100", 5000)
response = conn.send_receive("Hello")
conn.close()
```

### `yolo_detector.py`

Wrapper for YOLO object detection with support for OpenCV images and base64-encoded images.

**Key Classes:**
- `YOLODetector`: YOLO model loader and detection runner

**Key Functions:**
- `encode_image_to_base64(image)`: Encode OpenCV image to base64 string

**Usage:**
```python
from yolo_detector import YOLODetector

# Initialize and load model
detector = YOLODetector('yolo/yolo11n.pt')
detector.load()

# Detect from file or OpenCV image
detections = detector.detect('image.jpg', thresh=0.5)

# Detect from base64 image
detections, image_size = detector.detect_from_base64(image_b64, thresh=0.5)

# Each detection is a dict:
# {'class': 'person', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]}
```

### `server.py`

TCP server that runs on your computer. Receives data from the Raspberry Pi and performs heavy computation (YOLO detection, path planning, etc.).

**Key Class:**
- `RobotServer`: Multi-threaded server handling multiple message types

**Message Handlers:**
- `handle_ping()`: Connection test
- `handle_detection()`: Process detection results from Pi
- `handle_sensor()`: Process sensor data from Pi
- `handle_status()`: Process status updates from Pi
- `handle_detect_image()`: Run YOLO detection on received image

**Usage:**
```bash
# Run with default model
python3 server.py

# Run with custom model
python3 server.py yolo/train/runs/train/exp2/weights/best.pt
```

### `client.py`

TCP client that runs on the Raspberry Pi. Sends camera images, sensor data, and status updates to the computer server.

**Key Class:**
- `RobotClient`: Client for communicating with server

**Key Methods:**
- `ping()`: Test connection
- `send_detection(objects)`: Send pre-computed detections
- `send_sensor_data(encoders, **kwargs)`: Send sensor readings
- `send_status(battery, cpu_temp, **kwargs)`: Send status updates
- `detect_image(image, thresh)`: Send image for remote YOLO detection

**Usage:**
```bash
# Edit SERVER_HOST in client.py first, then run
python3 client.py
```

## Message Protocol

All messages are JSON-formatted with a `type` field indicating the message type.

### Message Types

#### 1. Ping (Connection Test)

**Request:**
```json
{"type": "ping"}
```

**Response:**
```json
{"status": "ok", "message": "pong"}
```

#### 2. Detection Results (Pi → Server)

**Request:**
```json
{
  "type": "detection",
  "objects": [
    {"class": "person", "confidence": 0.95, "bbox": [100, 150, 300, 400]},
    {"class": "bottle", "confidence": 0.87, "bbox": [450, 200, 550, 350]}
  ]
}
```

**Response:**
```json
{
  "status": "ok",
  "motor_commands": [
    {"channel": 1, "speed": 50},
    {"channel": 2, "speed": -30}
  ]
}
```

#### 3. Sensor Data (Pi → Server)

**Request:**
```json
{
  "type": "sensor",
  "encoders": [1234, 5678],
  "timestamp": 1234567890.123
}
```

**Response:**
```json
{"status": "ok", "message": "Sensor data received"}
```

#### 4. Status Update (Pi → Server)

**Request:**
```json
{
  "type": "status",
  "battery": 85,
  "cpu_temp": 52
}
```

**Response:**
```json
{"status": "ok"}
```

#### 5. Image Detection (Pi → Server)

**Request:**
```json
{
  "type": "detect_image",
  "image": "<base64-encoded JPEG>",
  "thresh": 0.5
}
```

**Response:**
```json
{
  "status": "ok",
  "detections": [
    {"class": "person", "confidence": 0.95, "bbox": [100, 150, 300, 400]}
  ],
  "image_size": [640, 480]
}
```

**Error Response:**
```json
{"status": "error", "message": "Error description"}
```

## Setup and Configuration

### 1. Find Your Computer's IP Address

On your computer (Mac/Linux):
```bash
ifconfig
# Look for inet address on active interface (e.g., en0 for WiFi)
```

On Windows:
```bash
ipconfig
# Look for IPv4 Address
```

### 2. Configure Client

Edit `client.py` and update the server IP:
```python
SERVER_HOST = '192.168.1.100'  # Replace with your computer's IP
```

### 3. Install Dependencies

Both computer and Raspberry Pi need:
```bash
pip3 install opencv-python numpy ultralytics
```

### 4. Run the System

**On Computer (Server):**
```bash
cd tcp/
python3 server.py
```

**On Raspberry Pi (Client):**
```bash
cd tcp/
python3 client.py
```

## Example Integration

### Raspberry Pi - Continuous Detection Loop

```python
from client import RobotClient
import cv2
import time

client = RobotClient(host='192.168.1.100', port=5000)
client.connect()

cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Send to server for detection
        response = client.detect_image(frame, thresh=0.5)

        if response and response.get('status') == 'ok':
            detections = response.get('detections', [])
            print(f"Detected {len(detections)} objects")

            # Process detections (e.g., track target object)
            for det in detections:
                if det['class'] == 'bottle':
                    bbox = det['bbox']
                    # TODO: Calculate motor commands to approach bottle
                    print(f"Found bottle at {bbox}")

        time.sleep(0.1)  # 10 FPS

except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    client.disconnect()
```

### Computer - Custom Detection Handler

```python
from server import RobotServer

class CustomServer(RobotServer):
    def handle_detect_image(self, data):
        # Run parent detection
        response_json = super().handle_detect_image(data)
        response = json.loads(response_json)

        # Add custom logic
        if response.get('status') == 'ok':
            detections = response.get('detections', [])

            # Example: Calculate motor commands based on detection positions
            if detections:
                # Find center of first detection
                bbox = detections[0]['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                image_width = response['image_size'][0]

                # Simple centering logic
                if center_x < image_width / 3:
                    response['motor_commands'] = [
                        {'channel': 1, 'speed': 30},  # Turn left
                        {'channel': 2, 'speed': 50}
                    ]
                elif center_x > 2 * image_width / 3:
                    response['motor_commands'] = [
                        {'channel': 1, 'speed': 50},  # Turn right
                        {'channel': 2, 'speed': 30}
                    ]
                else:
                    response['motor_commands'] = [
                        {'channel': 1, 'speed': 50},  # Go straight
                        {'channel': 2, 'speed': 50}
                    ]

        return json.dumps(response)

# Run custom server
server = CustomServer(model_path='yolo/yolo11n.pt')
server.load_model()
server.start()
```

## Performance Considerations

### Image Transmission

- Images are JPEG-compressed before base64 encoding (reduces size ~10x)
- 640x480 image ≈ 50-100 KB after JPEG compression
- Base64 encoding increases size by ~33% (final size ≈ 70-130 KB)
- Transmission time over WiFi: ~50-200ms depending on signal strength

### YOLO Detection

- YOLOv11n inference: ~30-100ms per frame on typical computer
- Transfer learning model: similar performance
- Consider running detection at lower frame rate (5-10 FPS) to allow time for motor control

### Network Latency

- Typical round-trip time: 100-300ms
  - Network transmission: 50-100ms
  - YOLO detection: 30-100ms
  - Protocol overhead: 10-50ms

## Troubleshooting

### Connection Refused

- Check firewall settings on computer (allow port 5000)
- Verify computer and Pi are on same network
- Confirm SERVER_HOST is correct in client.py

### Image Detection Fails

- Ensure YOLO model path is correct on server
- Check model loaded successfully (look for "YOLO model loaded successfully" message)
- Verify image encoding is successful on client side

### Slow Performance

- Reduce image resolution before sending
- Lower YOLO confidence threshold to reduce processing time
- Use NCNN model on Pi for local detection instead of remote

### Connection Drops

- Check WiFi signal strength
- Add reconnection logic to client
- Implement heartbeat/ping mechanism for connection monitoring

## Future Enhancements

- [ ] Add authentication/encryption for secure communication
- [ ] Implement connection pooling for multiple clients
- [ ] Add compression for image transmission (e.g., reduce JPEG quality)
- [ ] Support video streaming with frame buffering
- [ ] Add timeout handling and automatic reconnection
- [ ] Implement command queuing for motor control
- [ ] Add telemetry logging and visualization

## License

This code is part of the MASLAB 2026 project.
