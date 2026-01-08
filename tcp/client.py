#!/usr/bin/env python3
"""
TCP Client - Run this on the Raspberry Pi
Connects to the computer server and sends detection/sensor data
"""

import json
import time
import cv2
from tcp_protocol import TCPConnection
from yolo_detector import encode_image_to_base64

# Configuration
SERVER_HOST = '192.168.1.100'  # Replace with your computer's IP address
SERVER_PORT = 5000


class RobotClient:
    """Client for sending data to computer server and receiving commands"""

    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self.connection = TCPConnection()

    def connect(self):
        """
        Connect to the TCP server

        Returns:
            bool: True if connection successful, False otherwise
        """
        success = self.connection.connect(self.host, self.port)
        if success:
            print(f"Connected to server at {self.host}:{self.port}")
        else:
            print(f"Failed to connect to server at {self.host}:{self.port}")
        return success

    def is_connected(self):
        """Check if connected to server"""
        return self.connection.connected

    def send_data(self, msg_type, **kwargs):
        """
        Send data to the server

        Args:
            msg_type: Message type (e.g., 'detection', 'sensor', 'status')
            **kwargs: Additional data parameters

        Returns:
            Server response dict, or None if failed
        """
        data = {'type': msg_type, **kwargs}
        message = json.dumps(data)
        response = self.connection.send_receive(message)

        if response:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return response
        return None

    def ping(self):
        """
        Send a ping to test connection

        Returns:
            Server response dict
        """
        return self.send_data('ping')

    def send_detection(self, objects):
        """
        Send YOLO detection results to server

        Args:
            objects: List of detected objects with format:
                     [{'class': 'person', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]}, ...]

        Returns:
            Server response with motor commands
        """
        return self.send_data('detection', objects=objects)

    def send_sensor_data(self, encoders=None, **kwargs):
        """
        Send sensor data to server

        Args:
            encoders: List of encoder values
            **kwargs: Additional sensor data (e.g., battery, temperature)

        Returns:
            Server response dict
        """
        data = {'encoders': encoders or []}
        data.update(kwargs)
        return self.send_data('sensor', **data)

    def send_status(self, battery=None, cpu_temp=None, **kwargs):
        """
        Send robot status to server

        Args:
            battery: Battery percentage
            cpu_temp: CPU temperature
            **kwargs: Additional status information

        Returns:
            Server response dict
        """
        data = {}
        if battery is not None:
            data['battery'] = battery
        if cpu_temp is not None:
            data['cpu_temp'] = cpu_temp
        data.update(kwargs)
        return self.send_data('status', **data)

    def detect_image(self, image, thresh=0.5):
        """
        Send image to server for YOLO object detection

        Args:
            image: OpenCV image (numpy array) or path to image file
            thresh: Confidence threshold for detections (default: 0.5)

        Returns:
            Server response with detections:
            {'status': 'ok', 'detections': [...], 'image_size': [width, height]}
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    print(f"Failed to load image from {image}")
                    return None
                image = img

            # Encode image to base64
            image_b64 = encode_image_to_base64(image)

            # Send to server
            print(f"Sending image for detection ({image.shape[1]}x{image.shape[0]}, {len(image_b64)} bytes)...")
            return self.send_data('detect_image', image=image_b64, thresh=thresh)

        except Exception as e:
            print(f"Error sending image for detection: {e}")
            return None

    def disconnect(self):
        """Close the connection"""
        self.connection.close()
        print("Disconnected from server")


def main():
    """Example usage of the TCP client on Raspberry Pi"""
    client = RobotClient()

    # Connect to server
    if not client.connect():
        return

    try:
        # Example 1: Ping test
        print("\nSending ping...")
        response = client.ping()
        print(f"Response: {response}")

        time.sleep(1)

        # Example 2: Send detection results
        print("\nSending detection results...")
        detections = [
            {'class': 'person', 'confidence': 0.95, 'bbox': [100, 150, 300, 400]},
            {'class': 'bottle', 'confidence': 0.87, 'bbox': [450, 200, 550, 350]}
        ]
        response = client.send_detection(detections)
        print(f"Response: {response}")

        # Check for motor commands in response
        if isinstance(response, dict) and 'motor_commands' in response:
            print("Motor commands received:")
            for cmd in response['motor_commands']:
                print(f"  Channel {cmd['channel']}: Speed {cmd['speed']}")

        time.sleep(1)

        # Example 3: Send sensor data
        print("\nSending sensor data...")
        response = client.send_sensor_data(encoders=[1234, 5678], timestamp=time.time())
        print(f"Response: {response}")

        time.sleep(1)

        # Example 4: Send status update
        print("\nSending status update...")
        response = client.send_status(battery=85, cpu_temp=52)
        print(f"Response: {response}")

        time.sleep(1)

        # Example 5: Send image for detection (if you have a test image)
        # Option 1: From file
        # response = client.detect_image('test_image.jpg', thresh=0.5)

        # Option 2: From camera
        print("\nCapturing image from camera...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()

            if ret:
                print("Sending image for detection...")
                response = client.detect_image(frame, thresh=0.5)
                print(f"Response: {response}")

                if isinstance(response, dict) and response.get('status') == 'ok':
                    detections = response.get('detections', [])
                    print(f"\nDetected {len(detections)} objects:")
                    for det in detections:
                        print(f"  - {det['class']}: {det['confidence']:.2f} at {det['bbox']}")
            else:
                print("Failed to capture frame from camera")
        else:
            print("Camera not available - skipping image detection example")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        client.disconnect()


if __name__ == '__main__':
    main()
