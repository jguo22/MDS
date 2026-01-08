"""
TCP Client - Run this on the Raspberry Pi
Sends images to computer server for YOLO object detection
"""

import json
import time
import cv2
from .tcp_protocol import TCPConnection
from .yolo_detector import encode_image_to_base64

# Configuration
SERVER_HOST = '192.168.1.100'  # Replace with your computer's IP address
SERVER_PORT = 5000


class DetectionClient:
    """Client for sending images to server for object detection"""

    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self.connection = TCPConnection()

    def connect(self):
        """
        Connect to the detection server

        Returns:
            bool: True if connection successful, False otherwise
        """
        success = self.connection.connect(self.host, self.port)
        if success:
            print(f"Connected to detection server at {self.host}:{self.port}")
        else:
            print(f"Failed to connect to server at {self.host}:{self.port}")
        return success

    def is_connected(self):
        """Check if connected to server"""
        return self.connection.connected

    def detect(self, image, thresh=0.5):
        """
        Send image to server for YOLO object detection

        Args:
            image: OpenCV image (numpy array) or path to image file
            thresh: Confidence threshold for detections (default: 0.5)

        Returns:
            List of detections, or None if failed. Each detection is a dict:
            {'class': 'person', 'confidence': 0.95, 'bbox': [x1, y1, x2, y2]}
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

            # Prepare request
            request = json.dumps({
                'image': image_b64,
                'thresh': thresh
            })

            # Send to server and get response
            print(
                f"Sending image for detection ({image.shape[1]}x{image.shape[0]}, {len(image_b64)} bytes)...")
            response = self.connection.send_receive(request)

            if not response:
                print("No response from server")
                return None

            # Parse response
            data = json.loads(response)

            if data.get('status') == 'ok':
                detections = data.get('detections', [])
                image_size = data.get('image_size', [0, 0])
                print(f"Received {len(detections)} detections from server")
                return detections
            else:
                error_msg = data.get('message', 'Unknown error')
                print(f"Server error: {error_msg}")
                return None

        except json.JSONDecodeError as e:
            print(f"Failed to parse server response: {e}")
            return None
        except Exception as e:
            print(f"Error during detection: {e}")
            return None

    def disconnect(self):
        """Close the connection"""
        self.connection.close()
        print("Disconnected from server")


def main():
    """Example usage of the detection client"""
    client = DetectionClient()

    # Connect to server
    if not client.connect():
        return

    try:
        # Example 1: Detect from camera
        print("\nCapturing image from camera...")
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()

            if ret:
                print("Running detection on captured frame...")
                detections = client.detect(frame, thresh=0.5)

                if detections:
                    print(f"\nDetected {len(detections)} objects:")
                    for det in detections:
                        print(
                            f"  - {det['class']}: {det['confidence']:.2f} at {det['bbox']}")
                else:
                    print("No detections or detection failed")
            else:
                print("Failed to capture frame from camera")
        else:
            print("Camera not available")

        time.sleep(1)

        # Example 2: Detect from image file (if you have a test image)
        # detections = client.detect('test_image.jpg', thresh=0.5)
        # if detections:
        #     print(f"Detected {len(detections)} objects in test_image.jpg")

        # Example 3: Continuous detection loop
        print("\n--- Starting continuous detection (press Ctrl+C to stop) ---")
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run detection every 10 frames (~3 FPS at 30 FPS capture)
                frame_count += 1
                if frame_count % 10 == 0:
                    detections = client.detect(frame, thresh=0.5)
                    if detections:
                        print(f"Frame {frame_count}: {len(detections)} objects detected")
                        for det in detections:
                            print(f"  {det['class']}: {det['confidence']:.2f}")

                # Small delay to reduce CPU usage
                time.sleep(0.03)  # ~30 FPS

            cap.release()
        else:
            print("Could not start continuous detection - camera not available")

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        client.disconnect()


if __name__ == '__main__':
    main()
