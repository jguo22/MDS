"""
TCP Server - Run this on your computer
Receives images from Raspberry Pi and returns YOLO detection results
"""

import argparse
import socket
import threading
import json
from tcp_protocol import TCPProtocol
from yolo_detector import YOLODetector

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000       # Port to listen on


class DetectionServer:
    """Server for running YOLO detection on images from Raspberry Pi"""

    def __init__(self, host=HOST, port=PORT, model_path='yolo/yolo11n.pt'):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.detector = YOLODetector(model_path)

    def load_model(self):
        """Load YOLO model for object detection"""
        return self.detector.load()

    def start(self):
        """Start the TCP server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        print(f"Detection server listening on {self.host}:{self.port}")

        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                print(f"Connection from {address}")

                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()

            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")
                break

    def handle_client(self, client_socket, address):
        """Handle communication with a connected client"""
        try:
            while self.running:
                # Receive message using TCP protocol
                message = TCPProtocol.recv_message(client_socket)
                if not message:
                    break

                print(
                    f"Received message from {address} ({len(message)} bytes)")

                # Process the message
                response = self.process_message(message)

                # Send response back to client
                TCPProtocol.send_message(client_socket, response)

        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection closed: {address}")

    def process_message(self, message):
        """Process incoming detection request"""
        try:
            # Parse JSON message
            data = json.loads(message)

            # Extract image and threshold
            image_b64 = data.get('image')
            thresh = data.get('thresh', 0.5)

            if not image_b64:
                return json.dumps(
                    {'status': 'error', 'message': 'No image provided'})

            if not self.detector.is_loaded():
                return json.dumps(
                    {'status': 'error', 'message': 'YOLO model not loaded'})

            # Run detection on base64 image
            detections, image_size = self.detector.detect_from_base64(
                image_b64, thresh)

            print(
                f"Running YOLO detection on image ({image_size[0]}x{image_size[1]})...")
            print(f"Detected {len(detections)} objects")

            return json.dumps({
                'status': 'ok',
                'detections': detections,
                'image_size': image_size
            })

        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            return json.dumps(
                {'status': 'error', 'message': 'Invalid JSON format'})
        except ValueError as e:
            print(f"Error during detection: {e}")
            return json.dumps({'status': 'error', 'message': str(e)})
        except Exception as e:
            print(f"Unexpected error: {e}")
            return json.dumps({'status': 'error',
                               'message': f'Detection failed: {str(e)}'})

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='Path to YOLO model file',
        required=True)

    args = parser.parse_args()

    model_path = args.model

    server = DetectionServer(model_path=model_path)

    if not server.load_model():
        print("Failed to load YOLO model. Exiting.")
        return

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()


if __name__ == '__main__':
    main()
