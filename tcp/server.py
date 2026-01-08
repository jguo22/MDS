#!/usr/bin/env python3
"""
TCP Server - Run this on your computer
Listens for incoming connections from Raspberry Pi and sends motor commands
"""

import socket
import threading
import json
from tcp_protocol import TCPProtocol
from yolo_detector import YOLODetector

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000       # Port to listen on


class RobotServer:
    """Server for receiving data from Raspberry Pi and sending commands"""

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
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True

        print(f"Server listening on {self.host}:{self.port}")

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

                print(f"Received message from {address} ({len(message)} bytes)")

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
        """
        Process incoming message from Raspberry Pi and return response
        Override this method to implement custom message handling
        """
        try:
            # Try to parse as JSON
            data = json.loads(message)
            msg_type = data.get('type', '')

            # Route to appropriate handler
            if msg_type == 'ping':
                return self.handle_ping(data)
            elif msg_type == 'detection':
                return self.handle_detection(data)
            elif msg_type == 'sensor':
                return self.handle_sensor(data)
            elif msg_type == 'status':
                return self.handle_status(data)
            elif msg_type == 'detect_image':
                return self.handle_detect_image(data)
            else:
                return json.dumps({'status': 'error', 'message': 'Unknown message type'})

        except json.JSONDecodeError:
            # Handle plain text messages
            if message.lower() == 'ping':
                return 'pong'
            else:
                return f'Echo: {message}'

    def handle_ping(self, data):
        """Handle ping request"""
        return json.dumps({'status': 'ok', 'message': 'pong'})

    def handle_detection(self, data):
        """
        Handle detection results sent from Pi
        Process detections and return motor commands
        """
        objects = data.get('objects', [])
        print(f"Detected {len(objects)} objects: {objects}")

        # TODO: Process detection results and compute motor commands
        # Example response with motor commands
        return json.dumps({
            'status': 'ok',
            'motor_commands': [
                {'channel': 1, 'speed': 50},
                {'channel': 2, 'speed': -30}
            ]
        })

    def handle_sensor(self, data):
        """Handle sensor data from Pi"""
        encoders = data.get('encoders', [])
        print(f"Sensor data - Encoders: {encoders}")

        # TODO: Process sensor data
        return json.dumps({'status': 'ok', 'message': 'Sensor data received'})

    def handle_status(self, data):
        """Handle status updates from Pi"""
        battery = data.get('battery')
        cpu_temp = data.get('cpu_temp')
        print(f"Status - Battery: {battery}%, CPU Temp: {cpu_temp}°C")
        return json.dumps({'status': 'ok'})

    def handle_detect_image(self, data):
        """
        Handle image detection request
        Receive base64 image, run YOLO detection, return results
        """
        image_b64 = data.get('image')
        thresh = data.get('thresh', 0.5)

        if not image_b64:
            return json.dumps({'status': 'error', 'message': 'No image provided'})

        if not self.detector.is_loaded():
            return json.dumps({'status': 'error', 'message': 'YOLO model not loaded'})

        try:
            # Run detection on base64 image
            detections, image_size = self.detector.detect_from_base64(image_b64, thresh)

            print(f"Running YOLO detection on image ({image_size[0]}x{image_size[1]})...")
            print(f"Detected {len(detections)} objects")

            return json.dumps({
                'status': 'ok',
                'detections': detections,
                'image_size': image_size
            })

        except ValueError as e:
            print(f"Error during detection: {e}")
            return json.dumps({'status': 'error', 'message': str(e)})
        except Exception as e:
            print(f"Unexpected error during detection: {e}")
            return json.dumps({'status': 'error', 'message': f'Detection failed: {str(e)}'})

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped")


def main():
    import sys

    # Optional: specify model path as command line argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'yolo/yolo11n.pt'

    server = RobotServer(model_path=model_path)
    server.load_model()

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()


if __name__ == '__main__':
    main()
