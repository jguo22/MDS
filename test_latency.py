"""
Latency testing tool for Pi-Computer communication.

Usage:
    # On computer (server):
    python3 test_latency.py --mode server

    # On Pi (client):
    python3 test_latency.py --mode client --host 192.168.1.101

    # Test with large images (realistic video frame size):
    python3 test_latency.py --mode client --image-size 1000
"""

import socket
import time
import argparse
import statistics
import cv2
import numpy as np
from connection import protocol, message_types
import config
COMMAND_PORT = config.COMMAND_PORT
VIDEO_PORT = config.VIDEO_PORT


def test_latency_server(
        port: int = COMMAND_PORT,
        num_tests: int = 100,
        use_images: bool = False):
    """
    Run latency test as server (computer side).
    Receives timestamps (or frames) and sends back small acknowledgments.

    Args:
        port: Port to listen on
        num_tests: Number of ping-pong exchanges
        use_images: If True, receive large frames but send small ACKs
    """
    print(f"Starting latency test server on port {port}...")
    if use_images:
        print("Image mode: Will receive frames and send small ACKs")
    else:
        print("Command mode: Will echo back small messages")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('', port))
    server_sock.listen(1)

    print("Waiting for connection...")
    client_sock, addr = server_sock.accept()
    print(f"Connected to {addr}")

    try:
        for i in range(num_tests):
            if use_images:
                # Receive frame
                result = protocol.recv_frame(client_sock)
                if result is None or result == 0:
                    print("Connection lost")
                    break

                _, frame_id, _, _, _, _ = result

                # Send small ACK instead of echoing the frame
                # Send back just the frame_id as acknowledgment
                if not protocol.send_command(
                    client_sock, message_types.ADD_MOVEMENT, [
                        float(frame_id), 0.0, 0.0]):
                    print("Failed to send ACK")
                    break
            else:
                # Receive command
                result = protocol.recv_command(client_sock)
                if result is None:
                    print("Connection lost")
                    break

                msg_type, args = result

                # Echo back the same timestamp
                if not protocol.send_command(client_sock, msg_type, args):
                    print("Failed to send response")
                    break

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_tests} pings")

    finally:
        protocol.close_socket(client_sock)
        protocol.close_socket(server_sock)
        print("Server closed")


def test_latency_client(
        host: str,
        port: int = COMMAND_PORT,
        num_tests: int = 100,
        image_size: int = 0,
        jpeg_quality: int = config.JPEG_QUALITY):
    """
    Run latency test as client (Pi side).
    Sends timestamps and measures round-trip time.

    Args:
        host: Server IP address
        port: Server port
        num_tests: Number of ping-pong exchanges
        image_size: If > 0, send images of this size (e.g., 1000 for 1000x1000)
        jpeg_quality: JPEG quality for image compression (0-100)
    """
    print(f"Connecting to {host}:{port}...")

    use_images = image_size > 0
    frame_data = b''
    data_size_kb = 0.0

    # Generate test image if needed
    if use_images:
        print(f"Generating {image_size}x{image_size} test image...")
        # Create random color image
        test_image = np.random.randint(
            0, 256, (image_size, image_size, 3), dtype=np.uint8)

        # Encode to JPEG
        _, encoded = cv2.imencode(
            '.jpg', test_image, [
                cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        frame_data = encoded.tobytes()
        data_size_kb = len(frame_data) / 1024
        print(
            f"Test image size: {data_size_kb:.1f} KB (JPEG quality: {jpeg_quality})")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)

    try:
        sock.connect((host, port))
        print("Connected!")

        latencies = []
        data_sizes = []

        for i in range(num_tests):
            send_time = time.perf_counter()

            if use_images:
                # Send frame with timestamp as frame_id
                frame_id = i
                if not protocol.send_frame(
                        sock, frame_data, frame_id, 0.0, 0.0, 0.0, 0.0):
                    print("Failed to send frame")
                    break

                # Wait for small ACK (not full frame echo)
                result = protocol.recv_command(sock)
                recv_time = time.perf_counter()

                if result is None:
                    print("Failed to receive ACK")
                    break

                # Verify we got the right frame_id back
                _, args = result
                if len(args) > 0 and int(args[0]) != frame_id:
                    print(
                        f"Warning: Expected frame_id {frame_id}, got {int(args[0])}")

                data_sizes.append(len(frame_data))
            else:
                # Send timestamp using ADD_MOVEMENT message
                if not protocol.send_command(
                    sock, message_types.ADD_MOVEMENT, [
                        send_time, 0.0, 0.0]):
                    print("Failed to send ping")
                    break

                # Wait for echo
                result = protocol.recv_command(sock)
                recv_time = time.perf_counter()

                if result is None:
                    print("Failed to receive pong")
                    break

            # Calculate round-trip time
            rtt_ms = (recv_time - send_time) * 1000
            latencies.append(rtt_ms)

            if (i + 1) % 10 == 0:
                print(f"Ping {i + 1}/{num_tests}: {rtt_ms:.2f}ms")

            # Small delay between tests
            time.sleep(0.01)

        # Print statistics
        if latencies:
            print("\n" + "=" * 50)
            print("Latency Statistics:")
            print("=" * 50)
            if use_images:
                print(f"Test mode:       Image ({image_size}x{image_size})")
                print(f"Data size:       {data_size_kb:.1f} KB per frame")
                print(f"JPEG quality:    {jpeg_quality}")
                print(f"Note:            Server sends small ACK, not full frame echo")
            else:
                print(f"Test mode:       Command messages (full echo)")
            print(f"Tests completed: {len(latencies)}/{num_tests}")
            print(f"Min:             {min(latencies):.2f}ms")
            print(f"Max:             {max(latencies):.2f}ms")
            print(f"Mean:            {statistics.mean(latencies):.2f}ms")
            print(f"Median:          {statistics.median(latencies):.2f}ms")
            if len(latencies) > 1:
                print(f"Std Dev:         {statistics.stdev(latencies):.2f}ms")

            if use_images:
                # For images: measure is send time + small ACK time (mostly
                # one-way)
                print(f"Interpretation:  ~One-way send latency + small ACK")
                # Calculate throughput based on actual time
                avg_latency_s = statistics.mean(latencies) / 1000
                throughput_mbps = (data_size_kb * 8 / 1024) / avg_latency_s
                print(f"Throughput:      {throughput_mbps:.2f} Mbps")
            else:
                # For commands: full round-trip echo
                print(
                    f"One-way (est):   {statistics.mean(latencies) / 2:.2f}ms")

            print("=" * 50)

    finally:
        protocol.close_socket(sock)
        print("Client closed")


def main():
    parser = argparse.ArgumentParser(
        description='Test latency between Pi and computer')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                        help='Run as server (computer) or client (Pi)')
    parser.add_argument('--host', type=str, default=config.COMPUTER_IP,
                        help='Server IP address (client mode only)')
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help=f'Port to use (default: COMMAND_PORT for small messages, VIDEO_PORT for images)')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of ping-pong tests (default: 100)')
    parser.add_argument(
        '--image-size',
        type=int,
        default=0,
        help='Image size for testing (e.g., 1000 for 1000x1000). 0 = use small messages (default)')
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=config.JPEG_QUALITY,
        help='JPEG quality for image compression (0-100, default: 80)')

    args = parser.parse_args()

    # Auto-select port based on test mode
    if args.port is None:
        if args.image_size > 0:
            args.port = VIDEO_PORT
            print(f"Image mode: Using VIDEO_PORT ({VIDEO_PORT})")
        else:
            args.port = COMMAND_PORT
            print(f"Command mode: Using COMMAND_PORT ({COMMAND_PORT})")

    use_images = args.image_size > 0

    try:
        if args.mode == 'server':
            test_latency_server(args.port, args.num_tests, use_images)
        else:
            test_latency_client(
                args.host,
                args.port,
                args.num_tests,
                args.image_size,
                args.jpeg_quality)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
