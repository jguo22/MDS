"""
Latency testing tool for Pi-Computer communication.

Usage:
    # On computer (server):
    python3 test_latency.py --mode server

    # On Pi (client):
    python3 test_latency.py --mode client --host 192.168.1.101
"""

import socket
import time
import argparse
import statistics
from connection import protocol, message_types
import config
COMMAND_PORT = config.COMMAND_PORT


def test_latency_server(port: int = COMMAND_PORT, num_tests: int = 100):
    """
    Run latency test as server (computer side).
    Receives timestamps and echoes them back.

    Args:
        port: Port to listen on
        num_tests: Number of ping-pong exchanges
    """
    print(f"Starting latency test server on port {port}...")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(('', port))
    server_sock.listen(1)

    print("Waiting for connection...")
    client_sock, addr = server_sock.accept()
    print(f"Connected to {addr}")

    try:
        for i in range(num_tests):
            # Receive timestamp from client
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
        num_tests: int = 100):
    """
    Run latency test as client (Pi side).
    Sends timestamps and measures round-trip time.

    Args:
        host: Server IP address
        port: Server port
        num_tests: Number of ping-pong exchanges
    """
    print(f"Connecting to {host}:{port}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)

    try:
        sock.connect((host, port))
        print("Connected!")

        latencies = []

        for i in range(num_tests):
            # Send timestamp using ADD_MOVEMENT message (any message type
            # works)
            send_time = time.perf_counter()

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
            print(f"Tests completed: {len(latencies)}/{num_tests}")
            print(f"Min RTT:         {min(latencies):.2f}ms")
            print(f"Max RTT:         {max(latencies):.2f}ms")
            print(f"Mean RTT:        {statistics.mean(latencies):.2f}ms")
            print(f"Median RTT:      {statistics.median(latencies):.2f}ms")
            if len(latencies) > 1:
                print(f"Std Dev:         {statistics.stdev(latencies):.2f}ms")
            print(f"One-way (est):   {statistics.mean(latencies) / 2:.2f}ms")
            print("=" * 50)

    finally:
        protocol.close_socket(sock)
        print("Client closed")


def main():
    parser = argparse.ArgumentParser(
        description='Test latency between Pi and computer')
    parser.add_argument('--mode', choices=['server', 'client'], required=True,
                        help='Run as server (computer) or client (Pi)')
    parser.add_argument('--host', type=str, default='192.168.1.101',
                        help='Server IP address (client mode only)')
    parser.add_argument('--port', type=int, default=COMMAND_PORT,
                        help=f'Port to use (default: {COMMAND_PORT})')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of ping-pong tests (default: 100)')

    args = parser.parse_args()

    try:
        if args.mode == 'server':
            test_latency_server(args.port, args.num_tests)
        else:
            test_latency_client(args.host, args.port, args.num_tests)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
