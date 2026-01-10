import time
import argparse
from nav import Nav
import math
import threading
from connection import config
from connection.PiStreamer import PiStreamer


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Video Streamer")
    parser.add_argument(
        "--host",
        default=config.COMPUTER_IP,
        help=f"Computer IP address (default: {config.COMPUTER_IP})")
    parser.add_argument(
        "--camera",
        default="usb0",
        help="Camera source: usb0, usb1, picamera0, etc. (default: usb0)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Maximum FPS (default: 30)")
    parser.add_argument("--video-port", type=int, default=config.VIDEO_PORT,
                        help=f"Video port (default: {config.VIDEO_PORT})")
    parser.add_argument("--coord-port", type=int, default=config.COORD_PORT,
                        help=f"Coordinate port (default: {config.COORD_PORT})")
    args = parser.parse_args()

    # Create streamer
    streamer = PiStreamer(args.host, args.video_port, args.coord_port)
    nav = Nav()

    thread = threading.Thread(target=nav.activate)

    # Start the thread's execution
    thread.start()

    # Set up coordinate callback

    def on_coords(x, y, frame_id, extra):
        print(f"Received coords: x={x:.2f}, y={y:.2f}, frame={frame_id}")
        distance = math.sqrt(x * x + y * y)
        theta = math.atan(x / y)
        print(distance)
        print(theta)
        nav.start_rotate(theta)
        time.sleep(1)  # TODO: make it actually check when its finished
        nav.start_forward_mm(distance)

    streamer.set_coordinate_callback(on_coords)

    # Start camera
    if not streamer.start_camera(args.camera):
        return

    # Connect and stream
    while True:
        if streamer.connect():
            streamer.stream(max_fps=args.fps)
        print(f"Reconnecting in {config.RECONNECT_DELAY}s...")
        time.sleep(config.RECONNECT_DELAY)


if __name__ == "__main__":
    main()
