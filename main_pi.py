import time
import argparse
from nav import Nav
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
    parser.add_argument(
        "--coord-port",
        type=int,
        default=config.MOVEMENT_PORT,
        help=f"Coordinate port (default: {config.MOVEMENT_PORT})")
    args = parser.parse_args()

    # Create streamer
    streamer = PiStreamer(args.host, args.video_port, args.coord_port)
    nav = Nav()

    # activate the navigation in another thread
    thread = threading.Thread(target=nav.activate, daemon=True)
    thread.start()

    # Set up movement callback
    # movement callback gets called when the pi receives a movement command
    # from the computer in the form of l_c, r_c, dist,
    # and calls a function with those three arguments
    streamer.set_movement_callback(nav.startPath)

    # Start camera
    if not streamer.start_camera(args.camera):
        return

    # Connect and stream
    try:
        if streamer.connect():
            print("connected")
            streamer.stream(max_fps=args.fps)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        streamer.shutdown()


if __name__ == "__main__":
    main()
