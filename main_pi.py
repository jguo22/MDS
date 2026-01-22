import time
import argparse
from nav import Nav, NavMove
import threading
import traceback
from connection import config, message_types
from connection.PiStreamer import PiStreamer
from connection.CameraCapture import CameraCapture


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Video Streamer")
    parser.add_argument(
        "--camera",
        default="usb0",
        help="Camera source: usb0, usb1. (default: usb0)")
    args = parser.parse_args()

    # Create camera (managed externally, persists across reconnections)
    camera = CameraCapture(
        args.camera,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera.open():
        print(f"Failed to open camera: {args.camera}")
        return

    nav = Nav()
    # activate the navigation in another thread
    thread = threading.Thread(target=nav.startLoop, daemon=True)
    thread.start()

    def command_callback(messageType: int, args: list[float]):
        if messageType == message_types.ADD_MOVEMENT:
            assert (len(args) == 3)
            print(
                f"ADD_MOVEMENT: left={
                    args[0]}, right={
                    args[1]}, dist={
                    args[2]}")
            nav.addPath(NavMove(args[0], args[1], args[2], False))
        elif messageType == message_types.OVERRIDE_MOVEMENTS:
            assert (len(args) % 3 == 0)
            print(f"OVERRIDE_MOVEMENTS: {len(args) // 3} moves")
            moves = []
            for i in range(len(args) // 3):
                moves.append(
                    NavMove(args[3 * i], args[3 * i + 1], args[3 * i + 2], False))
            nav.overridePaths(moves)
        elif messageType == message_types.SEND_WORLD_XY:
            assert (len(args) == 2)
            world_x, world_y = args[0], args[1]
            print(f"SEND_WORLD_XY: x={world_x}, y={world_y}")
            nav.override_paths_world_xy(world_x, world_y)

    # Reconnection loop - each connection uses a new PiStreamer instance
    while True:
        try:
            print(f"\nConnecting to {config.COMPUTER_IP}...")

            # Create new streamer instance for this connection
            streamer = PiStreamer(
                camera,
                nav.raven,
                nav.imu_wrapper,
                host=config.COMPUTER_IP,
                video_port=config.VIDEO_PORT,
                command_port=config.COMMAND_PORT
            )

            # Set up movement callback
            streamer.set_command_callback(command_callback)

            # Attempt connection and stream
            if streamer.connect():
                print("Connected! Streaming...")
                streamer.stream(max_fps=config.DEFAULT_MAX_FPS)
                # stream() blocks until disconnected
                print("Stream ended")
            else:
                print("Connection failed")
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # Brief pause before reconnecting
            print(f"Reconnecting in {config.RECONNECT_DELAY}s...")
            time.sleep(config.RECONNECT_DELAY)

    camera.close()
    print("Camera closed")


if __name__ == "__main__":
    main()
