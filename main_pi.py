import time
import argparse
from IMUWrapper import IMUWrapper
from RavenWrapper import RAVEN_WRAPPER
from nav import Nav, NavMove
import threading
import traceback
import config
from connection import message_types
from connection.PiStreamer import PiStreamer
from connection.CameraCapture import CameraCapture
from connection.DummyCameraCapture import DummyCameraCapture
from servos import gripClaw, releaseClaw, moveGripperHeight


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi Video Streamer")
    parser.add_argument("--camera-top", default="/dev/videoblacktop")
    parser.add_argument("--camera-bottom", default="/dev/videoblackbot")
    args = parser.parse_args()

    # Create cameras (managed externally, persists across reconnections)
    # Using DummyCameraCapture for testing (replace with CameraCapture for
    # real cameras)
    camera_top = DummyCameraCapture(
        args.camera_top,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera_top.open():
        print(f"Failed to open top camera: {args.camera_top}")
        return

    camera_bottom = DummyCameraCapture(
        args.camera_bottom,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera_bottom.open():
        print(f"Failed to open bottom camera: {args.camera_bottom}")
        camera_top.close()
        return

    imuWrapper = IMUWrapper()
    nav = Nav(imuWrapper)

    # activate the navigation in another thread
    thread = threading.Thread(target=nav.startLoop, daemon=True)
    thread.start()

    def command_callback(msg_type: int, args: list[float]):
        if msg_type == message_types.ADD_MOVEMENT:
            assert (len(args) == 3)
            print(
                f"ADD_MOVEMENT: left={
                    args[0]}, right={
                    args[1]}, dist={
                    args[2]}")
            nav.addPath(NavMove(args[0], args[1], args[2], False))

        elif msg_type == message_types.OVERRIDE_MOVEMENTS:
            assert (len(args) % 3 == 0)
            print(f"OVERRIDE_MOVEMENTS: {len(args) // 3} moves")
            moves = []
            for i in range(len(args) // 3):
                moves.append(
                    NavMove(args[3 * i], args[3 * i + 1], args[3 * i + 2], False))
            nav.overridePaths(moves)

        elif msg_type == message_types.SEND_WORLD_XY:
            assert (len(args) == 2)
            world_x, world_y = args[0], args[1]
            print(f"SEND_WORLD_XY: x={world_x}, y={world_y}")
            nav.override_paths_world_xy(world_x, world_y)

        elif msg_type == message_types.GRIP_CAN:
            assert (len(args) == 1)
            height = args[0]
            gripClaw()
            moveGripperHeight(height)

        elif msg_type == message_types.RELEASE_CAN:
            assert (len(args) == 1)
            height = args[0]
            moveGripperHeight(height)
            releaseClaw()

        elif msg_type == message_types.SEND_GRIPPER_HEIGHT:
            assert (len(args) == 1)
            height = args[0]
            print(f"SEND_GRIPPER_HEIGHT: height={height}")
            moveGripperHeight(height)

    # Reconnection loop - each connection uses a new PiStreamer instance
    running = False
    while running:
        try:
            print(f"\nConnecting to {config.COMPUTER_IP}...")

            # Create new streamer instance for this connection
            streamer = PiStreamer(
                camera_top,
                camera_bottom,
                RAVEN_WRAPPER,
                imuWrapper,
                host=config.COMPUTER_IP,
                video_port=config.VIDEO_PORT,
                command_port=config.COMMAND_PORT)

            # Set up movement callback
            streamer.set_command_callback(command_callback)

            # Attempt connection and stream
            if streamer.connect():
                print("Connected! Streaming...")
                streamer.stream(max_fps=config.FPS)
                # stream() blocks until disconnected
                print("Stream ended")
            else:
                print("Connection failed")
        except Exception as e:
            print(e)
            traceback.print_exc()
        except KeyboardInterrupt:
            print("\nShutting down...")
            running = False
        finally:
            # Brief pause before reconnecting
            if running:
                print(f"Reconnecting in {config.RECONNECT_DELAY}s...")
                time.sleep(config.RECONNECT_DELAY)

    camera_top.close()
    camera_bottom.close()
    print("Cameras closed")


if __name__ == "__main__":
    main()
