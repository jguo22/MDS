import time
import argparse
from IMUWrapper import IMUWrapper
from RavenWrapper import RAVEN_WRAPPER
from distanceSensorWrapper import DistanceSensorWrapper
from nav import Nav
import threading
import traceback
import config
from connection import message_types
from connection.PiStreamer import PiStreamer
from connection.CameraCapture import CameraCapture
from DirectRobotCommander import DirectRobotCommander


def run_network_mode(camera_top, camera_bottom, robot_commander, imu_wrapper):
    """Run network mode - stream to computer and receive commands."""
    print("\n=== RUNNING IN NETWORK MODE ===")
    print(f"Connecting to computer at {config.COMPUTER_IP}\n")

    def command_callback(msg_type: int, args: list[float]):
        """Route network commands to DirectRobotCommander."""
        if msg_type == message_types.OVERRIDE_MOVEMENTS:
            assert (len(args) % 3 == 0)
            print(f"OVERRIDE_MOVEMENTS: {len(args) // 3} moves")
            robot_commander.override_movement(args)

        elif msg_type == message_types.OVERRIDE_WAYPOINTS:
            assert (len(args) % 2 == 0)
            print(f"OVERRIDE_WAYPOINTS: {len(args) // 2 - 1} waypoints")
            robot_commander.override_waypoints(args)

        elif msg_type == message_types.OVERRIDE_RELATIVE_XY:
            assert (len(args) == 2)
            x, y = args[0], args[1]
            print(f"OVERRIDE_RELATIVE_XY: x={x}, y={y}")
            robot_commander.override_relative_xy(x, y)

        elif msg_type == message_types.OVERRIDE_WORLD_XY:
            assert (len(args) == 2)
            world_x, world_y = args[0], args[1]
            print(f"OVERRIDE_WORLD_XY: x={world_x}, y={world_y}")
            robot_commander.override_world_xy(world_x, world_y)

        elif msg_type == message_types.PICKUP_CAN:
            assert (len(args) == 0)
            print("PICKUP_CAN")
            robot_commander.pickup_can()

        elif msg_type == message_types.RELEASE_CAN:
            assert (len(args) == 0)
            print("RELEASE_CAN")
            robot_commander.release_can()

        elif msg_type == message_types.EARLY_GAME:
            assert (len(args) == 6)
            golden = (args[0], args[1])
            left = (args[2], args[3])
            right = (args[4], args[5])
            print(f"EARLY_GAME: golden={golden}, left={left}, right={right}")
            robot_commander.early_game(golden, left, right)

    # Reconnection loop - each connection uses a new PiStreamer instance
    running = True
    while running:
        try:
            print(f"\nConnecting to {config.COMPUTER_IP}...")

            # Create new streamer instance for this connection
            streamer = PiStreamer(
                camera_top,
                camera_bottom,
                RAVEN_WRAPPER,
                imu_wrapper,
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


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi Robot - Network or Autonomous Mode")
    parser.add_argument("--camera-top", default="/dev/videoblacktop",
                        help="Top camera device path")
    parser.add_argument("--camera-bottom", default="/dev/videoblackbot",
                        help="Bottom camera device path")
    parser.add_argument(
        "--local",
        help="Run autonomously on Pi without network (default: connect to computer)")
    parser.add_argument("--fps", type=int, default=config.FPS,
                        help="Target frames per second (local mode only)")
    args = parser.parse_args()

    print("Initializing hardware...")

    # Initialize IMU (must be first!)
    imu_wrapper = IMUWrapper()
    print("IMU initialized")

    # Initialize distance sensor
    distance_sensor = DistanceSensorWrapper()
    print("Distance sensor initialized")

    # Initialize navigation
    nav = Nav(imu_wrapper)
    print("Nav initialized")

    # Start navigation loop in background thread
    nav_thread = threading.Thread(target=nav.startLoop, daemon=True)
    nav_thread.start()
    print("Nav loop started")

    # Initialize cameras
    camera_top = CameraCapture(
        args.camera_top,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera_top.open():
        print(f"Failed to open top camera: {args.camera_top}")
        return

    camera_bottom = CameraCapture(
        args.camera_bottom,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera_bottom.open():
        print(f"Failed to open bottom camera: {args.camera_bottom}")
        camera_top.close()
        return
    print("Cameras initialized")

    # Create direct robot commander for command execution
    robot_commander = DirectRobotCommander(nav, distance_sensor, IMUWrapper())
    print("DirectRobotCommander initialized")

    # Branch based on mode
    print("network")
    run_network_mode(
        camera_top,
        camera_bottom,
        robot_commander,
        imu_wrapper)


if __name__ == "__main__":
    main()
