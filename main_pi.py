import time
import argparse
from IMUWrapper import IMUWrapper
from IRobotCommander import IRobotCommander
from RavenWrapper import RAVEN_WRAPPER
from distanceSensorWrapper import DistanceSensorWrapper
from nav import Nav
import threading
import traceback
import config
from connection import message_types
from connection.PiStreamer import PiStreamer
from connection.CameraCapture import CameraCapture
from connection import command_tracker
from DirectRobotCommander import DirectRobotCommander


def run_network_mode(
        camera_top,
        camera_bottom,
        robot_commander: IRobotCommander,
        imu_wrapper,
        nav):
    """Run network mode - stream to computer and receive commands."""

    def command_callback(msg_type: int, command_id: int, args: list[float]):
        """Route network commands to DirectRobotCommander."""

        # All commands now block until complete
        if msg_type == message_types.OVERRIDE_MOVEMENTS:
            assert (len(args) % 3 == 0)
            robot_commander.override_movement(args)

        elif msg_type == message_types.OVERRIDE_WAYPOINTS:
            assert (len(args) % 2 == 0)
            robot_commander.override_waypoints(args)

        elif msg_type == message_types.OVERRIDE_RELATIVE_XY:
            assert (len(args) == 2)
            x, y = args[0], args[1]
            robot_commander.override_relative_xy(x, y)

        elif msg_type == message_types.OVERRIDE_WORLD_XY:
            assert (len(args) == 2)
            world_x, world_y = args[0], args[1]
            robot_commander.override_world_xy(world_x, world_y)

        elif msg_type == message_types.APPROACH_CAN_DS:
            assert (len(args) == 0)
            robot_commander.approach_can_with_ds()

        elif msg_type == message_types.EARLY_GAME:
            assert (len(args) == 6)
            golden = (args[0], args[1])
            left = (args[2], args[3])
            right = (args[4], args[5])
            robot_commander.early_game(golden, left, right)

        elif msg_type == message_types.STACK:
            assert (len(args) == 5)
            temp_pos = (args[0], args[1])
            stack_pos = (args[2], args[3])
            stacked_cans = int(args[4])
            robot_commander.stack(temp_pos, stack_pos, stacked_cans)

        # Non-movement commands (complete immediately)
        elif msg_type == message_types.PICKUP_CAN:
            assert (len(args) == 0)
            robot_commander.pickup_can()

        elif msg_type == message_types.RELEASE_CAN:
            assert (len(args) == 0)
            robot_commander.release_can()

        elif msg_type == message_types.RESET_GRIPPER:
            assert (len(args) == 0)
            robot_commander.reset_gripper()

        elif msg_type == message_types.SET_DOWN_CAN:
            assert (len(args) == 0)
            robot_commander.set_down_can()

        elif msg_type == message_types.BACKUP:
            assert (len(args) == 0)
            robot_commander.backup()

        elif msg_type == message_types.WAIT_MOVEMENT_FINISHED:
            assert (len(args) == 0)
            robot_commander.waitFinishedMoving()

        # Mark command complete in tracker
        command_tracker.mark_complete(command_id)

    # Reconnection loop - each connection uses a new PiStreamer instance
    running = True
    while running:
        try:
            # Create new streamer instance for this connection
            streamer = PiStreamer(
                camera_top,
                camera_bottom,
                RAVEN_WRAPPER,
                imu_wrapper,
                nav,
                robot_commander,
                host=config.COMPUTER_IP,
                video_port=config.VIDEO_PORT,
                command_port=config.COMMAND_PORT)

            # Set up movement callback
            streamer.set_command_callback(command_callback)

            # Attempt connection and stream
            if streamer.connect():
                streamer.stream(max_fps=config.FPS)
                # stream() blocks until disconnected
        except Exception:
            traceback.print_exc()
        except KeyboardInterrupt:
            running = False
        finally:
            # Brief pause before reconnecting
            if running:
                time.sleep(config.RECONNECT_DELAY)

    camera_top.close()
    camera_bottom.close()


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

    # Initialize IMU (must be first!)
    imu_wrapper = IMUWrapper()

    # Initialize distance sensor
    distance_sensor = DistanceSensorWrapper()

    # Initialize navigation
    nav = Nav(imu_wrapper)

    # Start navigation loop in background thread
    nav_thread = threading.Thread(target=nav.startLoop, daemon=True)
    nav_thread.start()

    # Initialize cameras
    camera_top = CameraCapture(
        args.camera_top,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera_top.open():
        return

    camera_bottom = CameraCapture(
        args.camera_bottom,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera_bottom.open():
        camera_top.close()
        return

    # Create direct robot commander for command execution
    robot_commander = DirectRobotCommander(nav, distance_sensor, imu_wrapper)

    # Branch based on mode
    run_network_mode(
        camera_top,
        camera_bottom,
        robot_commander,
        imu_wrapper,
        nav)


if __name__ == "__main__":
    main()
