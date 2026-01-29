import cv2
from config import FRAME_HEIGHT, FRAME_WIDTH
from IRobotCommander import IRobotCommander  # type: ignore
from vision.pixelTo3D import transform_uv_to_xy
from RobotHandler import RobotHandler
from RobotHandler_Simple import RobotHandlerSimple


class InputProcessor():
    def __init__(
            self,
            robot_commander: IRobotCommander,
            window_name: str,
            robotHandler: RobotHandlerSimple):
        self.robot_commander = robot_commander
        self.window_name = window_name
        self.robotHandler = robotHandler
        self.frame_size = (FRAME_WIDTH, FRAME_HEIGHT)  # (width, height)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = transform_uv_to_xy(x, y)
            if xy is not None:
                x_scaled, y_scaled = xy
                # self.robot_commander.override_relative_xy(x_scaled, y_scaled)

    # Interactive input thread for manual movement commands
    def handleKeyboardMovementsLoop(self):
        print("\n--- Manual Movement Control ---")
        print("Commands:")
        print("  r x y    - Send relative coordinates (robot-relative)")
        print("  w x y    - Send world coordinates (absolute position)")
        print("  pickup   - Pick up can with gripper")
        print("  release  - Release can from gripper")
        print("  start    - Start autonomous robot operation")
        print("  pause    - Pause autonomous operation")
        print("  resume   - Resume autonomous operation")
        print("  quit     - Exit\n")
        while True:
            try:
                user_input = input("Enter command: ").strip()
                if user_input.lower() == 'quit':
                    break
                if not user_input:
                    continue

                parts = user_input.split()

                # Start autonomous operation
                if user_input.lower() == 'start':
                    if self.robotHandler is not None:
                        self.robotHandler.started = True
                        print("  → Robot autonomous operation STARTED")
                    else:
                        print("  → Error: RobotHandler not available")

                # Pause autonomous operation
                elif user_input.lower() == 'pause':
                    if self.robotHandler is not None:
                        self.robotHandler.paused = True
                        print("  → Robot autonomous operation PAUSED")
                    else:
                        print("  → Error: RobotHandler not available")

                # Resume autonomous operation
                elif user_input.lower() == 'resume':
                    if self.robotHandler is not None:
                        self.robotHandler.paused = False
                        print("  → Robot autonomous operation RESUMED")
                    else:
                        print("  → Error: RobotHandler not available")

                # Pickup can
                elif user_input.lower() == 'pickup':
                    success = self.robot_commander.pickup_can()
                    if success:
                        print("  → Sent pickup can command")
                    else:
                        print(
                            "  → ERROR: Failed to send pickup command (no connection?)")

                # Release can
                elif user_input.lower() == 'release':
                    success = self.robot_commander.release_can()
                    if success:
                        print("  → Sent release can command")
                    else:
                        print(
                            "  → ERROR: Failed to send release command (no connection?)")

                # Relative coordinates: r x y
                elif len(parts) == 3 and parts[0].lower() == 'r':
                    x = float(parts[1])
                    y = float(parts[2])
                    success = self.robot_commander.override_relative_xy(x, y)
                    if success:
                        print(f"  → Sent relative movement: x={x}, y={y}")
                    else:
                        print(
                            "  → ERROR: Failed to send relative movement (no connection?)")

                # World coordinates: w x y
                elif len(parts) == 3 and parts[0].lower() == 'w':
                    x = float(parts[1])
                    y = float(parts[2])
                    success = self.robot_commander.override_world_xy(x, y)
                    if success:
                        print(f"  → Sent world coordinates: x={x}, y={y}")
                    else:
                        print(
                            "  → ERROR: Failed to send world coordinates (no connection?)")

                # Backward compatibility: plain x y defaults to relative
                elif len(parts) == 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    success = self.robot_commander.override_relative_xy(x, y)
                    if success:
                        print(
                            f"  → Sent relative movement: x={x}, y={y} (default mode)")
                    else:
                        print(
                            "  → ERROR: Failed to send relative movement (no connection?)")

                else:
                    print(
                        "Invalid command. Use: r x y (relative), w x y (world), pickup, release, start, pause, or resume")

            except ValueError:
                print("Invalid numbers. Try again.")
            except EOFError:
                break
            except KeyboardInterrupt:
                break
