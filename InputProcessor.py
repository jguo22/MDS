import cv2
from config import FRAME_HEIGHT, FRAME_WIDTH
from connection.ComputerReceiver import ComputerReceiver
from yolo.pixelTo3D import transform_uv_to_xy
from RobotHandler import RobotHandler


class InputProcessor():
    def __init__(
            self,
            computerReceiver: ComputerReceiver,
            window_name: str,
            robotHandler: RobotHandler):
        self.computerReceiver = computerReceiver
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
                # self.computerReceiver.send_xy(x_scaled, y_scaled)

    # Interactive input thread for manual movement commands
    def handleKeyboardMovementsLoop(self):
        print("\n--- Manual Movement Control ---")
        print("Commands:")
        print("  r x y    - Send relative coordinates (robot-relative)")
        print("  w x y    - Send world coordinates (absolute position)")
        print("  start    - Start autonomous robot operation")
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

                # Relative coordinates: r x y
                elif len(parts) == 3 and parts[0].lower() == 'r':
                    x = float(parts[1])
                    y = float(parts[2])
                    self.computerReceiver.send_xy(x, y)
                    print(f"  → Sent relative movement: x={x}, y={y}")

                # World coordinates: w x y
                elif len(parts) == 3 and parts[0].lower() == 'w':
                    x = float(parts[1])
                    y = float(parts[2])
                    self.computerReceiver.send_world_xy(x, y)
                    print(f"  → Sent world coordinates: x={x}, y={y}")

                # Backward compatibility: plain x y defaults to relative
                elif len(parts) == 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    self.computerReceiver.send_xy(x, y)
                    print(
                        f"  → Sent relative movement: x={x}, y={y} (default mode)")

                else:
                    print("Invalid command. Use: r x y (relative) or w x y (world)")

            except ValueError:
                print("Invalid numbers. Try again.")
            except EOFError:
                break
            except KeyboardInterrupt:
                break
