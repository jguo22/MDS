import cv2
import numpy as np
from connection.ComputerReceiver import ComputerReceiver
from yolo.pixelTo3D import transform_uv_to_xy

from .FrameProcessor import FrameProcessor


class ClickAndKeyboardProcessor(FrameProcessor):
    def __init__(
            self,
            computerReceiver: ComputerReceiver,
            window_name: str = "Pi Camera"):
        self.computerReceiver = computerReceiver
        self.window_name = window_name
        self.frame_size = (640, 480)  # (width, height)
        # list of time of starting path, l_c, r_c, dist

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert to normalized coordinates (0-1)
            # x_norm = (x / (self.frame_size[1])) + 1 / self.frame_size[1] / 2
            # y_norm = y / (self.frame_size[0]) + 1 / self.frame_size[0] / 2
            # x = x_norm * 640
            # y = y_norm * 480
            # print(
            #     f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}")
            #
            # x_scaled, y_scaled = pixel_to_robot_horizontal(x, y)
            #
            # print(f'({x_scaled}, {y_scaled})')

            xy = transform_uv_to_xy(x, y)
            if xy is not None:
                x_scaled, y_scaled = xy
                self.computerReceiver.send_xy(x_scaled, y_scaled)

    def process(
            self,
            frame: np.ndarray,
            frame_id: int,
            x: float,
            y: float,
            theta: float):
        # Update frame dimensions
        self.frame_size = (frame.shape[1], frame.shape[0])
        return None

    # Interactive input thread for manual movement commands
    def handleKeyboardMovementsLoop(self):
        print("\n--- Manual Movement Control ---")
        print("Commands:")
        print("  r x y    - Send relative coordinates (robot-relative)")
        print("  w x y    - Send world coordinates (absolute position)")
        print("  quit     - Exit\n")
        while True:
            try:
                user_input = input("Enter command: ").strip()
                if user_input.lower() == 'quit':
                    break
                if not user_input:
                    continue

                parts = user_input.split()

                # Relative coordinates: r x y
                if len(parts) == 3 and parts[0].lower() == 'r':
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
