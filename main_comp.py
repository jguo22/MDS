import time
import argparse
import cv2
import numpy as np
import math
import threading
from nav import Nav
from typing import Optional, Tuple
from connection import config
from connection.ComputerReceiver import ComputerReceiver


class MovementCommander:
    def __init__(self, computerReceiver: ComputerReceiver):
        self.running = True

        self.planned_moves: list[Tuple[float, float, float, float]] = []
        self._lock = threading.Lock()

        self.computerReceiver = computerReceiver
        self.nav = Nav()

        threading.Thread(target=self._commandLoop, daemon=True).start()

    def queue_xy(self, x, y):
        """
        take in x,y in mm and plan send out instructions
        """
        distance = math.sqrt(x * x + y * y)

        # forward is y axis, so we want angle from y axis
        # while atan calculates angle from x axis
        theta = math.atan2(y, x) - math.pi / 2

        rotate = (time.time(), *self.nav.get_rotate(theta))
        move = (time.time() + 1, *self.nav.get_forward_mm(distance))

        print(
            f'sent movement x={x} y={y} theta={theta} distance={distance} rotate={rotate} move={move}')

        with self._lock:
            self.planned_moves = [rotate, move]

    def _commandLoop(self):
        while self.running:
            movement = []
            with self._lock:
                if self.planned_moves:
                    # get the earliest planned move
                    plan = self.planned_moves[0]
                    # check if the first plan is ready to be executed
                    if (time.time() >= plan[0]):
                        self.planned_moves = self.planned_moves[1:]
                        movement = plan[1:]
            if movement:
                self.computerReceiver.send_movement(*movement)

            time.sleep(1 / config.DEFAULT_MAX_FPS)

    def stop(self):
        self.running = False


class ClickProcessor:
    def __init__(
            self,
            movementCommander: MovementCommander,
            window_name: str = "Pi Camera"):
        self.movementCommander = movementCommander
        self.window_name = window_name
        self.frame_size = (1000, 1000)  # (width, height)
        # list of time of starting path, l_c, r_c, dist

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # using this for calcuations only
        self.nav = Nav()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert to normalized coordinates (0-1)
            x_norm = x / (self.frame_size[1] - 1)
            y_norm = y / (self.frame_size[0] - 1)
            # Scale to range of [-scale, scale] (centered at 0)
            scale = 10
            x_scaled = (x_norm * scale * 2) - scale
            y_scaled = -((y_norm * scale * 2) - scale)
            print(
                f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}, {y_norm:.3f}) -> Scaled: ({x_scaled:.1f}, {y_scaled:.1f})")

            self.movementCommander.queue_xy(x_scaled, y_scaled)

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float, float]]:
        # Update frame dimensions
        self.frame_size = (frame.shape[1], frame.shape[0])


def main():
    parser = argparse.ArgumentParser(description="Computer Video Receiver")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--video-port", type=int, default=config.VIDEO_PORT,
                        help=f"Video port (default: {config.VIDEO_PORT})")
    parser.add_argument(
        "--coord-port",
        type=int,
        default=config.COMMAND_PORT,
        help=f"Coordinate port (default: {config.COMMAND_PORT})")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable video display")
    args = parser.parse_args()

    window_name = "Pi Camera"
    # Create receiver and click processor
    receiver = ComputerReceiver(args.host, args.video_port, args.coord_port)
    movementCommander = MovementCommander(receiver)
    click_processor = ClickProcessor(movementCommander, window_name)

    # Set the frame callback to use our processor
    receiver.set_frame_callback(click_processor.process)

    # Interactive input thread for manual movement commands
    def input_thread():
        print("\n--- Manual Movement Control ---")
        print("Type two numbers separated by space (x y) and press Enter")
        print("Type 'quit' to exit\n")
        while True:
            try:
                user_input = input("Enter movement (x y): ").strip()
                if user_input.lower() == 'quit':
                    break
                if not user_input:
                    continue

                parts = user_input.split()
                if len(parts) == 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    movementCommander.queue_xy(x, y)
                    print(f"Queued movement to: x={x} y={y}")
            except ValueError:
                print("Invalid numbers. Try again.")
            except EOFError:
                break
            except KeyboardInterrupt:
                break

    # Start input thread
    threading.Thread(target=input_thread, daemon=True).start()

    # Start servers
    if not receiver.start_servers():
        return

    while True:
        print("connecting")
        if receiver.wait_for_connection():
            print("Starting receive loop...")
            receiver.receive_loop(
                show_video=not args.no_display,
                window_name=window_name
            )

        wait_time = config.RECONNECT_DELAY
        print(f"Waiting for {wait_time} seconds before reconnecting")
        time.sleep(wait_time)


if __name__ == "__main__":
    main()
