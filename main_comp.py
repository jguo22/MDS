import time
import argparse
import cv2
import numpy as np
import math
import traceback
from nav import Nav
from typing import Optional, Tuple
from connection import config
from connection.ComputerReceiver import ComputerReceiver


class ClickProcessor:
    def __init__(self, window_name: str = "Pi Camera"):
        self.window_name = window_name
        self.frame_size = (1000, 1000)  # (width, height)
        # list of time of starting path, l_c, r_c, dist
        self.planned_moves: list[Tuple[float, float, float, float]] = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # using this for calcuations only
        self.nav = Nav()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start = time.time()
            # Convert to normalized coordinates (0-1)
            x_norm = x / (self.frame_size[1] - 1)
            y_norm = y / (self.frame_size[0] - 1)
            # Scale to range of [-scale, scale] (centered at 0)
            scale = 10
            x_scaled = (x_norm * scale * 2) - scale
            y_scaled = -((y_norm * scale * 2) - scale)
            print(
                f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}, {y_norm:.3f}) -> Scaled: ({x_scaled:.1f}, {y_scaled:.1f})")

            distance = math.sqrt(x_scaled * x_scaled + y_scaled * y_scaled)
            theta = math.atan(x_scaled / y_scaled)
            print(distance)
            print(theta)

            rotate = (time.time(), *self.nav.get_rotate(theta))
            move = (time.time() + 1, *self.nav.get_forward_mm(distance))
            self.planned_moves = [rotate, move]
            print(f'mouse  callback took: {time.time()-start}')

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float, float]]:
        # Update frame dimensions
        self.frame_size = (frame.shape[1], frame.shape[0])

        if len(self.planned_moves) == 0:
            return None

        # get the earliest planned move
        plan = self.planned_moves[0]
        # check if the first plan is ready to be executed
        if (time.time() >= plan[0]):
            self.planned_moves = self.planned_moves[1:]
            return plan[1:]
        else:
            return None


def main():
    parser = argparse.ArgumentParser(description="Computer Video Receiver")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--video-port", type=int, default=config.VIDEO_PORT,
                        help=f"Video port (default: {config.VIDEO_PORT})")
    parser.add_argument(
        "--coord-port",
        type=int,
        default=config.MOVEMENT_PORT,
        help=f"Coordinate port (default: {config.MOVEMENT_PORT})")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable video display")
    args = parser.parse_args()

    window_name = "Pi Camera"
    # Create receiver and click processor
    receiver = ComputerReceiver(args.host, args.video_port, args.coord_port)
    click_processor = ClickProcessor(window_name)

    # Set the frame callback to use our processor
    receiver.set_frame_callback(click_processor.process)

    # use protocol.sendmovement to send movement without waiting for frame

    # Start servers
    if not receiver.start_servers():
        return

    if not receiver.wait_for_connection():
        return

    try:
        print("Starting receive loop...")
        receiver.receive_loop(
            show_video=not args.no_display,
            window_name=window_name
        )
    except (ConnectionError, OSError) as e:
        traceback.print_exc()
        print(f"Connection error: {e}")
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        traceback.print_exc()
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
