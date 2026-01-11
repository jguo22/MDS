import time
import argparse
import cv2
import numpy as np
import math
from nav import Nav
from typing import Optional, Tuple
from connection import config
from connection.ComputerReceiver import ComputerReceiver


class ClickProcessor:
    def __init__(self, window_name: str = "Pi Camera"):
        self.window_name = window_name
        self.click_coords = None
        self.frame_size = (1000, 1000)  # (width, height)
        # list of time of starting path, l_c, r_c, dist
        self.planned_moves: list[Tuple[float, float, float, float]] = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        # using this for calcuations only
        self.nav = Nav()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frame_size[0] > 0 and self.frame_size[1] > 0:
                # Convert to normalized coordinates (0-1)
                x_norm = x / (self.frame_size[0] - 1)
                y_norm = y / (self.frame_size[1] - 1)
                scale = 10
                # Scale to -150 to 150 range (centered at 0)
                x_scaled = (x_norm * scale) - scale / 2
                y_scaled = -(y_norm * scale) - scale / 2
                self.click_coords = (x_scaled, y_scaled)
                print(
                    f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}, {y_norm:.3f}) -> Scaled: ({x_scaled:.1f}, {y_scaled:.1f})")

                distance = math.sqrt(x * x + y * y)
                theta = math.atan(x / y)
                print(distance)
                print(theta)

                rotate = (time.time(), *self.nav.get_rotate(theta))
                move = (time.time() + 1, *self.nav.get_forward_mm(distance))
                self.planned_moves = [rotate, move]

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
    parser.add_argument("--coord-port", type=int, default=config.COORD_PORT,
                        help=f"Coordinate port (default: {config.COORD_PORT})")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable video display")
    args = parser.parse_args()

    window_name = "Pi Camera"
    # Create receiver and click processor
    receiver = ComputerReceiver(args.host, args.video_port, args.coord_port)
    click_processor = ClickProcessor(window_name)

    # Set the frame callback to use our processor
    receiver.set_frame_callback(click_processor.process)

    # Start servers
    if not receiver.start_servers():
        return

    # Main loop - wait for connections and process
    def restart_servers():
        """Close everything and restart servers."""
        # Close client connections
        if receiver.client_video:
            try:
                receiver.client_video.close()
            except BaseException:
                pass
        if receiver.client_coord:
            try:
                receiver.client_coord.close()
            except BaseException:
                pass
        receiver.client_video = None
        receiver.client_coord = None

        # Close servers
        if receiver.video_server:
            try:
                receiver.video_server.close()
            except BaseException:
                pass
        if receiver.coord_server:
            try:
                receiver.coord_server.close()
            except BaseException:
                pass
        receiver.video_server = None
        receiver.coord_server = None

        # Restart servers
        time.sleep(0.5)  # Brief delay before rebinding
        return receiver.start_servers()

    while True:
        if not receiver.wait_for_connection():
            print("Connection failed. Restarting servers...")
            if not restart_servers():
                print("Failed to restart servers. Retrying...")
                time.sleep(1)
            continue

        try:
            print("Starting receive loop...")
            receiver.receive_loop(
                show_video=not args.no_display,
                window_name=window_name
            )
        except (ConnectionError, OSError) as e:
            print(f"Connection error: {e}")
            print("Disconnected. Restarting servers...")
            if not restart_servers():
                print("Failed to restart servers. Retrying...")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Restarting servers...")
            if not restart_servers():
                print("Failed to restart servers. Retrying...")
                time.sleep(1)


if __name__ == "__main__":
    main()
