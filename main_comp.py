import time
import argparse
import cv2
import numpy as np
from typing import Optional, Tuple
from connection import config
from connection.computer_receiver import ComputerReceiver


class ClickProcessor:
    def __init__(self, window_name: str = "Pi Camera"):
        self.window_name = window_name
        self.click_coords = None
        self.frame_size = (1000, 1000)  # (width, height)
        self._setup = False

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frame_size[0] > 0 and self.frame_size[1] > 0:
                # Convert to normalized coordinates (0-1)
                x_norm = x / (self.frame_size[0] - 1)
                y_norm = y / (self.frame_size[1] - 1)
                # Scale to -150 to 150 range (centered at 0)
                x_scaled = (x_norm * 300) - 150
                y_scaled = (y_norm * 300) - 150
                self.click_coords = (x_scaled, y_scaled)
                print(
                    f"Click: ({x}, {y}) -> Normalized: ({x_norm:.3f}, {y_norm:.3f}) -> Scaled: ({x_scaled:.1f}, {y_scaled:.1f})")

    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float]]:
        if not self._setup:
            cv2.namedWindow(self.window_name)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            self._setup = True

        # Update frame dimensions
        self.frame_size = (frame.shape[1], frame.shape[0])

        # Return and clear click coordinates
        coords = self.click_coords
        self.click_coords = None
        return coords


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
    while True:
        if receiver.wait_for_connection():
            try:
                # run the receiving loops
                # which displays video
                # and runs the frame callback
                receiver.receive_loop(
                    show_video=not args.no_display,
                    window_name=window_name)
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in receive_loop: {e}")
        print("Waiting for reconnection...")
        time.sleep(config.RECONNECT_DELAY)


if __name__ == "__main__":
    main()
