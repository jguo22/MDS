import time
import argparse
import threading
from connection import config
from connection.ComputerReceiver import ComputerReceiver
from connection.frame_processor.ClickProcessor import ClickProcessor
from connection.frame_processor.SaveImageProcessor import SaveImageProcessor
import numpy as np

from handleKeyboardMovements import handleKeyboardMovementsLoop


def main():
    # ----------------- GET INPUTS -----------------
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

    # ----------------- CREATE RECEIVER AND PROCESSORS -----------------
    receiver = ComputerReceiver(args.host, args.video_port, args.coord_port)
    click_processor = ClickProcessor(receiver, window_name)
    save_image_processor = SaveImageProcessor(2)

    def process(frame: np.ndarray, frame_id: int) -> None:
        # Update frame dimensions
        # save_image_processor.process(frame, frame_id)
        click_processor.process(frame, frame_id)
        pass

    # Set the frame callback to use our processor
    receiver.set_frame_callback(process)

    # Start keyboard input thread
    threading.Thread(target=handleKeyboardMovementsLoop, daemon=True).start()

    # ----------------- START SERVERS -----------------
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
