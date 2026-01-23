import time
import argparse
import threading
from RobotHandler import RobotHandler
from connection import config
from connection.ComputerReceiver import ComputerReceiver
from connection.frame_processor.ClickProcessor import ClickAndKeyboardProcessor
from connection.frame_processor.SaveImageProcessor import SaveImageProcessor
import numpy as np


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
    computer_receiver = ComputerReceiver(
        args.host, args.video_port, args.coord_port)
    inputProcessor = ClickAndKeyboardProcessor(computer_receiver, window_name)
    save_image_processor = SaveImageProcessor(2)
    robotHandler = RobotHandler(computer_receiver)

    def process(
            frame: np.ndarray,
            frame_id: int,
            x: float,
            y: float,
            theta: float) -> None:
        # Update frame dimensions
        # save_image_processor.process(frame, frame_id)
        inputProcessor.process(frame, frame_id, x, y, theta)
        # robotHandler.handleFrame(frame, frame_id, x, y, theta)

    # Set the frame callback to use our processor
    computer_receiver.set_frame_callback(process)

    # Start keyboard input thread
    threading.Thread(
        target=inputProcessor.handleKeyboardMovementsLoop,
        daemon=True).start()

    # ----------------- START SERVERS -----------------
    if not computer_receiver.start_servers():
        return

    while True:
        print("connecting")
        if computer_receiver.wait_for_connection():
            print("Starting receive loop...")
            try:
                computer_receiver.receive_loop(
                    show_video=not args.no_display,
                    window_name=window_name
                )
            except KeyboardInterrupt:
                print("\nSaving profiler data before exit...")
                robotHandler.profiler.save_profile()
                raise

        wait_time = config.RECONNECT_DELAY
        print(f"Waiting for {wait_time} seconds before reconnecting")
        time.sleep(wait_time)


if __name__ == "__main__":
    main()
