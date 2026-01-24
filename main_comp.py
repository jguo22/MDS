import time
import argparse
import threading
from profiler import Profiler
from RobotHandler import RobotHandler
from connection.ComputerReceiver import ComputerReceiver
from connection.frame_processor.ClickProcessor import ClickAndKeyboardProcessor
from connection.frame_processor.SaveImageProcessor import SaveImageProcessor
from connection.frame_info import FrameInfo
import config


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

    window_name_top = "Top Camera"
    window_name_bottom = "Bottom Camera"

    # ----------------- CREATE RECEIVER AND PROCESSORS -----------------
    computer_receiver = ComputerReceiver(
        args.host, args.video_port, args.coord_port)
    inputProcessor = ClickAndKeyboardProcessor(computer_receiver, window_name_top)
    _save_image_processor = SaveImageProcessor(2)
    robotHandler = RobotHandler(computer_receiver)

    # Create main profiler for frame processing pipeline
    main_profiler = Profiler()

    def process(frame_info: FrameInfo) -> None:
        # Process frame using FrameInfo
        main_profiler.start_frame()
        inputProcessor.process(frame_info)
        main_profiler.record("inputProcessor")
        robotHandler.handleFrame(frame_info)
        main_profiler.record("robotHandler")
        main_profiler.end_frame()

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
                    window_name_top=window_name_top,
                    window_name_bottom=window_name_bottom
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
