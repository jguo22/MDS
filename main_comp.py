import time
import argparse
import threading
from MovementCommander import MovementCommander
from connection import config
from connection.ComputerReceiver import ComputerReceiver
from connection.frame_processor.ClickProcessor import ClickProcessor
from connection.frame_processor.SaveImageProcessor import SaveImageProcessor
from typing import Optional, Tuple
import numpy as np


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
    save_image_processor = SaveImageProcessor()

    def process(frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float, float]]:
        # Update frame dimensions
        save_image_processor.process(frame, frame_id)
        return click_processor.process(frame, frame_id)

    # Set the frame callback to use our processor
    receiver.set_frame_callback(process)

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
