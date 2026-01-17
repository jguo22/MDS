import time
import argparse
import threading
from connection import config
from connection.ComputerReceiver import ComputerReceiver
from connection.frame_processor.ClickProcessor import ClickProcessor
from connection.frame_processor.SaveImageProcessor import SaveImageProcessor
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
    click_processor = ClickProcessor(receiver, window_name)
    save_image_processor = SaveImageProcessor(2)

    def process(frame: np.ndarray,
                frame_id: int) -> None:
        # Update frame dimensions
        # save_image_processor.process(frame, frame_id)
        click_processor.process(frame, frame_id)

    # Set the frame callback to use our processor
    receiver.set_frame_callback(process)

    # Interactive input thread for manual movement commands
    def input_thread():
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
                    receiver.send_xy(x, y)
                    print(f"  → Sent relative movement: x={x}, y={y}")

                # World coordinates: w x y
                elif len(parts) == 3 and parts[0].lower() == 'w':
                    x = float(parts[1])
                    y = float(parts[2])
                    receiver.send_world_xy(x, y)
                    print(f"  → Sent world coordinates: x={x}, y={y}")

                # Backward compatibility: plain x y defaults to relative
                elif len(parts) == 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    receiver.send_xy(x, y)
                    print(f"  → Sent relative movement: x={x}, y={y} (default mode)")

                else:
                    print("Invalid command. Use: r x y (relative) or w x y (world)")

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
