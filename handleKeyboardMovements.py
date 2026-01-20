from connection.ComputerReceiver import ComputerReceiver


# Interactive input thread for manual movement commands
def handleKeyboardMovementsLoop(receiver: ComputerReceiver):
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
