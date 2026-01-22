import time
import argparse
from nav import Nav, NavMove, get_rotate, get_forward_mm
import threading
import traceback
from connection import config, message_types
from connection.PiStreamer import PiStreamer
from connection.CameraCapture import CameraCapture
from raven import Raven
from yolo.segment import getQuadrilateralsAndClasses, segmentImage, getClassName
from pixelTo3D import transform_uv_to_xy
import numpy as np
from thetaStar import ThetaStarPlanner


RobotState = "Finding Can"
heldCanColor = "None"

grid_size = 0.5  # [m]
robot_radius = 1.0  # [m]

# set obstacle positions
ox, oy = [], []
# Can locations
cx, cy = [], []
# Zone locations
red_zone_x, red_zone_y = 0, 0
green_zone_x, green_zone_y = 0, 0
yellow_zone_x, yellow_zone_y = 0, 0

def main():

    def getCanColor():
        rotateCameraDown()
        print("a")
        time.sleep(0.1)
        print("b")
        image = camera.cap.read()[1]
        result = segmentImage(image)
        detections = result.boxes
        lowest_y = 10000
        class_name = "None"
        for detection in detections:
            print(detection.conf.item())
            if (detection.conf.item() < 0.6):
                continue
            xyxy_tensor = detection.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            if ymax < lowest_y:
                lowest_y = ymax
                classidx = int(detection.cls.item())
                class_name = getClassName(classidx)

        print(class_name)


    # Set down to be looking directly down at held can
    def rotateCameraDown():
        nav.raven.set_servo_position(Raven.ServoChannel.CH4, -90)
    def rotateCameraUp():
        nav.raven.set_servo_position(Raven.ServoChannel.CH4, 30)


    parser = argparse.ArgumentParser(description="Raspberry Pi Video Streamer")
    parser.add_argument(
        "--camera",
        default="usb0",
        help="Camera source: usb0, usb1. (default: usb0)")
    args = parser.parse_args()

    # Create camera (managed externally, persists across reconnections)
    camera = CameraCapture(
        args.camera,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT)
    if not camera.open():
        print(f"Failed to open camera: {args.camera}")
        return

    nav = Nav()

    if False:
        segmentImage(camera.cap.read()[1])
        image = camera.cap.read()[1]
        result = segmentImage(image)

        quads, classes = getQuadrilateralsAndClasses(result, image)

        # Transform each vertex from pixel coordinates to ground plane coordinates
        transformed_quads = []
        for quad in quads:
            # quad shape is (4, 1, 2) -> reshape to (4, 2)
            vertices = quad.reshape(4, 2)

            # Transform each vertex
            transformed_vertices = []
            for vertex in vertices:
                u, v = vertex[0], vertex[1]  # pixel coordinates
                x, y = transform_uv_to_xy(u, v)  # ground plane coordinates (mm)
                transformed_vertices.append([x, y])

            transformed_quads.append(np.array(transformed_vertices))

        # Calculate center xy point of each quad
        centers = []
        for transformed_quad in transformed_quads:
            # Average all x and y coordinates
            center_x = np.mean(transformed_quad[:, 0])
            center_y = np.mean(transformed_quad[:, 1])
            centers.append((center_x, center_y))

        # Get only the closest of each class
        # Group by class and find closest (minimum distance from origin/robot)
        temp_closest = {}
        for i, (center, class_name) in enumerate(zip(centers, classes)):
            x, y = center
            # Distance from robot (at origin): sqrt(x^2 + y^2)
            distance = np.sqrt(x**2 + y**2)

            if class_name not in temp_closest:
                temp_closest[class_name] = {
                    'index': i,
                    'center': center,
                    'distance': distance,
                    'quad': transformed_quads[i]
                }
            else:
                # Update if this one is closer
                if distance < temp_closest[class_name]['distance']:
                    temp_closest[class_name] = {
                        'index': i,
                        'center': center,
                        'distance': distance,
                        'quad': transformed_quads[i]
                    }


        # Convert to parallel arrays
        closest_quads_xy = []  # Array of quads with xy coordinates
        closest_centers = []   # Array of xy center points
        closest_classes = []   # Array of class names

        for class_name, data in temp_closest.items():
            closest_quads_xy.append(data['quad'])
            closest_centers.append(data['center'])
            closest_classes.append(class_name)

        print(f"Found {len(closest_classes)} unique classes:")
        for i, class_name in enumerate(closest_classes):
            x, y = closest_centers[i]
            distance = temp_closest[class_name]['distance']
            print(f"  {class_name}: center=({x:.1f}, {y:.1f}) mm, distance={distance:.1f} mm")

        print(closest_centers)
        print(closest_classes)

    # TODO: Set zone locations


    rotateCameraDown()
    getCanColor()
# Smooth = False
    # nav.addPath(NavMove(1.005, .995, get_forward_mm(2100)[2], False, True))
    # nav.addPath(NavMove(1.3, 0.7, get_rotate() / 2 - 200, False, True))
    # nav.addPath(NavMove(1.12, .88, get_forward_mm(700)[2], False, True))
    # nav.addPath(NavMove(1.5, 0.5, get_rotate() / 2 - 300, False, True))
    # nav.addPath(NavMove(1, 1, 5000, False, True))
    # # x, y = closest_centers[0]
    # # nav.override_paths_world_xy(x, y)


    # nav.addPath(NavMove(-1, -1, 4000, False, True))
    # nav.addPath(NavMove(-1.5, -0.5, get_rotate() / 2 - 100, False, True))

    # nav.addPath(NavMove(-1, -1, 7000, False, True))
    # nav.addPath(NavMove(-1.5, -0.5, get_rotate() / 2- 200, False, True))
    # nav.addPath(NavMove(0.7, 1.3, get_rotate() / 2, False, True))
    # nav.addPath(NavMove(.88, 1.12, get_forward_mm(700)[2], False, True))
    # nav.addPath(NavMove(0.5, 1.5, get_rotate() / 2 - 300, False, True))
    # nav.addPath(NavMove(1, 1, 5000, False, True))
# Smooth = True
    # nav.addPath(NavMove(1.05, 0.95, 10000, True, True))
    # nav.addPath(NavMove(1.3, 0.7, get_rotate() / 2, True, True))
    # nav.addPath(NavMove(1.05, 0.95, 10000, True, True))
    # nav.addPath(NavMove(1.5, 0.5, 3000, True, False))
    # nav.addPath(NavMove(1, 1, 2000, True, True))
    # nav.addPath(NavMove(-1, -1, 2000, True, True))
    # nav.addPath(NavMove(-0.5, -1.5, 3700, True, False))
    # nav.addPath(NavMove(1, 1, 25000, True, True))

    def movement_callback(messageType: int, args: list[float]):
        if messageType == message_types.ADD_MOVEMENT:
            assert (len(args) == 3)
            print(
                f"ADD_MOVEMENT: left={args[0]}, right={args[1]}, dist={args[2]}")
            nav.addPath(NavMove(args[0], args[1], args[2], False))
        elif messageType == message_types.OVERRIDE_MOVEMENTS:
            assert (len(args) % 3 == 0)
            print(f"OVERRIDE_MOVEMENTS: {len(args)//3} moves")
            moves = []
            for i in range(len(args) // 3):
                moves.append(
                    NavMove(args[3 * i], args[3 * i + 1], args[3 * i + 2], False))
            nav.overridePaths(moves)

    # activate the navigation in another thread
    thread = threading.Thread(target=nav.startLoop, daemon=True)
    thread.start()

    if False:
        while True:
            if RobotState == "Finding Can":
                if len(cx) == 0:
                    print("No cans found!")
                    RobotState = "Done"
                else:
                    # Go to first can
                    nav.override_paths_world_xy(cx[0], cy[0])
                    RobotState = "Going to Can"
            elif RobotState == "Going to Can":
                # Check if reached can
                if nav.moving == False:
                    RobotState = "Grabbing Can"
            elif RobotState == "Grabbing Can":
                nav.grabCan() # TODO: implement grabCan
                heldCanColor = nav.getCanColor() # TODO: implement getCanColor
                RobotState = "Scoring Can"
                if heldCanColor == "Red":
                    gx = red_zone_x
                    gy = red_zone_y
                elif heldCanColor == "Green":
                    gx = green_zone_x
                    gy = green_zone_y
                elif heldCanColor == "Yellow":
                    gx = yellow_zone_x
                    gy = yellow_zone_y
                theta_star = ThetaStarPlanner(ox, oy, grid_size, robot_radius)
                robot_x, robot_y = nav.raven.get_odometry()
                rx, ry = theta_star.planning(robot_x, robot_y, gx, gy)
                for (x, y) in zip(rx, ry):
                    nav.add_paths_world_xy(x, y) # TODO: implement addPathsWorldXY
            elif RobotState == "Scoring Can":
                if nav.moving == False:
                    nav.releaseGrip() # TODO: implement releaseGrip
                    RobotState = "Finding Can"

    # Reconnection loop - each connection uses a new PiStreamer instance
    while True:
        try:
            print(f"\nConnecting to {config.COMPUTER_IP}...")

            # Create new streamer instance for this connection
            streamer = PiStreamer(
                camera=camera,
                host=config.COMPUTER_IP,
                video_port=config.VIDEO_PORT,
                command_port=config.COMMAND_PORT
            )

            # Set up movement callback
            streamer.set_movement_callback(movement_callback)

            # Attempt connection and stream
            if streamer.connect():
                print("Connected! Streaming...")
                streamer.stream(max_fps=config.DEFAULT_MAX_FPS)
                # stream() blocks until disconnected
                print("Stream ended")
            else:
                print("Connection failed")
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            # Brief pause before reconnecting
            print(f"Reconnecting in {config.RECONNECT_DELAY}s...")
            time.sleep(config.RECONNECT_DELAY)

    camera.close()
    print("Camera closed")

if __name__ == "__main__":
    main()

def gripClaw(nav: Nav):
    nav.raven.set_servo_position(Raven.ServoChannel.CH1, 90)
def releaseClaw(nav: Nav):
    nav.raven.set_servo_position(Raven.ServoChannel.CH1, -90)

def moveElevatorUp(nav: Nav):
    nav.raven.set_servo_position(Raven.ServoChannel.CH4, 90)
    time.sleep(1.2)
    nav.raven.set_servo_position(Raven.ServoChannel.CH4, 0)
def moveElevatorDown(nav: Nav):
    nav.raven.set_servo_position(Raven.ServoChannel.CH4, -90)
    time.sleep(1.2)
    nav.raven.set_servo_position(Raven.ServoChannel.CH4, 0)
