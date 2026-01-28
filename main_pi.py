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
from coordinates.relativeCoordinates import relative_to_world

from thetaStar import addCan, addBorder
import math

RobotState = "Finding Can"
heldCanColor = "None"

grid_size = 0.5  # [m]
robot_radius = 1.0  # [m]

# set obstacle positions
ox, oy = [], []
# Can locations
cx, cy = [], []
red_zone_x, red_zone_y = 0, 0
green_zone_x, green_zone_y = 0, 0
yellow_zone_x, yellow_zone_y = 0, 0
golden_x, golden_y = 0, 0

# locations of collected cans
ccx = []
ccy = []

# Positions for stacking
center_pos = (1200, 0)
offset_pos = (1200, -400)
stacked_cans = 0
can_in_center_pos = True

def main():
    def getCanColor():
        print("a")
        time.sleep(0.1)
        print("b")
        image = camera.cap.read()[1]
        result = segmentImage(image)
        detections = result.boxes
        lowest_y = 0
        class_name = "None"
        for detection in detections:
            print(detection.conf.item())
            if (detection.conf.item() < 0.6):
                continue
            xyxy_tensor = detection.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            if ymax > lowest_y:
                lowest_y = ymax
                classidx = int(detection.cls.item())
                class_name = getClassName(classidx)

        print(class_name)

    def path_find(goal_x, goal_y):
        theta_star = ThetaStarPlanner(ox, oy, grid_size, robot_radius)
        robot_x, robot_y = nav.ravenWrapper.get_odometry()
        rx, ry = theta_star.planning(robot_x, robot_y, goal_x, goal_y)
        for (x, y) in zip(rx, ry):
            nav.override_paths_world_xy(x, y)
            time.sleep(5)
        # TODO: make the last path have the claw go to the x y, not the robot 0 0

    def store_can_locations():
        global cx, cy, golden_x, golden_y
        image = camera.cap.read()[1]
        result = segmentImage(image)
        detections = result.boxes

        for detection in detections:
            if (detection.conf.item() < 0.6):
                continue
            xyxy_tensor = detection.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx = int(detection.cls.item())
            class_name = getClassName(classidx)

            if "Can" not in class_name:
                continue
            x, y = relative_to_world(transform_uv_to_xy((xmin + xmax) / 2, ymax)) # TODO: properly convert relative to world

            cx.append(x)
            cy.append(y)

            if class_name == "Golden Can":
                golden_x = x
                golden_y = y
        # TODO: move to golden pringle can
        # Sort the cans from leftmost to rightmost
        y_sorted, x_sorted = zip(*sorted(zip(cy, cx)))

        cx = list(x_sorted)
        cy = list(y_sorted)

    def goto_golden_can():
        nav.ravenWrapper.lower_left_arm()
        nav.override_paths_world_xy(golden_x, golden_y)

    def getCans():
        global right_cans, left_cans
        print("current pos is ", nav.ravenWrapper.get_odometry())
        # Collect right side cans first
        print("collecting right cans")
        # TODO: change these from override to add paths
        nav.override_paths_world_xy(cx[-1], cy[-1])
        print(f"going to rightmost can at {cx[-1]}, {cy[-1]}")
        time.sleep(2)
        # Deposit cans on our side
        nav.addPath(NavMove(1.3, 0.7, 6000, False, True))
        time.sleep(2)
        # TODO: STORE all can LOCATIONs from down facing camera
        x, y = nav.get_world_claw_position()
        ccx.append(x)
        ccy.append(y)
        nav.addPath(NavMove(-1.3, -0.7, 6000, False, True))
        time.sleep(1)
        nav.ravenWrapper.raise_left_arm()
        time.sleep(1)
        # Collect cans from other side
        print("collecting left cans now")
        nav.addPath(NavMove(*get_rotate(math.pi), False, False))
        time.sleep(1)
        nav.ravenWrapper.lower_right_arm()
        time.sleep(1)
        print(f"going to leftmost can at {cx[0]} {cy[0]}")
        nav.override_paths_world_xy(cx[0], cy[0])
        time.sleep(2)
        # Deposit cans on our side
        nav.addPath(NavMove(0.7, 1.3, 6000, False, True))
        # TODO: STORE all can LOCATIONS from down facing camera
        x, y = nav.get_world_claw_position()
        ccx.append(x)
        ccy.append(y)
        time.sleep(2)
        nav.addPath(NavMove(-0.7, -1.3, 6000, False, True))
        time.sleep(2)
        nav.ravenWrapper.raise_right_arm()
        nav.override_paths_world_xy(0, 0)
        time.sleep(5)
        # Next movement
        go_to_closest_can()
    # All the cans have been collected on our side
    def go_to_closest_can():
        # Find closest can
        x, y = nav.ravenWrapper.get_odometry()
        min_distance = float('inf')
        for i in range(len(ccx)):
            can_x = ccx[i]
            can_y = ccy[i]
            distance = ((can_x - x)**2 + (can_y - y)**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_can_x = can_x
                closest_can_y = can_y
        nav.override_paths_world_xy(closest_can_x, closest_can_y, use_claw=True)
        time.sleep(5)
        # Next movement
        grab_closest_can()

    def approach_can_with_ds():
        # Approach can with distance sensor
        while nav.distance_sensor.get_distance() > 100:
            nav.overridePaths([NavMove(*get_forward_mm(nav.distance_sensor.get_distance() - 85))])
            time.sleep(1.2)


    # Use camera to grab closest can
    def grab_closest_can():
        # TODO: Use down facing camera to get image of cans
        # TODO: get closest can, rotate towards it
        approach_can_with_ds()
        nav.ravenWrapper.lower_elevator()
        time.sleep(1.5)
        nav.ravenWrapper.close_gripper()
        time.sleep(1.5)
        nav.ravenWrapper.raise_elevator()
        time.sleep(1.5)
        stack()
        # TODO: Remove can from collected cans list

    # Ran at the start of the game
    def store_zone_locations():
        # TODO: use camera to find and store zone locations
        return

    def stack():
        global can_in_center_pos, stacked_cans
        # Assume robot is gripping can
        nav.override_paths_world_xy(*offset_pos if can_in_center_pos else center_pos, use_claw=True)
        time.sleep(4)
        nav.ravenWrapper.lower_elevator()
        time.sleep(1)
        nav.ravenWrapper.open_gripper()
        time.sleep(1)
        nav.ravenWrapper.raise_elevator()
        time.sleep(1)
        nav.addPath(NavMove(-1, -1, 3000))
        time.sleep(1.5)
        if (stacked_cans > 0):
            nav.override_rotate_world_xy(*center_pos if can_in_center_pos else offset_pos)
            time.sleep(1)
            approach_can_with_ds()
            nav.ravenWrapper.lower_elevator()
            time.sleep(1)
            nav.ravenWrapper.close_gripper()
            time.sleep(0.3)
            nav.ravenWrapper.raise_elevator()
            time.sleep(0.5)
            nav.addPath(NavMove(-1, -1, 3000))
            time.sleep(2)
            nav.override_rotate_world_xy(*offset_pos if can_in_center_pos else center_pos)
            time.sleep(1)
            approach_can_with_ds()
            nav.ravenWrapper.open_gripper()

        can_in_center_pos = not can_in_center_pos
        stacked_cans += 1
        time.sleep(2)

    # Run when having a held can
    def score_held_can():
        if heldCanColor == "Red":
            goal_x = red_zone_x
            goal_y = red_zone_y
        elif heldCanColor == "Green":
            goal_x = green_zone_x
            goal_y = green_zone_y
        elif heldCanColor == "Yellow":
            goal_x = yellow_zone_x
            goal_y = yellow_zone_y
        else:
            return
        path_find(goal_x, goal_y)
        # Add can to obstacle list
        addCan(goal_x, goal_y)
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
    # if not camera.open():
    #     print(f"Failed to open camera: {args.camera}")
    #     return

    nav = Nav()
    # activate the navigation in another thread
    thread = threading.Thread(target=nav.startLoop, daemon=True)
    thread.start()

    # nav.ravenWrapper.lower_right_arm()
    # nav.ravenWrapper.open_gripper()
  # use 2438
    golden_x = 2438
    golden_y = 0
    # cx.append(2483)
    # cy.append(500)
    # cx.append(2483)
    # cy.append(400)
    # cx.append(2483)
    # cy.append(200)
    cx.append(2483)
    cy.append(100)
    cx.append(2483)
    cy.append(50)
    cx.append(golden_x)
    cy.append(golden_y)
    cx.append(2483)
    cy.append(-50)
    cx.append(2483)
    cy.append(-100)
    cx.append(2483)
    cy.append(-150)
    cx.append(2483)
    cy.append(-200)
    cx.append(2483)
    cy.append(-300)
    # cx.append(2483)
    # cy.append(-400)
    # cx.append(2483)
    # cy.append(-500)

    print(nav.ravenWrapper.get_odometry())

    # nav.addPath(NavMove(*get_rotate(math.pi/2)))
    # goto_golden_can()
    # time.sleep(4)
    # getCans()
    # go_to_closest_can()
    ox.append(1000)
    oy.append(0)
    path_find(3000, 0)
    # go_to_closest_can()
    # ccx.append(2483)
    # ccy.append(400)
    # go_to_closest_can()
    # stack()
    print(nav.ravenWrapper.get_odometry())
    print("DONE!")
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
