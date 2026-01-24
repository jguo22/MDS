import time
from RavenWrapper import RAVEN_WRAPPER, CAMERA_SERVO, ELEVATOR_SERVO, GRIPPER_SERVO


def moveGripperHeight(height):
    pass


def gripClaw():
    RAVEN_WRAPPER.set_servo_position(GRIPPER_SERVO, 90)


def releaseClaw():
    RAVEN_WRAPPER.set_servo_position(GRIPPER_SERVO, -90)


def releaseGrip():
    RAVEN_WRAPPER.set_servo_position(CAMERA_SERVO, -90)
    if heldCanColor != "None":
        claw_x, claw_y = nav.get_world_claw_position()
        ox.append(claw_x)
        oy.append(claw_y)
    heldCanColor = "None"


def gripCan():
    moveElevatorDown()
    getCanColor()
    gripClaw()
    time.sleep(0.5)
    moveElevatorUp()


def moveElevatorUp():
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 90)
    time.sleep(1.2)
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 0)


def moveElevatorDown():
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, -90)
    time.sleep(1.2)
    RAVEN_WRAPPER.set_servo_position(ELEVATOR_SERVO, 0)


def rotateCameraDown():
    RAVEN_WRAPPER.set_servo_position(CAMERA_SERVO, -90)


def rotateCameraUp():
    RAVEN_WRAPPER.set_servo_position(CAMERA_SERVO, 30)


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


def path_find(goal_x, goal_y):
    theta_star = ThetaStarPlanner(ox, oy, grid_size, robot_radius)
    robot_x, robot_y = RAVEN_WRAPPER.get_odometry()
    rx, ry = theta_star.planning(robot_x, robot_y, goal_x, goal_y)
    for (x, y) in zip(rx, ry):
        nav.add_paths_world_xy(x, y)


def goto_right_cans():
    path_find(goto_right_cans_x, goto_right_cans_y)
    nav.override_paths_world_xy(right_cans_x, right_cans_y, use_claw=True)
    gripCan()


def set_camera_angle(self, degree: float) -> bool:
    """Set camera servo angle in degrees"""
    return RAVEN_WRAPPER.set_servo_position(CAMERA_SERVO, degree)
