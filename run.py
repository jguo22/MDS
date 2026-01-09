from robot import Robot
import time
from robot import RobotState

robot = Robot()
robot.state = RobotState.STEP_1
# while True:
#     if (robot.state == RobotState.STEP_1):
#         robot.nav.startPath(1, 0, 1000) # rotate 90 degrees
#         robot.state = RobotState.STEP_2
#     robot.nav.updatePath(.05)
#     time.sleep(.05)

while True:
    robot.setNowTime()

    match robot.state:
        case RobotState.STEP_1:
            print("step 1")
            robot.nav.startPath(0.95, 1.05, 23000)
            robot.state = RobotState.STEP_2
            robot.state_start = robot.now
            print("moving on from step 1")
        case RobotState.STEP_2:
            print("step 2")
            if (robot.now - robot.state_start > 5):
                print("moving on from step 2")
                robot.nav.startPath(1, 0, robot.nav.TICK_ROTATION / robot.nav.BASE_RATIO / 2) # rotate 90 degrees
                print("calling rotation")
                robot.state = RobotState.STEP_3
                robot.state_start = robot.now
        case RobotState.STEP_3:
            print("step 3")
            if (robot.now - robot.state_start > 2):
                print("moving on from step 3")
                robot.nav.startPath(1.13, .87, 15000) # Move down
                robot.state = RobotState.DONE
                robot.state_start = robot.now

    robot.nav.updatePath(.05)
    time.sleep(.05) # TODO: Measure the dt and set the .05 properly
