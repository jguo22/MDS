from robot import Robot
import time
from robot import RobotState

robot = Robot()
robot.state = RobotState.STEP_1

while True:
    if (robot.state == RobotState.STEP_1):
        robot.nav.startPath(1, 1, 40000) # rotate 90 degrees
        robot.state = RobotState.STEP_2
    robot.nav.updatePath(.05)
    time.sleep(.05)

while True:
    robot.setNowTime()

    match robot.state:
        case RobotState.STEP_1:
            print("step 1")
            robot.nav.startPath(0.97, 1.03, 24000)
            robot.state = RobotState.STEP_2
            robot.state_start = robot.now
            print("moving on from step 1")
        case RobotState.STEP_2:
            print("step 2")
            if (robot.now - robot.state_start > 4):
                print("moving on from step 2")
                robot.nav.startPath(1, 0, robot.nav.TICK_ROTATION / robot.nav.BASE_RATIO / 2.5) # rotate
                print("calling rotation")
                robot.state = RobotState.STEP_3
                robot.state_start = robot.now
        case RobotState.STEP_3:
            print("step 3")
            if (robot.now - robot.state_start > 1.4):
                print("moving on from step 3")
                robot.nav.startPath(1.1, .9, 17000) # Move down
                robot.state = RobotState.STEP_4
                robot.state_start = robot.now
        case RobotState.STEP_4:
            print("step 4")
            if (robot.now - robot.state_start > 3):
                print("moving on from step 4")
                robot.nav.startPath(1, -1, 30000) # SPIN
                robot.state = RobotState.STEP_5
                robot.state_start = robot.now

    robot.nav.updatePath(.05)
    time.sleep(.05) # TODO: Measure the dt and set the .05 properly
