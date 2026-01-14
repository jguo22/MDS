from robot import Robot
import time
from robot import RobotState

import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR


# Let IMU Setup
i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
bno = BNO08X_I2C(i2c)
bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
for i in range(5):
    print(bno.quaternion)
    time.sleep(0.02)
robot = Robot(bno)

robot.state = RobotState.STEP_1
while True:
    if (robot.state == RobotState.STEP_1):
        if (robot.fetchGoldenPringleCan()):
            robot.state = RobotState.STEP_2
        # robot.nav.addPath(NavMove(1, 0, robot.nav.get() / 4, True))
        # robot.nav.addPath(NavMove(1, 1, 20000, False))
        # robot.nav.addPath(NavMove(1, 0.5, robot.nav.get() / 1, True))
        # robot.nav.addPath(NavMove(0.5, 1, robot.nav.get() / 1, True))
        # robot.nav.addPath(NavMove(1, 1, robot.nav.get() / 1, True))

        # robot.state = RobotState.STEP_2
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
