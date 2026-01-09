from robot import Robot
import time

robot = Robot()

robot.startPath(0.96, 1.04, 23000)
time.sleep(3)
robot.startPath(1, 0, robot.TICK_ROTATION / robot.BASE_RATIO /2) # rotate 90 degrees
time.sleep(3)
robot.startPath(1.15, .85, 18000) # Move down
time.sleep(3)

while True:
    robot.updatePath(.05)
    robot.time.sleep(.05) # TODO: Measure the dt and set the .05 properly
