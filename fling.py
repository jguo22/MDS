from RavenWrapper import RAVEN_WRAPPER
from nav import Nav, NavMove
import time
import argparse
from IMUWrapper import IMUWrapper
from IRobotCommander import IRobotCommander
from RavenWrapper import RAVEN_WRAPPER
from distanceSensorWrapper import DistanceSensorWrapper
from nav import Nav, NavMove
import threading
import traceback
import config
from connection import message_types
from connection.PiStreamer import PiStreamer
from connection.CameraCapture import CameraCapture
from connection import command_tracker
from DirectRobotCommander import DirectRobotCommander

# Initialize IMU (must be first!)
imu_wrapper = IMUWrapper()

# Initialize distance sensor
distance_sensor = DistanceSensorWrapper()

# Initialize navigation
nav = Nav(imu_wrapper)

# Start navigation loop in background thread
nav_thread = threading.Thread(target=nav.startLoop, daemon=True)
nav_thread.start()

# Create direct robot commander for command execution
robot_commander = DirectRobotCommander(nav, distance_sensor, imu_wrapper)

RAVEN_WRAPPER.raise_arms()
# nav.addPath(NavMove(1, 1, 10000))
# time.sleep(2)
# RAVEN_WRAPPER.lower_left_arm()
# nav.addPath(NavMove(-2, 2, 1500))
# nav.addPath(NavMove(1, 1, 1000))
# RAVEN_WRAPPER.raise_arms()
# nav.overridePaths([NavMove(-2, 2, 1500)])
# time.sleep(2)
