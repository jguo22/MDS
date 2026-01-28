from DirectRobotCommander import DirectRobotCommander
from IMUWrapper import IMUWrapper
from nav import Nav, NavMove
from distanceSensorWrapper import DistanceSensorWrapper
import time
from navHelpers import get_forward_mm
from threading import Thread

imu = IMUWrapper()
nav = Nav(imu)
Thread(target=nav.startLoop).start()
ds = DistanceSensorWrapper()
directCommander = DirectRobotCommander(nav, ds, imu)


def approach_can_with_ds() -> bool:
    # Approach can with distance sensor

    while ds.get_distance() > 100:
        if ds.get_distance() > 800:
            return False
        nav.overridePaths(
            [NavMove(*get_forward_mm(ds.get_distance() - 85))])
        time.sleep(1.2)
    return True


directCommander.pickup_can()
approach_can_with_ds()
directCommander.release_can()
