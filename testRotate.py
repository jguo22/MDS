"""
Test script to rotate robot 30 degrees using nav.overridePaths()
Run this on the Pi to test rotation functionality.
"""

import math
import time
from threading import Thread
from nav import Nav, NavMove, get_rotate
from IMUWrapper import IMUWrapper


def main():
    print("Initializing IMU...")
    # CRITICAL: Initialize IMU before Raven board
    imu = IMUWrapper()

    print("Initializing Nav system...")
    nav = Nav(imu)

    Thread(target=nav.startLoop).start()

    while True:
        # Override paths with 30-degree rotation
        nav.overridePaths([
            NavMove(*get_rotate(math.pi / 100), smooth=False)
        ])
        time.sleep(0.03)


if __name__ == "__main__":
    main()
