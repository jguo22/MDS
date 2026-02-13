import time
import math
from distanceSensorWrapper import DistanceSensorWrapper
from nav import NavMove, Nav, get_rotate, get_forward_mm
from RavenWrapper import RAVEN_WRAPPER


center_pos = (1200, 0)
offset_pos = (1200, -400)
stacked_cans = 0
can_in_center_pos = True


class EarlyGame():
    def __init__(
            self,
            nav: Nav,
            distanceSensor: DistanceSensorWrapper,
            golden,
            left,
            right):
        self.nav: Nav = nav
        self.golden_x, self.golden_y = golden
        self.cx = [left[0], self.golden_x, right[0]]
        self.cy = [left[1], self.golden_y, right[1]]
        self.ccx = []
        self.ccy = []
        self.distance_sensor = distanceSensor

    def performEarlyGame(self):
        self.goto_golden_can()
        time.sleep(4)
        self.getCans()

    def goto_golden_can(self):
        RAVEN_WRAPPER.lower_left_arm()
        self.nav.override_paths_world_xy(self.golden_x, self.golden_y)

    def getCans(self):
        global right_cans, left_cans
        print("current pos is ", RAVEN_WRAPPER.get_odometry())
        # Collect right side cans first
        print("collecting right cans")
        # TODO: change these from override to add paths
        self.nav.override_paths_world_xy(self.cx[-1], self.cy[-1])
        print(f"going to rightmost can at {self.cx[-1]}, {self.cy[-1]}")
        time.sleep(2)
        # Deposit cans on our side
        self.nav.addPath(NavMove(1.3, 0.7, 6000, False, True))
        time.sleep(2)
        # TODO: STORE all can LOCATIONs from down facing camera
        x, y = self.nav.get_world_claw_position()
        self.ccx.append(x)
        self.ccy.append(y)
        self.nav.addPath(NavMove(-1.3, -0.7, 6000, False, True))
        time.sleep(1)
        RAVEN_WRAPPER.raise_left_arm()
        time.sleep(1)
        # Collect cans from other side
        print("collecting left cans now")
        self.nav.addPath(NavMove(*get_rotate(math.pi), False, False))
        time.sleep(1)
        RAVEN_WRAPPER.lower_right_arm()
        time.sleep(1)
        print(f"going to leftmost can at {self.cx[0]} {self.cy[0]}")
        self.nav.override_paths_world_xy(self.cx[0], self.cy[0])
        time.sleep(2)
        # Deposit cans on our side
        self.nav.addPath(NavMove(0.7, 1.3, 6000, False, True))
        # TODO: STORE all can LOCATIONS from down facing camera
        x, y = self.nav.get_world_claw_position()
        self.ccx.append(x)
        self.ccy.append(y)
        time.sleep(2)
        self.nav.addPath(NavMove(-0.7, -1.3, 6000, False, True))
        time.sleep(2)
        RAVEN_WRAPPER.raise_right_arm()
        self.nav.override_paths_world_xy(0, 0)
        time.sleep(5)
        # Next movement
        self.go_to_closest_can()
# All the cans have been collected on our side

    def go_to_closest_can(self):
        # Find closest can
        x, y = RAVEN_WRAPPER.get_odometry()
        min_distance = float('inf')
        closest_can_x, closest_can_y = None, None
        for i in range(len(self.ccx)):
            can_x = self.ccx[i]
            can_y = self.ccy[i]
            distance = ((can_x - x)**2 + (can_y - y)**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_can_x = can_x
                closest_can_y = can_y
        if closest_can_x and closest_can_y:
            self.nav.override_paths_world_xy(
                closest_can_x, closest_can_y)
            time.sleep(5)
            # Next movement
            self.grab_closest_can()

    def approach_can_with_ds(self):
        # Approach can with distance sensor
        while self.distance_sensor.get_distance() > 100:
            self.nav.overridePaths(
                [NavMove(*get_forward_mm(self.distance_sensor.get_distance() - 85))])
            time.sleep(1.2)


# Use camera to grab closest can


    def grab_closest_can(self):
        # TODO: Use down facing camera to get image of cans
        # TODO: get closest can, rotate towards it
        self.approach_can_with_ds()
        RAVEN_WRAPPER.lower_elevator()
        time.sleep(1.5)
        RAVEN_WRAPPER.close_gripper()
        time.sleep(1.5)
        RAVEN_WRAPPER.raise_elevator()
        time.sleep(1.5)
        self.stack()
        # TODO: Remove can from collected cans list

# Ran at the start of the game

    def store_zone_locations(self):
        # TODO: use camera to find and store zone locations
        return

    def stack(self):
        global can_in_center_pos, stacked_cans
        # Assume robot is gripping can
        self.nav.override_paths_world_xy(
            *offset_pos if can_in_center_pos else center_pos,
        )
        time.sleep(4)
        RAVEN_WRAPPER.lower_elevator()
        time.sleep(1)
        RAVEN_WRAPPER.open_gripper()
        time.sleep(1)
        RAVEN_WRAPPER.raise_elevator()
        time.sleep(1)
        self.nav.addPath(NavMove(-1, -1, 3000))
        time.sleep(1.5)
        if (stacked_cans > 0):
            self.nav.override_rotate_world_xy(
                *center_pos if can_in_center_pos else offset_pos)
            time.sleep(1)
            self.approach_can_with_ds()
            RAVEN_WRAPPER.lower_elevator()
            time.sleep(1)
            RAVEN_WRAPPER.close_gripper()
            time.sleep(0.3)
            RAVEN_WRAPPER.raise_elevator()
            time.sleep(0.5)
            self.nav.addPath(NavMove(-1, -1, 3000))
            time.sleep(2)
            self.nav.override_rotate_world_xy(
                *offset_pos if can_in_center_pos else center_pos)
            time.sleep(1)
            self.approach_can_with_ds()
            RAVEN_WRAPPER.open_gripper()

        can_in_center_pos = not can_in_center_pos
        stacked_cans += 1
        time.sleep(2)
