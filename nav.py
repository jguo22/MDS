import math
import time
import threading
from vision.relativeCoordinates import world_to_relative
from spatialmath import SE2
from navHelpers import get_forward_mm, get_rotate
from config import CLAW_OFFSET, TICK_ROTATION, NAV_FRAME_TIME, ANGLE_D, ANGLE_PROP, TURN_CONSTANT
from RavenWrapper import RAVEN_WRAPPER, LEFT_MOTOR, RIGHT_MOTOR
from IMUWrapper import IMUWrapper


# Miguel's Navigation movement class
class NavMove:
    def __init__(
            self,
            left: float,
            right: float,
            dist: float,
            smooth: bool = False,
            correct_angle: bool = True):
        self.left = left
        self.right = right
        self.dist = dist
        self.smooth = smooth
        self.correct_angle = correct_angle
        self.lookingDown = False


class Nav:
    def __init__(self, imuWrapper: IMUWrapper):
        # I put it here so that it doesn't run on computer
        # IMU SETUP MUST BE BEFORE RAVEN SETUP
        self.imu_wrapper = imuWrapper

        self.max_velocity = 2.8 * TICK_ROTATION  # ticks/s
        self.acceleration = 3.0 * TICK_ROTATION  # ticks/s^2. Reach max v in 1s

        self.moves: list[NavMove] = []
        self._lock = threading.Lock()
        self.moving = False
        self.correct_angle = True

        # for imu
        self.last_angle = 0  # doesn't accumulate
        self.angle = 0  # accumulates
        self.diff_angle = 0

        # for path
        self.start_angle = 0
        self.start_left = 0
        self.start_right = 0

        self.total_distance = 0

        self.left_coef = 0
        self.right_coef = 0
        self.last_speed = 0
        self.current_distance = 0

        self._updateAngle()
        self.start_angle = self.angle

    def startLoop(self):
        # ONLY RUN ONE LOOP
        start_time = time.time()
        try:
            while True:
                # get delta_time and sleep
                delta_time = time.time() - start_time
                if (delta_time < NAV_FRAME_TIME):
                    time.sleep(NAV_FRAME_TIME - delta_time)
                    delta_time = NAV_FRAME_TIME
                start_time = time.time()

                # print(f'x, y is {self.raven.get_odometry()}')
                # print(f'angle is {self.raven.get_angle()}')

                self._updatePath(delta_time)
        except KeyboardInterrupt:
            self.imu_wrapper.hard_reset()
            print("keyboard interrupt")

    def addPath(self, nav_move: NavMove):
        # append is thread safe, but use the lock so that you don't edit it
        # while update path is using it
        with self._lock:
            self.moves.append(nav_move)

    def overridePaths(self, nav_moves: list[NavMove]):
        print(nav_moves)
        # copy it to not modify original
        nav_moves = nav_moves[:]
        # immediately use startPath to override current path
        with self._lock:
            if len(nav_moves) > 0:
                self._startPath(nav_moves.pop(0))
                self.moving = True
            else:
                self.moving = False
            self.moves = nav_moves

    def _startPath(self, nav_move: NavMove):
        self.total_distance = nav_move.dist
        self.current_distance = 0
        self.left_coef = -nav_move.left
        self.right_coef = nav_move.right
        self._updateAngle()
        self.start_angle = self.angle
        self.start_left = RAVEN_WRAPPER.get_motor_encoder(LEFT_MOTOR)
        self.start_right = RAVEN_WRAPPER.get_motor_encoder(RIGHT_MOTOR)

    def _updateAngle(self):
        current_angle = self.imu_wrapper.get_heading()  # -pi to pi

        # set angle for odometry
        RAVEN_WRAPPER.set_angle(current_angle)

        self.diff_angle = current_angle - self.last_angle
        if self.diff_angle > math.pi:
            self.diff_angle -= 2 * math.pi
        elif self.diff_angle < -math.pi:
            self.diff_angle += 2 * math.pi
        # diff angle is the change clamped to range of -pi to pi
        self.angle += self.diff_angle  # accumulates
        self.last_angle = current_angle  # range of -pi to pi

    def _updatePath(self, dt):
        with self._lock:
            # if we're not moving, start next move
            if not self.moving and len(self.moves) > 0:
                self.moving = True
                self.correct_angle = self.moves[0].correct_angle
                self._startPath(self.moves[0])
                self.moves.pop(0)

            # calculate target speed
            distance_left = self.total_distance - self.current_distance
            delta_speed = self.acceleration * dt
            target_speed = 0
            # check whether we have to slow down or not
            if (len(self.moves) == 0 or not self.moves[0].smooth) and (
                    distance_left <= self.last_speed ** 2 / (2 * self.acceleration)):
                target_speed = max(self.last_speed - delta_speed, 0)
            else:
                target_speed = min(
                    self.last_speed + delta_speed,
                    self.max_velocity)

            # average for more accuracy
            self.current_distance += (self.last_speed + target_speed) / 2 * dt
            self.last_speed = target_speed

            # calculate angle error and correct it
            target_angle = (self.right_coef + self.left_coef) / \
                2.0 * TURN_CONSTANT * self.current_distance

            self._updateAngle()

            # print(self.correct_angle)

            if self.correct_angle:
                angle_error = (self.angle - self.start_angle) - target_angle
                # if (self.moving):
                #     print(f'angle error: {angle_error}')

                self.start_left -= angle_error * ANGLE_PROP * dt - self.diff_angle * ANGLE_D * dt
                self.start_right -= angle_error * ANGLE_PROP * \
                    dt - self.diff_angle * ANGLE_D * dt

            target_left = self.start_left + \
                (self.current_distance * self.left_coef)
            target_right = self.start_right + \
                (self.current_distance * self.right_coef)
            RAVEN_WRAPPER.set_motor_target(LEFT_MOTOR, target_left)
            RAVEN_WRAPPER.set_motor_target(RIGHT_MOTOR, target_right)

            # move on to next thing if its done
            if distance_left <= 0:
                if len(self.moves) == 0:
                    self.moving = False
                else:
                    self.moving = True
                    nav_move = self.moves.pop(0)
                    self.total_distance = nav_move.dist
                    self.current_distance = 0
                    self.left_coef = -nav_move.left
                    self.right_coef = nav_move.right
                    self._updateAngle()
                    self.start_angle += target_angle
                    self.start_left = target_left
                    self.start_right = target_right

    def get_relative_position(self, world_x: float,
                              world_y: float) -> tuple[float, float]:
        """
        Convert world coordinates to robot-relative coordinates.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)

        Returns:
            (x_rel, y_rel): Position relative to robot
                x_rel: Forward distance (positive = ahead)
                y_rel: Lateral distance (positive = left)
        """
        # Get robot's current world position and orientation
        robot_x, robot_y = RAVEN_WRAPPER.get_odometry()
        robot_theta = self.angle

        # Use the centralized coordinate transformation function
        return world_to_relative(
            (world_x, world_y), SE2(robot_x, robot_y, robot_theta))

    def override_paths_world_xy(self, world_x, world_y):
        """
        Navigate to a world coordinate (x, y) by calculating rotation and forward movement.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)
        """
        x, y = self.get_relative_position(world_x, world_y)

        # Calculate angle to rotate to face the target
        # atan2(y, x) gives the angle from robot's forward axis to the target
        target_angle = math.atan2(y, x)

        # Calculate distance to target
        target_distance = math.sqrt(x**2 + y**2)

        # Create movement path: rotate, then forward
        movements = []

        movements.append(NavMove(*get_rotate(target_angle), False))

        movements.append(NavMove(*get_forward_mm(target_distance), False))

        # Override current path with new movements
        self.overridePaths(movements)

    def get_world_claw_position(self):
        """
        Get the position of the claw in world coordinates.
        """
        # Get robot's current world position
        # NOTE: odometry uses ROS coordinates
        # and get_heading uses theta=0 as forward
        robot_x, robot_y = RAVEN_WRAPPER.get_odometry()

        # Get claw offset in robot frame

        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)

        x_world = robot_x + CLAW_OFFSET * cos_angle
        y_world = robot_y + CLAW_OFFSET * sin_angle

        return x_world, y_world

    def override_rotate_world_xy(self, world_x, world_y):
        """
        Navigate to a world coordinate (x, y) by calculating rotation and forward movement.

        Args:
            world_x: Target x position in world frame (mm)
            world_y: Target y position in world frame (mm)
        """
        x, y = self.get_relative_position(world_x, world_y)

        # Calculate angle to rotate to face the target
        # atan2(y, x) gives the angle from robot's forward axis to the target
        target_angle = math.atan2(y, x)

        # Calculate distance to target
        target_distance = math.sqrt(x**2 + y**2)

        # Create movement path: rotate, then forward
        movements = []

        movements.append(NavMove(*get_rotate(target_angle), False))

        # Override current path with new movements
        self.overridePaths(movements)
