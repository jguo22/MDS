import math
import time
import threading
from typing import Tuple
from raven import Raven

import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR


# Miguel's Navigation movement class
LEFT_MOTOR = Raven.MotorChannel.CH2
RIGHT_MOTOR = Raven.MotorChannel.CH3
TICK_ROTATION = 64 * 50
WHEEL_D = 95  # TODO: measure in mm
BASE_D = 209
ANGLE_PROP = 5000
ANGLE_D = 5000

BASE_RATIO = WHEEL_D / BASE_D
TURN_CONSTANT = BASE_RATIO * 2 * math.pi / TICK_ROTATION

FRAME_TIME = 0.05  # 1/FPS


class NavMove:
    def __init__(self, left: float, right: float, dist: float, smooth: bool):
        self.left = left
        self.right = right
        self.dist = dist
        self.smooth = smooth


class Nav:
    def __init__(self):
        self.max_velocity = 10.0 * TICK_ROTATION  # ticks/s
        self.acceleration = 2.0 * TICK_ROTATION  # ticks/s^2. Reach max v in 1s

        self.moves: list[NavMove] = []
        self.moving = False

        # for imu
        self.last_angle = 0
        self.angle = 0
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

        self._lock = threading.Lock()

        self.raven = Raven()

        # Let IMU Setup
        i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
        self.bno = BNO08X_I2C(i2c)
        self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        for i in range(5):
            print(self.bno.quaternion)
            time.sleep(0.02)

        for motor in [LEFT_MOTOR, RIGHT_MOTOR]:
            self.raven.set_motor_encoder(motor, 0)
            self.raven.set_motor_max_current(motor, 5)
            self.raven.set_motor_mode(motor, Raven.MotorMode.POSITION)
            self.raven.set_motor_target(motor, 0)

        self.raven.set_motor_pid(RIGHT_MOTOR, p_gain=25, i_gain=5, d_gain=0.13)
        self.raven.set_motor_pid(LEFT_MOTOR, p_gain=20, i_gain=5, d_gain=0.1)

    def startLoop(self):
        # ONLY RUN ONE LOOP
        start_time = time.time()
        while True:
            # get delta_time and sleep
            delta_time = time.time() - start_time
            if (delta_time < FRAME_TIME):
                time.sleep(FRAME_TIME - delta_time)
                delta_time = FRAME_TIME
            start_time = time.time()

            self._updatePath(delta_time)

    def addPath(self, nav_move: NavMove):
        # append is thread safe, but use the lock so that you don't edit it
        # while update path is using it
        with self._lock:
            self.moves.append(nav_move)

    def overridePaths(self, nav_moves: list[NavMove]):
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
        self.start_left = self.raven.get_motor_encoder(LEFT_MOTOR)
        self.start_right = self.raven.get_motor_encoder(RIGHT_MOTOR)

    def _updateAngle(self):
        quat_i, quat_j, quat_k, quat_real = self.bno.quaternion
        current_angle = self.find_heading(quat_real, quat_i, quat_j, quat_k)
        self.diff_angle = current_angle - self.last_angle
        if self.diff_angle > math.pi:
            self.diff_angle -= 2 * math.pi
        elif self.diff_angle < -math.pi:
            self.diff_angle += 2 * math.pi
        self.angle += self.diff_angle
        self.last_angle = current_angle

    def _updatePath(self, dt):
        with self._lock:
            # if we're not moving, start next move
            if not self.moving and len(self.moves) > 0:
                self.moving = True
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
            angle_error = (self.angle - self.start_angle) - target_angle

            self.start_left -= angle_error * ANGLE_PROP * dt - self.diff_angle * ANGLE_D * dt
            self.start_right -= angle_error * ANGLE_PROP * \
                dt - self.diff_angle * ANGLE_D * dt

            target_left = self.start_left + \
                (self.current_distance * self.left_coef)
            target_right = self.start_right + \
                (self.current_distance * self.right_coef)
            self.raven.set_motor_target(LEFT_MOTOR, target_left)
            self.raven.set_motor_target(RIGHT_MOTOR, target_right)

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

    def find_heading(self, dqw, dqx, dqy, dqz):
        # normalize quaternion
        norm = math.sqrt(dqw * dqw + dqx * dqx + dqy * dqy + dqz * dqz)
        if (norm == 0.0):
            return self.angle
        dqw = dqw / norm
        dqx = dqx / norm
        dqy = dqy / norm
        dqz = dqz / norm

        ysqr = dqy * dqy

        t3 = +2.0 * (dqw * dqz + dqx * dqy)
        t4 = +1.0 - 2.0 * (ysqr + dqz * dqz)
        yaw_raw = math.atan2(t3, t4)
        return yaw_raw


def get_forward_mm(distance_mm: float) -> Tuple[float, float, float]:
    distance = distance_mm / (WHEEL_D * math.pi) * TICK_ROTATION
    return (1, 1, distance)


def get_rotate(theta: float) -> Tuple[float, float, float]:
    # make into range -pi to pi
    theta = theta % (2 * math.pi)
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < -math.pi:
        theta += 2 * math.pi

    # CCW is positive angle
    if theta >= 0:
        return (-1, 1, TICK_ROTATION / BASE_RATIO * theta / (2 * math.pi))
    else:
        return (1, -1, TICK_ROTATION / BASE_RATIO * -theta / (2 * math.pi))
