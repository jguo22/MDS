import math

from raven import Raven
import math
from queue import Queue

# Miguel's Navigation movement class
LEFT_MOTOR = Raven.MotorChannel.CH2
RIGHT_MOTOR = Raven.MotorChannel.CH3
TICK_ROTATION = 64 * 50
WHEEL_D = 95
BASE_D = 209
ANGLE_PROP = 5000
ANGLE_D = 5000

BASE_RATIO = WHEEL_D/BASE_D
TURN_CONSTANT = BASE_RATIO * 2 * math.pi/TICK_ROTATION

class NavMove:
    def __init__(self, left, right, dist, smooth):
        self.left = left
        self.right = right
        self.dist = dist
        self.smooth = smooth

class Nav:
    def get(self):
        return TICK_ROTATION / BASE_RATIO

    def __init__(self, bno):
        self.max_velocity = 10.0 * TICK_ROTATION # ticks/s
        self.acceleration = 2.0 * TICK_ROTATION # ticks/s^2. Reach max v in 1s

        self.moves = []
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

        self.raven = Raven()
        self.bno = bno

        for motor in [LEFT_MOTOR, RIGHT_MOTOR]:
            self.raven.set_motor_encoder(motor, 0)
            self.raven.set_motor_max_current(motor, 5)
            self.raven.set_motor_mode(motor, Raven.MotorMode.POSITION)
            self.raven.set_motor_target(motor, 0)

        self.raven.set_motor_pid(RIGHT_MOTOR, p_gain = 25, i_gain = 5, d_gain = 0.13)
        self.raven.set_motor_pid(LEFT_MOTOR, p_gain = 20, i_gain = 5, d_gain = 0.1)

    def addPath(self, nav_move):
        self.moves.append(nav_move)

    def overridePath(self, nav_move):
        self.moves = []
        self._startPath(nav_move)
        self.moving = True

    def _startPath(self, nav_move):
        self.total_distance = nav_move.dist
        self.current_distance = 0
        self.left_coef = -nav_move.left
        self.right_coef = nav_move.right
        self.updateAngle()
        self.start_angle = self.angle
        self.start_left = self.raven.get_motor_encoder(LEFT_MOTOR)
        self.start_right = self.raven.get_motor_encoder(RIGHT_MOTOR)

    def updateAngle(self):
        quat_i, quat_j, quat_k, quat_real = self.bno.quaternion
        current_angle = self.find_heading(quat_real, quat_i, quat_j, quat_k)
        self.diff_angle = current_angle - self.last_angle
        if self.diff_angle > math.pi:
            self.diff_angle -= 2*math.pi
        elif self.diff_angle < -math.pi:
            self.diff_angle += 2*math.pi
        self.angle += self.diff_angle
        self.last_angle = current_angle


    def updatePath(self, dt):
        if not self.moving and len(self.moves) > 0:
            self.moving = True
            self._startPath(self.moves[0])
            self.moves.pop(0)

        delta_speed = self.acceleration * dt
        target_speed = 0
        distance_left = self.total_distance - self.current_distance
        if (len(self.moves) == 0 or not self.moves[0].smooth) and (distance_left <= self.last_speed ** 2 / ( 2 * self.acceleration )):
            target_speed = max(self.last_speed - delta_speed, 0)
        else:
            target_speed = min(self.last_speed + delta_speed, self.max_velocity)

        self.current_distance += (self.last_speed + target_speed)/2*dt
        self.last_speed = target_speed

        target_angle = (self.right_coef + self.left_coef)/2.0 * TURN_CONSTANT * self.current_distance

        self.updateAngle()
        angle_error = (self.angle - self.start_angle) - target_angle

        self.start_left -= angle_error * ANGLE_PROP * dt - self.diff_angle * ANGLE_D * dt
        self.start_right -= angle_error * ANGLE_PROP * dt - self.diff_angle * ANGLE_D * dt

        target_left = self.start_left + (self.current_distance * self.left_coef)
        target_right = self.start_right + (self.current_distance * self.right_coef)
        self.raven.set_motor_target(LEFT_MOTOR, target_left)
        self.raven.set_motor_target(RIGHT_MOTOR, target_right)

        if distance_left <= 0:
            if len(self.moves) == 0:
                self.moving = False
            else:
                self.moving = True
                nav_move = self.moves[0]
                self.total_distance = nav_move.dist
                self.current_distance = 0
                self.left_coef = -nav_move.left
                self.right_coef = nav_move.right
                self.updateAngle()
                self.start_angle += target_angle
                self.start_left = target_left
                self.start_right = target_right
                self.moves.pop(0)


    def find_heading(self, dqw, dqx, dqy, dqz):
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
