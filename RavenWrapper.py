import threading
from typing import Optional, Tuple
from raven import Raven
from config import WHEEL_D, BASE_D

LEFT_MOTOR = Raven.MotorChannel.CH2
RIGHT_MOTOR = Raven.MotorChannel.CH3

CAMERA_SERVO = Raven.ServoChannel.CH2
GRIPPER_SERVO = Raven.ServoChannel.CH1
ELEVATOR_SERVO = Raven.ServoChannel.CH4


class RavenWrapper():
    """
    Thread-safe wrapper for Raven motor controller.
    Provides synchronized access to all Raven methods to prevent race conditions.
    """

    def __init__(self):
        self.raven = Raven()
        self.raven.set_base(WHEEL_D, BASE_D)

        for motor in [LEFT_MOTOR, RIGHT_MOTOR]:
            self.raven.set_motor_encoder(motor, 0)
            self.raven.set_motor_max_current(motor, 5)
            self.raven.set_motor_mode(motor, Raven.MotorMode.POSITION)
            self.raven.set_motor_target(motor, 0)

        self.raven.set_motor_pid(RIGHT_MOTOR, p_gain=25, i_gain=5, d_gain=0.13)
        self.raven.set_motor_pid(LEFT_MOTOR, p_gain=20, i_gain=5, d_gain=0.1)

        self._lock = threading.Lock()

    # Motor Encoder Methods
    def get_motor_encoder(
            self,
            motor_channel: Raven.MotorChannel) -> Optional[float]:
        """Get motor encoder count."""
        with self._lock:
            return self.raven.get_motor_encoder(motor_channel)

    def set_motor_encoder(
            self,
            motor_channel: Raven.MotorChannel,
            value: int) -> bool:
        """Set motor encoder count."""
        with self._lock:
            return self.raven.set_motor_encoder(motor_channel, value)

    # Motor Control Methods
    def set_motor_target(
            self,
            motor_channel: Raven.MotorChannel,
            value: float) -> bool:
        """Set motor target (position or velocity depending on mode)."""
        with self._lock:
            return self.raven.set_motor_target(motor_channel, value)

    # Odometry Methods
    def get_odometry(self) -> Tuple[float, float]:
        """Get robot position (x, y) in mm."""
        with self._lock:
            result = self.raven.get_odometry()
            if result is None:
                raise Exception("ERROR GETTING ODOMETRY")
            else:
                return result

    def set_odometry(self, x: float, y: float) -> bool:
        """Set robot position (x, y) in mm."""
        with self._lock:
            return self.raven.set_odometry(x, y)

    # Angle Methods
    def get_angle(self) -> Optional[float]:
        """Get robot heading in radians."""
        with self._lock:
            return self.raven.get_angle()

    def set_angle(self, angle: float) -> bool:
        """Set robot heading in radians."""
        with self._lock:
            return self.raven.set_angle(angle)

    # Servo Methods
    def set_servo_position(
            self,
            servo_channel: Raven.ServoChannel,
            degree: float,
            min_us=1000,
            max_us=2000,
            retry=0,
    ):
        """Set servo position."""
        with self._lock:
            return self.raven.set_servo_position(
                servo_channel, degree, min_us, max_us, retry)


RAVEN_WRAPPER = RavenWrapper()
