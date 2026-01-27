import threading
from typing import Optional, Tuple
from raven import Raven
from config import WHEEL_D, BASE_D
import time

LEFT_MOTOR = Raven.MotorChannel.CH2
RIGHT_MOTOR = Raven.MotorChannel.CH3


CAMERA_SERVO = Raven.ServoChannel.CH1


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
        self._camera_angle: float = 0.0

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
    def get_odometry(self) -> Optional[Tuple[float, float]]:
        """Get robot position (x, y) in mm."""
        with self._lock:
            return self.raven.get_odometry()

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

    # Camera Angle Methods
    def get_camera_angle(self) -> float:
        """Get camera servo angle in radians."""
        with self._lock:
            return self._camera_angle

    def set_camera_angle(self, angle: float) -> bool:
        """Set camera servo angle in radians."""
        import math
        with self._lock:
            self._camera_angle = angle
            degree = math.degrees(angle)
            return self.raven.set_servo_position(CAMERA_SERVO, degree)

    # Arm Methods
    def lower_left_arm(self) -> bool:
        """Lower left arm."""
        with self._lock:
            return self.raven.set_servo_position(
                Raven.ServoChannel.CH2, -90, 930, 1910)
    def raise_left_arm(self) -> bool:
        """Raise left arm."""
        with self._lock:
            return self.raven.set_servo_position(
                Raven.ServoChannel.CH2, 90, 930, 1910)
    def lower_right_arm(self) -> bool:
        """Lower right arm."""
        with self._lock:
            return self.raven.set_servo_position(
                Raven.ServoChannel.CH1, 90, 1000, 1970)
    def raise_right_arm(self) -> bool:
        """Raise right arm."""
        with self._lock:
            return self.raven.set_servo_position(
                Raven.ServoChannel.CH1, -90, 1000, 1970)

    # Gripper Methods
    def open_gripper(self) -> bool:
        """Open gripper."""
        with self._lock:
            return self.raven.set_servo_position(
                Raven.ServoChannel.CH4, 0)
    def close_gripper(self) -> bool:
        """Close gripper."""
        with self._lock:
            return self.raven.set_servo_position(
                Raven.ServoChannel.CH4, 67)

    # Elevator Methods
    def raise_elevator(self):
        """Raise elevator."""
        with self._lock:
            self.raven.set_servo_position(
                Raven.ServoChannel.CH3, 90)
            time.sleep(1.2)
            self.raven.set_servo_position(
                Raven.ServoChannel.CH3, 0)
    def lower_elevator(self):
        """Lower elevator."""
        with self._lock:
            self.raven.set_servo_position(
                Raven.ServoChannel.CH3, -90)
            time.sleep(1.2)
            self.raven.set_servo_position(
                Raven.ServoChannel.CH3, 0)



ravenWrapper = RavenWrapper()
