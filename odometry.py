import numpy as np
import math


class WheelOdometry:
    def __init__(
        self,
        wheel_diameter: float,
        track_width: float,
        cpr: int,
        left_encoder_init: int = 0,
        right_encoder_init: int = 0,
    ):
        self.reset()
        # Robot properties
        self.__WHEEL_DIAMETER = wheel_diameter
        self.__TRACK_WIDTH = track_width
        # distance per count, in units of wheel diameter
        self.__WHEEL_MOTOR_MPC = self.__WHEEL_DIAMETER * np.pi / cpr
        self.__left_encoder = left_encoder_init
        self.__right_encoder = right_encoder_init

    def reset(self):
        self.__x = 0.0
        self.__y = 0.0
        self.__theta = 0.0

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def theta(self) -> float:
        return self.__theta

    def __repr__(self):
        return f"x: {self.x}\ny: {self.y}\nheading: {self.theta * 180/np.pi} degree"

    def update(self, left_encoder: int, right_encoder: int):
        # Get encoder change
        d_left_encoder = left_encoder - self.__left_encoder
        d_right_encoder = right_encoder - self.__right_encoder

        # Update encoder values
        self.__left_encoder = left_encoder
        self.__right_encoder = right_encoder

        # Get distance change
        dL = d_left_encoder * self.__WHEEL_MOTOR_MPC
        dR = d_right_encoder * self.__WHEEL_MOTOR_MPC

        # Moves in an arc
        if d_left_encoder != d_right_encoder:
            r = self.__TRACK_WIDTH / 2
            arc_radius = r * (dL + dR) / abs(dR - dL)
            d_theta = max(dL, dR) / (arc_radius + r)
            d_x = 0
            d_y = 0
        else:
            d_theta = 0
            d_x = 0
            d_y = 0

        # Update reading
        self.__theta = (self.__theta + d_theta + np.pi) % (
            2 * np.pi
        ) - np.pi  # Wrapping to 2*pi
        self.__x += d_x
        self.__y += d_y
