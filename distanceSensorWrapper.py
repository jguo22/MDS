import busio
import board
import adafruit_vl53l0x


class DistanceSensorWrapper():
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.distance_sensor = adafruit_vl53l0x.VL53L0X(i2c)

    def get_distance(self) -> int:
        return self.distance_sensor.range
