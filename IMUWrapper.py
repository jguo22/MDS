import math
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR


class IMUWrapper():
    def __init__(self):
        # Let IMU Setup
        i2c = busio.I2C(board.SCL, board.SDA, frequency=800000)
        self.bno = BNO08X_I2C(i2c)
        self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

        self._offset = self._get_internal_heading()

    def _calculate_heading(self, dqw, dqx, dqy, dqz):
        # normalize quaternion
        norm = math.sqrt(dqw * dqw + dqx * dqx + dqy * dqy + dqz * dqz)
        if (norm == 0.0):
            return 0
        dqw = dqw / norm
        dqx = dqx / norm
        dqy = dqy / norm
        dqz = dqz / norm

        ysqr = dqy * dqy

        t3 = +2.0 * (dqw * dqz + dqx * dqy)
        t4 = +1.0 - 2.0 * (ysqr + dqz * dqz)
        yaw_raw = math.atan2(t3, t4)
        return yaw_raw

    def _get_internal_heading(self):
        quat_i, quat_j, quat_k, quat_real = self.bno.quaternion
        return self.calculate_heading(quat_real, quat_i, quat_j, quat_k)

    def get_heading(self):
        return self._get_internal_heading() - self._offset

    def hard_reset(self):
        return self.bno.hard_reset()
