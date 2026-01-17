from raven import Raven
import time
from nav import *

raven = Raven()
raven.set_base(WHEEL_D, BASE_D)
print(raven.get_base())

while True:
    print(raven.get_odometry())
    print(raven.get_angle())
    print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH2))
    print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH3))
    time.sleep(0.5)
 