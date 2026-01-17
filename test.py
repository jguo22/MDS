from raven import Raven
from nav import *

raven = Raven()

raven = Raven()
raven.set_base(WHEEL_D, BASE_D)

for motor in [LEFT_MOTOR, RIGHT_MOTOR]:
    raven.set_motor_encoder(motor, 0)
    raven.set_motor_max_current(motor, 5)
    raven.set_motor_mode(motor, Raven.MotorMode.POSITION)
    raven.set_motor_target(motor, 0)

raven.set_motor_pid(RIGHT_MOTOR, p_gain=25, i_gain=5, d_gain=0.13)
raven.set_motor_pid(LEFT_MOTOR, p_gain=25, i_gain=5, d_gain=0.13)

while True:
    print(raven.get_odometry())
    print(raven.get_angle())
    print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH2))
    print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH3))
    raven.set_motor_target(LEFT_MOTOR, -1000)
    raven.set_motor_target(RIGHT_MOTOR, 1000)

    time.sleep(0.5)

    print(raven.get_odometry())
    print(raven.get_angle())
    print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH2))
    print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH3))
    raven.set_motor_target(LEFT_MOTOR, 1000)
    raven.set_motor_target(RIGHT_MOTOR, -1000)
    time.sleep(0.5)
