from raven import Raven
import time

raven = Raven()

while True:
	print(raven.get_odometry())
	print(raven.get_angle())
	print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH2))
	print(raven.get_motor_encoder(motor_channel=Raven.MotorChannel.CH3))
	# raven.set_angle(0)
	time.sleep(0.5)
