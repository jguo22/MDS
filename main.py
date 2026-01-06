from raven import Raven

raven_board = Raven()

raven_board.set_motor_encoder(Raven.MotorChannel.CH1, 0) # Set encoder count for motor 1 to zero
print(raven_board.get_motor_encoder(Raven.MotorChannel.CH1)) # Print encoder count = "0"

raven_board.set_motor_mode(Raven.MotorChannel.CH1, Raven.MotorMode.DIRECT) # Set motor mode to DIRECT

# Speed controlled:
raven_board.set_motor_torque_factor(Raven.MotorChannel.CH1, 100) # Let the motor use all the torque to get to speed factor
raven_board.set_motor_speed_factor(Raven.MotorChannel.CH1, 100, reverse=True) # Spin at 10% max speed in reverse

# Torque controlled:
# raven_board.set_motor_speed_factor(Raven.MotorChannel.CH1, 100) # Make motor try to run at max speed forward
# raven_board.set_motor_torque_factor(Raven.MotorChannel.CH1, 100) # Let it use up to 10% available torque
while True:
    pass

# hi junhui