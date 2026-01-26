from RavenWrapper import RAVEN_WRAPPER, RavenWrapper
from raven import Raven
import time


def main():
    for channel in [
            Raven.ServoChannel.CH1,
            Raven.ServoChannel.CH2,
            Raven.ServoChannel.CH3,
            Raven.ServoChannel.CH4
    ]:
        print(RAVEN_WRAPPER.raven.get_servo_position(channel))

    # try:
    #     setElevatorSpeed(90)
    #     time.sleep(1)
    #     setElevatorSpeed(0)
    # except BaseException as e:
    #     print(e)
    #     setElevatorSpeed(0)
    RAVEN_WRAPPER.set_servo_position(
        Raven.ServoChannel.CH1, 10, min_us=500, max_us=2500)


def setLeftScooperUp():
    RAVEN_WRAPPER.set_servo_position(
        Raven.ServoChannel.CH2, 40, min_us=500, max_us=2500)


def setLeftScooperDown():
    RAVEN_WRAPPER.set_servo_position(
        Raven.ServoChannel.CH2, -50, min_us=500, max_us=2500)


def setElevatorSpeed(speed):
    RAVEN_WRAPPER.set_servo_position(
        Raven.ServoChannel.CH3, speed, min_us=500, max_us=2500)


# CHANNEL 1:
# CHANNEL 2: left scooper up at 40, down at -50, min_us=500, max_us=2500
# CHANNEL 3: elevator, continuous, positive is up
# CHANNEL 4: claw

if __name__ == "__main__":
    main()
