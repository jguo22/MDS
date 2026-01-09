from raven import Raven
import cv2
import numpy as np

raven_board = Raven()


def drawBox(classidx, frame, xmin, ymin, xmax, ymax, classname):
    color = bbox_colors[classidx % 10]
    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

    label = f'{classname}: {int(0.5*100)}%'
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

# Configuration
CAMERA_INDEX = 0
SERVO_CHANNEL = 4
MIN_ANGLE = -90
MAX_ANGLE = 90
MIN_US = 500
MAX_US = 2500
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASS = None
CENTER_DEADZONE = 5
DISPLAY_ENABLED = False

class RavenServoController:
    def __init__(
            self,
            servo_channel=1,
            min_angle=-90,
            max_angle=90,
            min_us=500,
            max_us=2500,
            use_servo=True):
        self.servo_channel = servo_channel
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_us = min_us
        self.max_us = max_us
        self.use_servo = use_servo
        self.current_angle = 0

        print(f"Using Servo: {use_servo}")
        if use_servo:
            self.raven_board = raven_board

            # Set servo channel
            if servo_channel == 1:
                self.channel = Raven.ServoChannel.CH3
            elif servo_channel == 2:
                self.channel = Raven.ServoChannel.CH2
            elif servo_channel == 3:
                self.channel = Raven.ServoChannel.CH3
            elif servo_channel == 4:
                self.channel = Raven.ServoChannel.CH4
            else:
                raise Exception("Invalid Servo Channel")

            # Initialize servo to center position
            self.set_angle(0)

            print(f"Servo initialized on channel {servo_channel}")
            print(f"Angle range: {min_angle}° to {max_angle}°")
            print(f"Pulse width: {min_us}us to {max_us}us")
        else:
            print(f"Simulation mode: Servo channel {servo_channel}")
            print(f"Angle range: {min_angle}° to {max_angle}°")

    def set_angle(self, angle):
        """
        Set servo angle based on target position
        angle: angle in degrees (within min_angle to max_angle range)
        """
        # Clamp angle to min/max range
        angle = max(self.min_angle, min(self.max_angle, angle))

        if self.use_servo:
            self.raven_board.set_servo_position(
                self.channel, angle, min_us=self.min_us, max_us=self.max_us)
        else:
            print(f"[SIMULATION] Servo angle: {angle:.3f}°")

        self.current_angle = angle

    def cleanup(self):
        """Clean up servo controller"""
        if self.use_servo:
            print("Servo stopped")
class RavenMotorControllers:
    def __init__(
            self):

        self.leftChannel = Raven.MotorChannel.CH3
        self.rightChannel = Raven.MotorChannel.CH2
        self.rotating = False

        raven_board.set_motor_encoder(self.leftChannel, 1)
        raven_board.set_motor_mode(self.leftChannel, Raven.MotorMode.DIRECT)
        raven_board.set_motor_encoder(self.rightChannel, 1)
        raven_board.set_motor_mode(self.rightChannel, Raven.MotorMode.DIRECT)


    def setTorque(self, torque):
        raven_board.set_motor_torque_factor(self.leftChannel, torque)
        raven_board.set_motor_torque_factor(self.rightChannel, torque)
    def setSpeed(self, speed, reverse = False):
        raven_board.set_motor_speed_factor(self.leftChannel, speed, reverse = reverse)
        raven_board.set_motor_speed_factor(self.rightChannel, speed, reverse = not reverse)
    def rotateInPlace(self, speed, clockwise = True):
        self.rotating = True
        raven_board.set_motor_torque_factor(self.leftChannel, 20)
        raven_board.set_motor_torque_factor(self.rightChannel, 20)
        raven_board.set_motor_speed_factor(self.leftChannel, speed, reverse = clockwise)
        raven_board.set_motor_speed_factor(self.rightChannel, speed, reverse = clockwise)
    def stopRotating(self):
        self.rotating = False
        raven_board.set_motor_torque_factor(self.leftChannel, 0)
        raven_board.set_motor_torque_factor(self.rightChannel, 0)
        raven_board.set_motor_speed_factor(self.leftChannel, 0)
        raven_board.set_motor_speed_factor(self.rightChannel, 0)


img_source = "usb"
min_thresh = float(0.2)
user_res = None

cap = cv2.VideoCapture(0)

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]


# Initialize Raven servo controller
servo = RavenServoController(
    servo_channel=SERVO_CHANNEL,
    min_angle=MIN_ANGLE,
    max_angle=MAX_ANGLE,
    min_us=MIN_US,
    max_us=MAX_US,
)

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

if (not DISPLAY_ENABLED):
    print("Running headless")

# robot = Robot()
# # Begin inference loop
# try:
#     while True:
#         # print("new cap")
#         t_start = time.perf_counter()

#         robot.setNowTime()

#         # print ("current state is " + robot.state.name)
#         match robot.state:
#             # Checking Search Mode. Is stationary and will check the image, if there's nothing, then it will rotate in place.
#             case RobotState.CHECKING_SEARCH:
#                 # Stop moving for 0.5s to stabilize image
#                 if (robot.now - robot.state_start > 1):
#                     # Check for humans. If found, seek. If not, return to searching.
#                     print("Checking image.")
#                     robot.checkImage(cap)

#             case RobotState.SEARCHING:
#                 # Change to Checking Search Mode and Rotate after checking for 0.2s (good for ~12 FPS)
#                 if (robot.now - robot.state_start > 0.3):
#                     print("Stopping search.")
#                     robot.stopSearching()
#             # Rotate for correction until in margin
#             case RobotState.SEEKING_CORRECTION:
#                 # Assuming we are already rotating, stop rotating when in margin
#                 # Rotate for 0.3s
#                 if (robot.now - robot.state_start > 0.3):
#                     robot.checkRotation(cap)
#             # Move foward for 2 seconds, then recheck for correction
#             case RobotState.SEEKING_MOVING:
#                 if (robot.now - robot.state_start > 2):
#                     print("rechecking image for humans.")
#                     robot.checkImage(cap)
#             # Look again
#             case RobotState.REFINDING_OBJECT:
#                 if (robot.now - robot.state_start > 1):
#                     print("trying to refind the object.")
#                     robot.checkRotation(cap)


#         # Calculate and draw framerate (if using video, USB, or Picamera source)
#         # if (DISPLAY_ENABLED):
#         #     cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate

#         #     # Display detection results
#         #     cv2.imshow('YOLO detection results',frame) # Display image
#         # else:
#         #     print(f"FPS: {avg_frame_rate:0.2f}")

#         # Calculate FPS for this frame
#         t_stop = time.perf_counter()
#         frame_rate_calc = float(1/(t_stop - t_start))

#         # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
#         if len(frame_rate_buffer) >= fps_avg_len:
#             temp = frame_rate_buffer.pop(0)
#             frame_rate_buffer.append(frame_rate_calc)
#         else:
#             frame_rate_buffer.append(frame_rate_calc)

#         # Calculate average FPS for past frames
#         avg_frame_rate = np.mean(frame_rate_buffer)


# except KeyboardInterrupt:
#     print("\nInterrupted by user")

# finally:
#     # Clean up
#     print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
#     cap.release()
#     cv2.destroyAllWindows()

#     raven_board.set_motor_torque_factor(Raven.MotorChannel.CH3, 0)
#     raven_board.set_motor_speed_factor(Raven.MotorChannel.CH3, 0)
#     raven_board.set_motor_torque_factor(Raven.MotorChannel.CH2, 0)
#     raven_board.set_motor_speed_factor(Raven.MotorChannel.CH2, 0)
