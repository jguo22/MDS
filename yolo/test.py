import os
import sys

from ultralytics import YOLO

from raven import Raven

# Get absolute path to model file (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'yolo11n_ncnn_model')

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
            self.raven_board = Raven()

            # Set servo channel
            if servo_channel == 1:
                self.channel = Raven.ServoChannel.CH1
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


# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f'ERROR: Model path "{MODEL_PATH}" not found.')
    sys.exit(1)

# Load YOLO model
model = YOLO(MODEL_PATH, task='detect')


# Initialize Raven servo controller
servo = RavenServoController(
    servo_channel=SERVO_CHANNEL,
    min_angle=MIN_ANGLE,
    max_angle=MAX_ANGLE,
    min_us=MIN_US,
    max_us=MAX_US,
)

print("\nStarting object tracking...")
print(
    f"Target class: {TARGET_CLASS if TARGET_CLASS else 'Any object'}")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
if DISPLAY_ENABLED:
    print(f"Press 'q' to quit, 's' to pause\n")
else:
    print(f"Press Ctrl+C to quit\n")

try:
    while True:
        results = model(source=0, verbose=False)
        detections = results[0].boxes

        for i in range(len(detections)):
            conf = detections[i].conf.item()

            if conf < CONFIDENCE_THRESHOLD:
                continue
            servo.set_angle(50)

        servo.set_angle(0)



except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    servo.cleanup()
    print("Cleanup complete")
