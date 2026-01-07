import os
import sys
import time

from ultralytics import YOLO

from raven import Raven
import cv2

raven_board = Raven()

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
CONFIDENCE_THRESHOLD = 0.3
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


# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f'ERROR: Model path "{MODEL_PATH}" not found.')
    sys.exit(1)

# Load YOLO model
model = YOLO(MODEL_PATH, task='detect')
labels = model.names


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
print(f"Press Ctrl+C to quit\n")

raven_board.set_motor_encoder(Raven.MotorChannel.CH3, 0) # Set encoder count for motor 1 to zero
raven_board.set_motor_mode(Raven.MotorChannel.CH3, Raven.MotorMode.DIRECT) # Set motor mode to DIRECT

raven_board.set_motor_encoder(Raven.MotorChannel.CH2, 0)
raven_board.set_motor_mode(Raven.MotorChannel.CH2, Raven.MotorMode.DIRECT)

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
object_count = 0



try:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        results = model(frame, verbose=False)
        detections = results[0].boxes
        for i in range(len(detections)):
            # Get bounding box coordinates
            # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
            xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # Get bounding box confidence
            conf = detections[i].conf.item()

            # Draw box if confidence threshold is high enough
            if conf > 0.5:

                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

                # Basic example: count the number of objects in the image
                object_count = object_count + 1

        changed = False

        # Display detection results
        cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
        cv2.imshow('YOLO detection results', frame) # Display image

        for i in range(len(detections)):
            conf = detections[i].conf.item()

            if conf < CONFIDENCE_THRESHOLD:
                continue
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # MOVE TOWARDS CENTER

            raven_board.set_motor_torque_factor(Raven.MotorChannel.CH3, 50)
            raven_board.set_motor_speed_factor(Raven.MotorChannel.CH3, 5, reverse=False)
            raven_board.set_motor_torque_factor(Raven.MotorChannel.CH2, 50)
            raven_board.set_motor_speed_factor(Raven.MotorChannel.CH2, 5, reverse=True)
            changed = True
            print("Found object " + classname)
            break

        if (changed == False):
            raven_board.set_motor_torque_factor(Raven.MotorChannel.CH3, 0)
            raven_board.set_motor_speed_factor(Raven.MotorChannel.CH3, 0)
            raven_board.set_motor_torque_factor(Raven.MotorChannel.CH2, 0)
            raven_board.set_motor_speed_factor(Raven.MotorChannel.CH2, 0)
            print("no object")
    # Calculate and draw framerate (if using video, USB, or Picamera source)
    cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate

    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO detection results',frame) # Display image

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)



except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    servo.cleanup()
    raven_board.set_motor_torque_factor(Raven.MotorChannel.CH3, 0)
    raven_board.set_motor_speed_factor(Raven.MotorChannel.CH3, 0)
    raven_board.set_motor_torque_factor(Raven.MotorChannel.CH2, 0)
    raven_board.set_motor_speed_factor(Raven.MotorChannel.CH2, 0)
    cv2.destroyAllWindows()
    print("Cleanup complete")
