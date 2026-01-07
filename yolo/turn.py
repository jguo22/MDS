import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

from raven import Raven

# Configuration
MODEL_PATH = 'yolo11n.pt'
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
labels = model.names

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_INDEX}")
    sys.exit(1)

# Test camera
ret, test_frame = cap.read()
if not ret:
    print("Error: Could not read from camera")
    sys.exit(1)

frame_height, frame_width = test_frame.shape[:2]
print(f"Camera resolution: {frame_width}x{frame_height}")

if not DISPLAY_ENABLED:
    print("Running in headless mode - no UI will be shown")

# Initialize Raven servo controller
servo = RavenServoController(
    servo_channel=SERVO_CHANNEL,
    min_angle=MIN_ANGLE,
    max_angle=MAX_ANGLE,
    min_us=MIN_US,
    max_us=MAX_US,
)

# Bounding box colors
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

# FPS tracking
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30

# Deadzone removed

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
        t_start = time.perf_counter()

        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            break

        # Run YOLO detection
        results = model(frame, verbose=False)
        detections = results[0].boxes

        # Find target object (largest matching object)
        target_detection = None
        target_center_x = None
        largest_area = 0

        for i in range(len(detections)):
            conf = detections[i].conf.item()

            if conf < CONFIDENCE_THRESHOLD:
                continue

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            # Filter by target class if specified
            if TARGET_CLASS and classname != TARGET_CLASS:
                continue

            # Get bounding box
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Calculate area
            area = (xmax - xmin) * (ymax - ymin)

            # Track largest object
            if area > largest_area:
                largest_area = area
                target_detection = detections[i]
                target_center_x = (xmin + xmax) / 2
                target_bbox = (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    classidx,
                    classname,
                    conf)

        # Draw all detections
        for i in range(len(detections)):
            conf = detections[i].conf.item()
            if conf < CONFIDENCE_THRESHOLD:
                continue

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]

            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            color = bbox_colors[classidx % 10]

            # Highlight target with thicker border
            is_target = (target_detection is not None and
                         detections[i].conf.item() == target_detection.conf.item() and
                         classidx == int(target_detection.cls.item()))

            thickness = 3 if is_target else 1
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)

            # Label
            label = f'{classname}: {int(conf*100)}%'
            if is_target:
                label = f'[TARGET] {label}'

            labelSize, baseLine = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(
                frame,
                (xmin,
                 label_ymin -
                 labelSize[1] -
                 10),
                (xmin +
                 labelSize[0],
                 label_ymin +
                 baseLine -
                 10),
                color,
                cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Control servo based on target position
        if target_center_x is not None:
            frame_center_x = frame_width / 2
            offset = target_center_x - frame_center_x

            # Draw target center and frame center line
            cv2.circle(frame, (int(target_center_x),
                               frame_height // 2), 10, (0, 255, 0), -1)
            cv2.line(frame, (int(frame_center_x), 0),
                     (int(frame_center_x), frame_height), (255, 255, 0), 1)

            # Calculate servo angle based on offset
            # Normalize offset to -1.0 to 1.0
            normalized_offset = offset / (frame_width / 2)

            # Convert to servo angle
            angle_range = MAX_ANGLE - MIN_ANGLE
            servo_angle = normalized_offset * (angle_range / 2)

            servo.set_angle(servo_angle)
            status_text = f'Tracking: {target_bbox[5]} | Angle: {servo.current_angle:.1f}°'
        else:
            # Don't change the angle when no target is detected, just update status
            status_text = f'No target detected | Angle: {servo.current_angle:.1f}°'

        # Display info on frame
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show frame or print status
        if DISPLAY_ENABLED:
            cv2.imshow('YOLO Object Tracking', frame)
            # Handle keyboard input
            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                cv2.waitKey()
        else:
            # Print status to console in headless mode
            print(f"\r{status_text} | FPS: {avg_frame_rate:0.1f}", end='', flush=True)

        # Calculate FPS
        t_stop = time.perf_counter()
        frame_rate_calc = float(1 / (t_stop - t_start))

        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)

        avg_frame_rate = np.mean(frame_rate_buffer)

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    print(f'\nAverage FPS: {avg_frame_rate:.2f}')
    servo.cleanup()
    cap.release()
    if DISPLAY_ENABLED:
        cv2.destroyAllWindows()
    print("Cleanup complete")
