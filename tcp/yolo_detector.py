#!/usr/bin/env python3
"""
YOLO Detector - Wrapper for YOLO object detection model
"""

import base64
import numpy as np
import cv2
from ultralytics import YOLO


class YOLODetector:
    """Wrapper for YOLO object detection model"""

    def __init__(self, model_path='yolo/yolo11n.pt'):
        """
        Initialize YOLO detector

        Args:
            model_path: Path to YOLO model file (.pt)
        """
        self.model_path = model_path
        self.model = None

    def load(self):
        """
        Load the YOLO model

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path, task='detect')
            print("YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            return False

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def detect(self, image, thresh=0.5):
        """
        Run object detection on an image

        Args:
            image: OpenCV image (numpy array) or path to image file
            thresh: Confidence threshold for detections (default: 0.5)

        Returns:
            List of detections, each containing:
            {'class': str, 'confidence': float, 'bbox': [x1, y1, x2, y2]}

        Raises:
            ValueError: If model not loaded or invalid image
        """
        if not self.is_loaded():
            raise ValueError("YOLO model not loaded")

        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image from {image}")

        # Run detection
        results = self.model(image, verbose=False)
        detections = results[0].boxes

        # Extract detection information
        objects = []
        for det in detections:
            conf = float(det.conf[0])
            if conf >= thresh:
                cls_id = int(det.cls[0])
                class_name = self.model.names[cls_id]
                bbox = det.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]

                objects.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': bbox
                })

        return objects

    def detect_from_base64(self, image_b64, thresh=0.5):
        """
        Run object detection on a base64-encoded image

        Args:
            image_b64: Base64-encoded image string
            thresh: Confidence threshold for detections (default: 0.5)

        Returns:
            Tuple of (detections, image_size):
            - detections: List of detection dicts
            - image_size: [width, height] of the image

        Raises:
            ValueError: If model not loaded, invalid image, or decode fails
        """
        if not self.is_loaded():
            raise ValueError("YOLO model not loaded")

        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image from base64")

            # Run detection
            detections = self.detect(image, thresh)

            # Get image size
            image_size = [image.shape[1], image.shape[0]]  # [width, height]

            return detections, image_size

        except Exception as e:
            raise ValueError(f"Error processing base64 image: {e}")


def encode_image_to_base64(image):
    """
    Encode an OpenCV image to base64 string

    Args:
        image: OpenCV image (numpy array) or path to image file

    Returns:
        Base64-encoded image string

    Raises:
        ValueError: If image encoding fails
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Failed to load image from {image}")

    # Encode image to JPEG
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image to JPEG")

    # Convert to base64
    return base64.b64encode(buffer).decode('utf-8')
