import os
import time
import cv2
import numpy as np

from .FrameProcessor import FrameProcessor


class SaveImageProcessor(FrameProcessor):
    """
    Frame processor that saves frames to disk with a cooldown period between saves.
    Images are saved in the 'images' directory with timestamps in their filenames.
    """

    def __init__(
            self,
            cooldown_seconds: float = 1.0,
            output_dir: str = "images"):
        """
        Initialize the ImageProcessor.

        Args:
            cooldown_seconds: Minimum time between saves in seconds
            output_dir: Directory to save images (will be created if it doesn't exist)
        """
        self.cooldown = cooldown_seconds
        self.output_dir = output_dir
        self.last_save_time = 0

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def process(
            self,
            frame: np.ndarray,
            frame_id: int,
            x: float,
            y: float,
            theta: float) -> None:
        """
        Process a frame, saving it to disk if cooldown has passed.

        Args:
            frame: Input frame as a numpy array
            frame_id: Frame identifier (unused in this implementation)

        Returns:
            Always returns None
        """
        print(frame.shape)
        current_time = time.time()

        # Check if cooldown has passed
        if current_time - self.last_save_time >= self.cooldown:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.output_dir,
                f"frame_{timestamp}_{int(current_time*1000)}.jpg")

            try:
                # Save the frame as a JPEG file
                cv2.imwrite(filename, frame)
                self.last_save_time = current_time
                print(f"Saved frame to {filename}")
            except Exception as e:
                print(f"Error saving frame: {e}")

        return None
