import os
import time
import cv2
from connection.frame_info import FrameInfo


class FrameSaver():
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

    def saveFrame(self, frame_info: FrameInfo) -> None:
        """
        Process a frame, saving it to disk if cooldown has passed.

        Args:
            frame_info: FrameInfo object containing frames and metadata

        Returns:
            Always returns None
        """
        current_time = time.time()

        # Check if cooldown has passed
        if current_time - self.last_save_time >= self.cooldown:
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename_top = os.path.join(
                self.output_dir,
                f"frame_top_{timestamp}_{int(current_time*1000)}.jpg")
            filename_bottom = os.path.join(
                self.output_dir,
                f"frame_bottom_{timestamp}_{int(current_time*1000)}.jpg")

            try:
                # Save both frames as JPEG files
                cv2.imwrite(filename_top, frame_info.frame_top)
                cv2.imwrite(filename_bottom, frame_info.frame_bottom)
                self.last_save_time = current_time
                print(f"Saved frames to {filename_top} and {filename_bottom}")
            except Exception as e:
                print(f"Error saving frames: {e}")

        return None
