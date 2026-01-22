import numpy as np
from abc import ABC, abstractmethod


class FrameProcessor(ABC):
    @abstractmethod
    def process(
            self,
            frame: np.ndarray,
            frame_id: int,
            x: float,
            y: float,
            theta: float) -> None:
        pass
