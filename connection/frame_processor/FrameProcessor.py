import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class FrameProcessor(ABC):
    @abstractmethod
    def process(self, frame: np.ndarray,
                frame_id: int) -> Optional[Tuple[float, float, float]]:
        pass

