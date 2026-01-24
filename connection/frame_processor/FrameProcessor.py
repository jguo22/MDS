from abc import ABC, abstractmethod
from connection.frame_info import FrameInfo


class FrameProcessor(ABC):
    @abstractmethod
    def process(self, frame_info: FrameInfo) -> None:
        pass
