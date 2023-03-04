from abc import ABC, abstractmethod


class AbstractDetectionService(ABC):
    @abstractmethod
    def infer_detection(self, source: str):
        pass
