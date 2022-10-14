import torch

from domain.contracts.repositories.abstract_detection_service import AbstractDetectionService
from yolo_v7_model.detect import detect
from yolo_v7_model.utils.general import strip_optimizer


class DetectionService(AbstractDetectionService):
    weights: str = 'weights/yolov7.pt'

    def infer_detection(self, source: str) -> str:
        with torch.no_grad():
            video_path = detect(weights=self.weights, source=source)
            strip_optimizer(self.weights)
        return video_path
