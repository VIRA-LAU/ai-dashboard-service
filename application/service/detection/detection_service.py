import torch
from detect import detect
from utils.general import strip_optimizer


class DetectionService:
    def __init__(self):
        self.weights = 'weights/yolov7.pt'

    def infer_detection(self, source: str):
        with torch.no_grad():
            video_path = detect(weights=self.weights, source=source)
            strip_optimizer(self.weights)
        return video_path
