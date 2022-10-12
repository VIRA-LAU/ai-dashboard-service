from fastapi import UploadFile, File
import torch
from yolo_v7_model.detect import detect
from yolo_v7_model.utils.general import strip_optimizer


class DetectionService:
    def __init__(self):
        self.weights = 'yolo_v7_model/weights/yolov7.pt'

    def infer_detection(self, source: str):
        with torch.no_grad():
            video_path = detect(self.weights, source)
            strip_optimizer(self.weights)
        return video_path
