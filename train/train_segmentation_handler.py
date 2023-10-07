from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate

def train_segmentation():
    model = YOLO('yolov8n-seg.pt')
    results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
    return

def auto_annotate_segmentation():
    auto_annotate(data="datasets/training_datasets/instance_segmentation/29_9_2023_PhoneDatasetThree_7", det_model="yolov8n-seg.pt", device=0)
    return