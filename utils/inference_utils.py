from pathlib import Path
import random

import torch
import cv2

from ultralytics import YOLO

from models.experimental import attempt_load

from utils.general import check_img_size, non_max_suppression, xyxy2xywh
from utils.torch_utils import TracedModel, time_synchronized
from utils.plots import plot_one_box

from persistence.repositories import paths


def make_inference_directories(weights_name, segmentation=False):
    '''Directories'''
    save_dir = paths.video_inferred_path / weights_name
    save_txt_bbox = paths.bbox_coordinates_path / weights_name / "bbox"
    save_label_bbox = paths.labels_path / weights_name / "bbox"

    save_dir.mkdir(parents=True, exist_ok=True)  # create directory
    save_txt_bbox.mkdir(parents=True, exist_ok=True)  # create directory
    save_label_bbox.mkdir(parents=True, exist_ok=True)  # create directory

    if segmentation:
        save_txt_seg = paths.bbox_coordinates_path / weights_name / "segments"
        save_label_seg = paths.labels_path / weights_name / "segments"
        save_txt_seg.mkdir(parents=True, exist_ok=True)  # create directory
        save_label_seg.mkdir(parents=True, exist_ok=True)  # create directory

        return save_dir, save_txt_bbox, save_label_bbox, save_txt_seg, save_label_seg
    else:
        return save_dir, save_txt_bbox, save_label_bbox
    

def save_inference_directories(frame, path, save_dir, save_txt_bbox, save_label_bbox, file_out, save_txt_seg = '', save_label_seg = '', segmentation = False):
    p = Path(path)  # to Path
    filename = (p.name.replace(" ", "_"))
    save_label_video_bbox = Path(save_label_bbox / (filename.split('.')[0]))
    save_label_video_bbox.mkdir(parents=True, exist_ok=True)  # make dir
    label_per_frame_bbox = str(save_label_video_bbox / (str(frame) + '.txt'))
    save_path = str(save_dir / (filename.split('.')[0] + file_out + ".mp4"))  # img.jpg
    txt_path_bbox = str(save_txt_bbox / (filename.split('.')[0] + '.txt'))
    
    if segmentation:
        save_label_video_seg = Path(save_label_seg / (filename.split('.')[0]))
        save_label_video_seg.mkdir(parents=True, exist_ok=True)  # make dir
        label_per_frame_seg = str(save_label_video_seg / (str(frame) + '.txt'))
        txt_path_seg = str(save_txt_seg / (filename.split('.')[0] + '.txt'))
        return p, save_label_video_bbox, label_per_frame_bbox, save_path, txt_path_bbox, save_label_video_seg, label_per_frame_seg, txt_path_seg
    else:
        return p, save_label_video_bbox, label_per_frame_bbox, save_path, txt_path_bbox


def init_detection_model(weights, half, device, img_size):
    '''Load model'''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    model = TracedModel(model, device, img_size)
    if half:
        model.half()  # to FP16

    '''Get names and colors'''
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    return model, imgsz, stride, names, colors


def init_segmentation_model(weights):
    '''Load Model'''
    model = YOLO(weights)

    '''Get names and colors'''
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    return model, names, colors


def unsqueeze_image(image, device, half):
    img = torch.from_numpy(image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def warmup(model, img, imgsz, device, augment, conf_thresh, iou_thresh):
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    if device.type != 'cpu' and (
            old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for _ in range(3):
            model(img, augment=augment)[0]
                
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    t2 = time_synchronized()

    '''Apply NMS'''
    pred = non_max_suppression(pred, conf_thresh, iou_thresh)
    t3 = time_synchronized()

    return pred, t1, t2, t3


def write_bbox(frame, im0, xyxy, txt_path, label_per_frame, gn, conf, names, cls, colors, segment = '', txt_path_seg = '', label_per_frame_seg = '', segmentation = False):
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
    xywh_label = ' '.join(map(str, ['%.5f' % elem for elem in xywh]))
    xywh = '\t'.join(map(str, ['%.5f' % elem for elem in xywh]))

    line = [str(frame), names[int(cls)], xywh, str(round(float(conf), 5))]
    with open(txt_path, 'a') as f:
        f.write(('\t'.join(line)) + '\n')

    label = [str(int(cls)), xywh_label]
    with open(label_per_frame, 'a') as f:
        f.write((' '.join(label)) + '\n')

    if segmentation:
        line = [str(frame), names[int(cls)], segment, str(round(float(conf), 5))]
        with open(txt_path_seg, 'a') as f:
            f.write(('\t'.join(line)) + '\n')

        label = [str(int(cls)), segment]
        with open(label_per_frame_seg, 'a') as f:
            f.write((' '.join(label)) + '\n')


def write_video(im0, save_path, vid_writer, vid_cap):
    if isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()  # release previous video writer
    if vid_cap:  # video
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        vid_w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:  # stream
        fps, w, h = 30, im0.shape[1], im0.shape[0]
        save_path += '.mp4'
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (vid_w, vid_h))

    return vid_writer
