from pathlib import Path
import time

import numpy as np

import torch
import cv2

from models.experimental import attempt_load

from utils.datasets import LoadImages
from utils.general import set_logging, scale_coords
from utils.torch_utils import select_device
from utils.plots import plot_one_box
from utils.inference_utils import init_detection_model,\
    unsqueeze_image, warmup, make_inference_directories, save_inference_directories, write_bbox, write_video

from persistence.repositories import paths

import utils.handle_db.basket_db_handler as basket_db

def detect_basketball(weights: str = 'yolov7.pt',
           source: str = 'inference/images',
           img_size: int = 640,
           conf_thresh: float = 0.6,
           iou_thresh: float = 0.45,
           device: str = '',
           view_img: bool = False,
           dont_save: bool = False,
           augment: bool = False) -> tuple:
    """
    Performs inference on an input video
    :param weights: YOLO-V7 .pt file
    :param source: path of the video/image to be processed
    :param img_size: value to resize frames to
    :param conf_thresh: confidence threshold
    :param iou_thresh: intersection over union threshold
    :param device: device to be used GPU/CPU
    :param view_img: view inferred frame
    :param dont_save: save inferred frames
    :param augment: augment frames
    """
    time.sleep(5)
    print("weigths: ", weights)

    save_img = not dont_save and not source.endswith('.txt')  # save inference images

    weights_name = weights.split('/')[1] 

    '''Directories'''
    save_dir, save_txt, save_label = make_inference_directories(weights_name)

    '''Initialize'''
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    '''Load model'''
    model, imgsz, stride, names, colors = init_detection_model(weights, half, device, img_size)

    '''Set Dataloader'''
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    '''Run inference'''
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, image, vid_cap in dataset:
        img = unsqueeze_image(img, device, half)

        '''Warmup'''
        pred, t1, t2, t3 = warmup(model, img, imgsz, device, augment, conf_thresh, iou_thresh)

        '''Extract frame attributes'''
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        '''Process detections'''
        for i, det in enumerate(pred):  # detections per image
                file_out = "_nethoopbasket_out"
                p, save_label_video, label_per_frame, save_path, txt_path = save_inference_directories(frame, p, save_dir, save_txt, save_label, file_out)
                
                cv2.imwrite(str(save_label_video / (str(frame) + ".jpg")), image)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    '''Rescale boxes from img_size to im0 size'''
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False).round()

                    '''Print results'''
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    '''Write results
                            bounding box: xyxy (x1 y1 x2 y2)
                                    convert xyxy --> xywh : x1 y1 width height
                    '''

                    for *xyxy, conf, cls in reversed(det):
                        write_bbox(frame, im0, xyxy, txt_path, label_per_frame, gn, conf, names, cls, colors)

                        label = names[int(cls)]
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1, kpt_label=False)

                        basket_db.insert_into_basket_table(
                            frame_num=frame,
                            shot=label,
                            bbox_coords= [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        )

                '''Print inference and NMS time'''
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                '''Stream results'''
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                '''Save results (video with detections)'''
                if save_img:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        vid_writer = write_video(im0, save_path, vid_writer, vid_cap)
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')