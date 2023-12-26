from pathlib import Path
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

import torch
import cv2
from ultralytics import YOLO

from utils.datasets import LoadImages
from utils.general import set_logging, scale_coords, non_max_suppression, custom_non_max_suppression
from utils.torch_utils import select_device
from utils.segmentation import overlay
from utils.plots import plot_one_box
from utils.inference_utils import init_detection_model, init_segmentation_model, unsqueeze_image,\
                                    make_inference_directories, save_inference_directories, write_bbox, write_video

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config

from detect.detect_color import detect_color

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

import utils.handle_db.person_db_handler as person_db

def detect_person(weights: str = 'yolov8.pt',
                        actions_weights: str = '',
                        pwb_weights: str = '',
                        source: str='',
                        conf: float=0.6,
                        iou: float=0.45,
                        img_size: int=640,
                        device= "0",
                        half: bool=True,
                        dont_save: bool=True,
                        save_dir: str = 'datasets/videos_inferred',
                        view_img: bool = False,
                        track: bool = True):

    weights_name = weights.split('/')[1] 

    color_det = not track
    save = not dont_save

    '''Directories'''
    save_dir, save_txt_bbox, save_label_bbox, save_txt_seg, save_label_seg = make_inference_directories(weights_name, segmentation=True)

    '''Initialize'''
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    '''Load Models'''
    model, names, colors = init_segmentation_model(weights)
    # action_model, _, _, action_names, action_colors = init_detection_model(action_weights, half, device, img_size)
    pwb_model, _, _, pwb_names, pwb_colors = init_detection_model(pwb_weights, half, device, img_size)

    '''Initialize Tracker'''
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    '''Set Dataloader'''
    # cap = cv2.VideoCapture(source)
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=img_size)

    '''Dummy Equation of Arc'''
    arcUp = [-0.0105011, 0.0977268, -0.308306, 0.315377, -0.229249, 2.11325]
    arcDown = [0.000527976, 0.00386626, 0.0291599, 0.121282, -2.22398]
    polyUp = np.poly1d(arcUp)
    polyDown = np.poly1d(arcDown)

    '''Color Reference'''
    reference_red_rgb = sRGBColor(1.0, 0.0, 0.0)
    reference_blue_rgb = sRGBColor(0.0, 0.0, 1.0)
    reference_red_lab = convert_color(reference_red_rgb, LabColor)
    reference_blue_lab = convert_color(reference_blue_rgb, LabColor)

    for path, img, im0s, image, vid_cap in dataset:
        '''Normalize image to torch tensors'''
        img = unsqueeze_image(img, device, half)

        '''Inference'''
        results = model.predict(img, stream=True)

        '''Extract frame attributes'''
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        h = int(im0.shape[0])
        w = int(im0.shape[1])

        '''Process detections'''
        for i, det in enumerate(results):  # detections per image
            '''Save to Directories'''
            file_out = '_person_out'
            (p, save_label_video_bbox, label_per_frame_bbox, save_path, 
             txt_path_bbox, save_label_video_seg, label_per_frame_seg, txt_path_seg) = save_inference_directories(frame, p, 
                                                                                                                  save_dir, save_txt_bbox, save_label_bbox,
                                                                                                                  file_out, 
                                                                                                                  save_txt_seg = save_txt_seg, 
                                                                                                                  save_label_seg = save_label_seg,
                                                                                                                  segmentation = True)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            cv2.imwrite(str(save_label_video_bbox / (str(frame) + ".jpg")), image)
            cv2.imwrite(str(save_label_video_seg / (str(frame) + ".jpg")), image)

            boxes = det.boxes  # Boxes object for bbox outputs
            masks = det.masks  # Masks object for segment masks outputs
            probs = det.probs  # Class probabilities for classification outputs

            if masks is not None:
                masks_xyn = det.masks.xyn
                masks = masks.data.cpu()
                for seg, seg_xyn, box in zip(masks, masks_xyn, boxes):
                    segment = '\t'.join(map(str, seg_xyn.reshape(-1).tolist()))

                    seg = seg.data.cpu().numpy()

                    seg = cv2.resize(seg, (w, h))
                    im0, colored_mask = overlay(im0, seg, colors[int(box.cls)], 0.4)

                    r_xyxy = scale_coords(img.shape[2:], box.xyxy.clone(), im0.shape, kpt_label=False).round()
                    write_bbox(frame, im0, r_xyxy.cpu().numpy()[0], txt_path_bbox, label_per_frame_bbox, gn, conf, names, box.cls, 
                               colors, segment = segment, txt_path_seg = txt_path_seg, label_per_frame_seg = label_per_frame_seg, segmentation = True)
                        
                    label = names[int(box.cls)]

                    b_x, b_y, b_w, b_h = box.xywh.cpu().numpy()[0]

                    x_center_below = b_x + b_w/2
                    y_center_below = b_y + b_h

                    yUp = polyUp(x_center_below)
                    yDown = polyDown(x_center_below)

                    position=""
                    if yUp > y_center_below and yDown < y_center_below:
                        position = "2_points"
                    else:
                        position = "3_points"

                    if color_det:
                        detect_color(colored_mask, reference_red_lab, reference_blue_lab)

                if track:
                    xywh_bboxs = []
                    confs = []
                    oids = []
                    outputs = []
                    for box in boxes:
                        xywh_bboxs.append(box.xywh.cpu().numpy()[0])
                        confs.append([box.conf.cpu().numpy()])
                        oids.append(box.cls)
                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    
                    outputs = deepsort.update(xywhs, confss, oids, im0)

                    if len(outputs) > 0:
                        track_entry = []
                        max_conf = 0
                        for track_det in outputs:
                            '''Track ID'''
                            class_name = "person"
                            bbox_xyxy = track_det[:4]
                            identities = track_det[-2]
                            object_id = track_det[-1]
                            categories = class_name

                            r_bbox_xyxy = scale_coords(img.shape[2:], torch.Tensor([bbox_xyxy]), im0.shape, kpt_label=False).round()

                            '''ROI'''
                            # box_x1, box_y1, box_x2, box_y2 = r_bbox_xyxy.cpu().numpy()[0]
                            box_x1, box_y1, box_x2, box_y2 = bbox_xyxy
                            roi = im0[int(box_y1)-50:int(box_y2)+50, int(box_x1)-50:int(box_x2)+50]
                            roi_height, roi_width, channels = roi.shape
                            roi_height = min(roi_height, 384)
                            roi_width = min(roi_width, 640)
                            roi = roi[0:roi_height, 0:roi_width]

                            # Padding
                            roi_padded = np.zeros([384, 640, 3], dtype=np.uint8)
                            roi_padded[0:roi_height, 0:roi_width] = roi
                            roi_padded = roi_padded.transpose((2, 0, 1))

                            frame_roi_tensor = unsqueeze_image(roi_padded, device, half)

                            '''Player with Basketball Model'''
                            pwb_label, pwb_conf = detect_pwb(pwb_model, pwb_names, pwb_colors, frame_roi_tensor, 0.1, 0.8, True)
                            if pwb_label == None:
                                pwb_label = ''

                            if pwb_conf is not None:
                                if pwb_conf.item() > max_conf:
                                    max_conf = pwb_conf.item()

                            '''New Actions Model'''
                            # action_label = detect_action(action_model, action_names, action_colors, roi, conf, iou, True)
                            action_label = 'action'

                            # Draw Boxes on Image
                            label = "player: " + str(identities)
                            plot_one_box(r_bbox_xyxy.cpu().numpy()[0], im0, colors[int(box.cls)], f'{label}')

                            '''Add tracked player to list entry to compare conf with other tracked players'''
                            entry = {
                                'player_num': int(identities),
                                'bbox_coords': r_bbox_xyxy.tolist(),
                                'feet_coords': [int(x_center_below), int(y_center_below)],
                                'position': position,
                                'action': action_label,
                                'player_with_basketball': pwb_label,
                                'conf': pwb_conf.item() if pwb_conf is not None else 0.0
                            }

                            track_entry.append(entry)

                        '''Write to DB'''
                        '''Get player with basketball wit highest confidence'''
                        for entry in track_entry:
                            if entry['conf'] < max_conf:
                                person_db.insert_into_person_table(
                                    frame_num=frame,
                                    player_num= entry['player_num'],
                                    bbox_coords= entry['bbox_coords'],
                                    feet_coords= entry['feet_coords'],
                                    position= entry['position'],
                                    action = entry['action'],
                                    player_with_basketball = ''
                                )
                            else:
                                person_db.insert_into_person_table(
                                    frame_num=frame,
                                    player_num= entry['player_num'],
                                    bbox_coords= entry['bbox_coords'],
                                    feet_coords= entry['feet_coords'],
                                    position= entry['position'],
                                    action = entry['action'],
                                    player_with_basketball = entry['player_with_basketball']
                                )

        '''Stream results'''
        if view_img:
            cv2.imshow(str('img'), im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        '''Save results (video with detections)'''
        if save:
            if vid_path != save_path:  # new video
                vid_path = save_path
                vid_writer = write_video(im0, save_path, vid_writer, vid_cap)
            vid_writer.write(im0)


def detect_pwb(model, names, colors, img, conf_thresh, iou_thresh, augment):
    pred = model(img, augment=augment)[0]
    pred = custom_non_max_suppression(pred, conf_thresh, iou_thresh, merge=False)

    for i, det in enumerate(pred):
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                if label == 'basketball':
                    return label, conf
    return None, None