from pathlib import Path
import yaml
import time

import torch
import yaml
import random
import os

import numpy as np

from models.experimental import attempt_load

from utils.paths.game import get_game_data
from persistence.repositories import paths
from utils.datasets import LoadImages
from utils.general import strip_optimizer, set_logging, non_max_suppression, non_max_suppression_kpt, scale_coords, xyxy2xywh, check_img_size
from utils.torch_utils import time_synchronized, select_device, TracedModel
import utils.handle_db.action_db_handler as action_db
import utils.handle_db.basket_db_handler as basket_db
import utils.handle_db.pose_db_handler as pose_db

from utils.args import *
from utils.frame_extraction import extract_frames
from utils.dominant_color import extractDominantColor, prety_print_data, plotColorBar
from utils.segmentation import overlay
from utils.plots import plot_one_box

from ultralytics import YOLO

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

import cv2
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


def extract_key_frames(dataset_folder: str = "", video_max_len: int = 200, fps: int = 5):
    out_dir = paths.video_input_path # output video with less frames
    out_dir_frames = paths.temporal_frames  # output frames

    source = os.path.join(paths.temporal_videos_input_path, dataset_folder)
    for video in os.listdir(source):
        vid_path = os.path.join(source, video)
        extract_frames(vid_path, fps, video_max_len, out_dir_frames)


def instance_segmentation(weights: str = 'yolov8.pt',
                        reid_model: Path = '',
                        tracking_method: str = 'deepocsort',
                        source: str='',
                        conf: float=0.6,
                        iou: float=0.45,
                        show: bool=False,
                        img_size: int=640,
                        stream: bool=True,
                        device= torch.device("cuda:0"),
                        half: bool=True,
                        show_conf: bool=False,
                        save_txt: bool=False,
                        show_labels: bool=True,
                        dont_save: bool=True,
                        save_mot: bool=True,
                        save_id_crops: bool=True,
                        verbose: bool=True,
                        exist_ok: bool=False,
                        save_dir: str = 'datasets/videos_inferred',
                        name: str = 'segmentation',
                        classes: int = 0,
                        per_class: bool = False,
                        vid_stride: int = 1,
                        line_width: int = 3,
                        view_img: bool = False,
                        track: bool = True):

    weights_name = weights.split('/')[1] 

    color_det = not track
    save = not dont_save

    '''Directories'''
    save_dir = paths.video_inferred_path / weights_name
    save_txt = paths.bbox_coordinates_path / weights_name
    save_label = paths.labels_path / weights_name

    save_dir.mkdir(parents=True, exist_ok=True)  # create directory
    save_txt.mkdir(parents=True, exist_ok=True)  # create directory
    save_label.mkdir(parents=True, exist_ok=True)  # create directory

    '''Load model'''
    model = YOLO(weights)

    '''Initialize Tracker'''
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    '''Logging'''
    seg_logs={
        'segmentation': {}
    }

    '''Set Dataloader'''
    # cap = cv2.VideoCapture(source)
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=img_size)

    '''Get names and colors'''
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

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
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        '''Inference'''
        results = model.predict(img, stream=True)

        '''Extract frame attributes'''
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        h = int(im0.shape[0])
        w = int(im0.shape[1])

        '''Add empty entry to logs'''
        seg_logs['segmentation'][frame] = {}

        '''Process detections'''
        for i, det in enumerate(results):  # detections per image
            p = Path(p)  # to Path
            filename = (p.name.replace(" ", "_"))
            save_label_video = Path(save_label / (filename.split('.')[0]))
            save_label_video.mkdir(parents=True, exist_ok=True)  # make dir
            label_per_frame = str(save_label_video / (str(frame) + '.txt'))
            save_path = str(save_dir / (filename.split('.')[0] + "_actions_out" + ".mp4"))  # img.jpg
            txt_path = str(save_txt / (filename.split('.')[0] + '.txt'))

            boxes = det.boxes  # Boxes object for bbox outputs
            masks = det.masks  # Masks object for segment masks outputs
            probs = det.probs  # Class probabilities for classification outputs

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            cv2.imwrite(str(save_label_video / (str(frame) + ".jpg")), image)

            if masks is not None:
                for box in boxes:
                    r_xyxy = scale_coords(img.shape[2:], box.xyxy.clone(), im0.shape, kpt_label=False).round()
                    # xywh = box.xywhn.cpu().numpy()[0] # normalized 
                    xywh = (xyxy2xywh(torch.tensor(r_xyxy.cpu().numpy()[0]).view(1, 4)) / gn).view(-1).tolist()
                    xywh_label = ' '.join(map(str, ['%.5f' % elem for elem in xywh]))
                    xywh = '\t'.join(map(str, ['%.5f' % elem for elem in xywh]))
                    line = [str(frame), names[int(box.cls)], xywh, str(round(float(conf), 5))]
                    with open(txt_path, 'a') as f:
                        f.write(('\t'.join(line)) + '\n')
                    label = [str(int(box.cls)), xywh_label]
                    
                    with open(label_per_frame, 'a') as f:
                        f.write((' '.join(label)) + '\n')
                        
                    if save:  # Add bbox to image
                        label = names[int(box.cls)]
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

                if (color_det):
                    masks = masks.data.cpu()
                    for seg, box in zip(masks.data.cpu().numpy(), boxes):
                        # r_xywh = scale_coords(img.shape[2:], box.xywh.clone(), im0.shape, kpt_label=False).round()

                        seg = cv2.resize(seg, (w, h))
                        im0, colored_mask = overlay(im0, seg, colors[int(box.cls)], 0.4)

                        
                        # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
                        dominantColors = extractDominantColor(colored_mask,hasThresholding=True)

                        # #Show in the dominant color information
                        # print("Color Information")
                        # prety_print_data(dominantColors)

                        # #Show in the dominant color as bar
                        # print("Color Bar")
                        # colour_bar = plotColorBar(dominantColors)
                        # plt.axis("off")
                        # plt.imshow(colour_bar)
                        # plt.show()

                        red_color_diff = []
                        blue_color_diff = []
                        for dc in dominantColors:
                            color = tuple(map(int,(dc['color'])))
                            color = [round(float(i)/255.0, 2) for i in color]
                            color_srgb = sRGBColor(color[0], color[1], color[2])
                            color_lab = convert_color(color_srgb, LabColor)

                            # Find the color difference
                            red_color_diff.append(delta_e_cie2000(reference_red_lab, color_lab))
                            blue_color_diff.append(delta_e_cie2000(reference_blue_lab, color_lab))

                        delta_red = min(red_color_diff)
                        delta_blue = min(blue_color_diff)

                        player_id = None
                        if delta_red < delta_blue:
                            player_id = 1
                        elif delta_blue < delta_red:
                            player_id = 2

                    pose_db.insert_into_pose_table(
                        frame_num=frame,
                        player_num= player_id,
                        bbox_coords= box.xyxy.tolist(),
                        feet_coords= [int(x_center_below), int(y_center_below)],
                        position=position
                    )

                    if(player_id not in seg_logs['segmentation'][frame]):
                        seg_logs['segmentation'][frame][player_id] = []

                    # Add Tracked Person to Logs
                    player_entry = {
                        "player_id": str(player_id),
                        "bbox_coords": box.xyxy.tolist(),
                        "feet_coords": [int(x_center_below), int(y_center_below)],
                        "position": position
                    }

                    seg_logs['segmentation'][frame][player_id].append(player_entry)

                    label = "player: " + str(player_id)
                    old_label = names[int(box.cls)]

                    plot_one_box(r_xyxy.cpu().numpy()[0], im0, colors[int(box.cls)], f'{label} {float(box.conf):.3}')

                elif track:
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
                        for track_det in outputs:
                            class_name = "person"
                            bbox_xyxy = track_det[:4]
                            identities = track_det[-2]
                            object_id = track_det[-1]
                            categories = class_name

                            player_id = "player_" + str(identities)

                            if(player_id not in seg_logs['segmentation'][frame]):
                                seg_logs['segmentation'][frame][player_id] = []

                            # Add Tracked Person to Logs
                            player_entry = {
                                "player_id": str(identities),
                                "bbox_coords": bbox_xyxy.tolist(),
                                "feet_coords": [int(x_center_below), int(y_center_below)],
                                "position": position
                            }
                            pose_db.insert_into_pose_table(
                                frame_num=frame,
                                player_num= int(identities),
                                bbox_coords= bbox_xyxy.tolist(),
                                feet_coords= [int(x_center_below), int(y_center_below)],
                                position= position
                            )
                            seg_logs['segmentation'][frame][player_id].append(player_entry)

                            # Draw Boxes on Image
                            label = "player: " + str(identities)
                            r_bbox_xyxy = scale_coords(img.shape[2:], torch.Tensor([bbox_xyxy]), im0.shape, kpt_label=False).round()
                            plot_one_box(r_bbox_xyxy.cpu().numpy()[0], im0, colors[int(box.cls)], f'{label}')
                    

        '''Stream results'''
        if view_img:
            cv2.imshow(str('img'), im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        '''Save results (video with detections)'''
        if save:
            if vid_path != save_path:  # new video
                vid_path = save_path
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
            vid_writer.write(im0)
        # vid_cap.release()

    return seg_logs


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
    :param track: track people in videos
    :return: vid_path, txt_path, frames_shot_made, shotmade
    """
    time.sleep(5)
    print("weigths: ", weights)
    ##################################################################################################################################################
    # Parameters
    ##################################################################################################################################################    
    save_img = not dont_save and not source.endswith('.txt')  # save inference images

    weights_name = weights.split('/')[1] 

    '''Directories'''
    save_dir = paths.video_inferred_path / weights_name
    save_txt = paths.bbox_coordinates_path / weights_name
    save_label = paths.labels_path / weights_name

    save_dir.mkdir(parents=True, exist_ok=True)  # create directory
    save_txt.mkdir(parents=True, exist_ok=True)  # create directory
    save_label.mkdir(parents=True, exist_ok=True)  # create directory

    '''Initialize'''
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    basketball_logs = {
        'basketball_detection': {}
    }

    '''Load model'''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    model = TracedModel(model, device, img_size)
    if half:
        model.half()  # to FP16

    '''Set Dataloader'''
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    '''Get names and colors'''
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    '''Run inference'''
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()

    for path, img, im0s, image, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        '''Warmup'''
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

        '''Extract frame attributes'''
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        '''Add empty entry to logs'''
        basketball_logs['basketball_detection'][frame] = [
            {
                "shot": None,
                "bbox_coords": None
            }
        ]
        '''Process detections'''
        for i, det in enumerate(pred):  # detections per image
                
                p = Path(p)  # to Path
                filename = (p.name.replace(" ", "_"))
                save_label_video = Path(save_label / (filename.split('.')[0]))
                save_label_video.mkdir(parents=True, exist_ok=True)  # make dir
                label_per_frame = str(save_label_video / (str(frame) + '.txt'))
                save_path = str(save_dir / (filename.split('.')[0] + "_nethoopbasket_out" + ".mp4"))  # img.jpg
                txt_path = str(save_txt / (filename.split('.')[0] + '.txt'))
                
                cv2.imwrite(str(save_label_video / (str(frame) + ".jpg")), image)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                dets_to_sort = np.empty((0, 6))
                if len(det):
                    '''Remove empty entry'''
                    first_entry = basketball_logs['basketball_detection'][frame][0]
                    if first_entry['shot'] == None:
                        basketball_logs['basketball_detection'][frame].remove(first_entry)

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
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh_label = ' '.join(map(str, ['%.5f' % elem for elem in xywh]))
                        xywh = '\t'.join(map(str, ['%.5f' % elem for elem in xywh]))
                        line = [str(frame), names[int(cls)], xywh, str(round(float(conf), 5))]
                        with open(txt_path, 'a') as f:
                            f.write(('\t'.join(line)) + '\n')
                        label = [str(int(cls)), xywh_label]
                        #if frame % NUMBER_OF_FRAMES_PER_LABEL == 0:
                        with open(label_per_frame, 'a') as f:
                            f.write((' '.join(label)) + '\n')
                            
                        if save_img or view_img:  # Add bbox to image
                            label = names[int(cls)]
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1, kpt_label=False)
                        label = names[int(cls)]
                        
                        frame_entry = {
                            "shot": label,
                            "bbox_coords": [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        }
                        basket_db.insert_into_basket_table(
                            frame_num=frame,
                            shot=label,
                            bbox_coords= [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        )
                        basketball_logs['basketball_detection'][frame].append(frame_entry)

                    dets_to_sort = np.empty((0, 6))
                    # NOTE: We send in detected object class too
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        if detclass == 0.0:
                            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

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
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    return basketball_logs


def detect_actions(weights: str = 'yolov7.pt',
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
    :param track: track people in videos
    :return: vid_path, txt_path, frames_shot_made, shotmade
    """
    time.sleep(5)
    print("weigths: ", weights)
    ##################################################################################################################################################
    # Parameters
    ##################################################################################################################################################
    
    save_img = not dont_save and not source.endswith('.txt')  # save inference images

    weights_name = weights.split('/')[1] 

    '''Directories'''
    save_dir = paths.video_inferred_path / weights_name
    save_txt = paths.bbox_coordinates_path / weights_name
    save_label = paths.labels_path / weights_name

    save_dir.mkdir(parents=True, exist_ok=True)  # create directory
    save_txt.mkdir(parents=True, exist_ok=True)  # create directory
    save_label.mkdir(parents=True, exist_ok=True)  # create directory

    '''Initialize'''
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    actions_logs={
        'action_detection': {}
    }

    '''Load model'''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    model = TracedModel(model, device, img_size)
    if half:
        model.half()  # to FP16

    '''Set Dataloader'''
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    '''Get names and colors'''
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    '''Run inference'''
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()

    for path, img, im0s, image, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        '''Warmup'''
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                model(img, augment=augment)[0]

        '''Inference'''                     
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        '''Apply NMS'''
        pred = non_max_suppression(pred, conf_thresh, iou_thresh)
        t3 = time_synchronized()

        '''Extract frame attributes'''
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        '''Add empty entry to logs'''
        actions_logs['action_detection'][frame] = [
            {
                "action": None,
                "bbox_coords": None
            }
        ]

        '''Process detections'''
        for i, det in enumerate(pred):  # detections per image
                p = Path(p)  # to Path
                filename = (p.name.replace(" ", "_"))
                save_label_video = Path(save_label / (filename.split('.')[0]))
                save_label_video.mkdir(parents=True, exist_ok=True)  # make dir
                label_per_frame = str(save_label_video / (str(frame) + '.txt'))
                save_path = str(save_dir / (filename.split('.')[0] + "_actions_out" + ".mp4"))  # img.jpg
                txt_path = str(save_txt / (filename.split('.')[0] + '.txt'))
                
                cv2.imwrite(str(save_label_video / (str(frame) + ".jpg")), image)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                dets_to_sort = np.empty((0, 6))
                if len(det):
                    '''Remove empty entry'''
                    first_entry = actions_logs['action_detection'][frame][0]
                    if first_entry['action'] == None:
                        actions_logs['action_detection'][frame].remove(first_entry)

                    # print(det[:, :4])
                    # print(type(det[:, :4]))

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
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh_label = ' '.join(map(str, ['%.5f' % elem for elem in xywh]))
                        xywh = '\t'.join(map(str, ['%.5f' % elem for elem in xywh]))
                        line = [str(frame), names[int(cls)], xywh, str(round(float(conf), 5))]
                        with open(txt_path, 'a') as f:
                            f.write(('\t'.join(line)) + '\n')
                        label = [str(int(cls)), xywh_label]
                        
                        with open(label_per_frame, 'a') as f:
                            f.write((' '.join(label)) + '\n')
                            
                        if save_img or view_img:  # Add bbox to image
                            label = names[int(cls)]
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1, kpt_label=False)
                        label = names[int(cls)]

                        frame_entry = {
                            "action": label,
                            "bbox_coords": [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        }
                        action_db.insert_into_action_table(
                            frame_num=frame,
                            action=label,
                            bbox_coords= [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                        )
                        actions_logs['action_detection'][frame].append(frame_entry)

                    dets_to_sort = np.empty((0, 6))
                    # NOTE: We send in detected object class too
                    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                        if detclass == 0.0:
                            dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))


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
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    return actions_logs


def writeToLog(logs_path, logs):
    try:
        with open(logs_path, 'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)
            cur_yaml.update(logs)

        if cur_yaml:
            with open(logs_path,'w') as yamlfile:
                yaml.safe_dump(cur_yaml, yamlfile)
    except:
        with open(logs_path,'w') as yamlfile:
            yaml.safe_dump(logs, yamlfile)

       
def detect_all(game_id: str = ''):
    videoPath, dataLogFilePath = get_game_data(game_id=game_id)
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    # with torch.no_grad():
    action_weights = 'weights/actions_2.pt'
    basket_weights = 'weights/net_hoop_basket_combined_april.pt'
    instance_weights = 'weights/yv8_person_seg.pt'

    '''
        Detect
    '''
    # Actions
    action_db.create_database(game_id)
    action_db.create_action_table()
    actions_logs = detect_actions(weights=action_weights, source=videoPath, dont_save=False)
    strip_optimizer(action_weights)
    action_db.close_db()
    writeToLog(dataLogFilePath, actions_logs)

    # Basketball
    basket_db.create_database(game_id)
    basket_db.create_basket_table()
    basketball_logs = detect_basketball(weights=basket_weights, source=videoPath, dont_save=False)
    strip_optimizer(basket_weights)
    basket_db.close_db()
    writeToLog(dataLogFilePath, basketball_logs)

    # Instance Segmentation
    pose_db.create_database(game_id)
    pose_db.create_pose_table()
    seg_logs = instance_segmentation(weights=instance_weights, source=videoPath, dont_save=False)
    pose_db.close_db()
    writeToLog(dataLogFilePath, seg_logs)


if __name__ == '__main__':
    extract_key_frames(dataset_folder='PhoneDatasetSeven', video_max_len = 10000)
    # detect_all(game_id='IMG_3050_Test')
