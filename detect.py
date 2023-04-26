import time
from pathlib import Path
import sys
import os
from multiprocessing import Pool
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from numpy import random

from GPUtil import showUtilization as gpu_usage
from numba import cuda

from torchvision import transforms

from models.experimental import attempt_load
from sort import Sort
from utils.datasets import LoadImages,letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, set_logging, xyxy2xywh, non_max_suppression_kpt
from utils.plots import plot_one_box, output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt
from utils.torch_utils import select_device, time_synchronized, TracedModel
from utils.TubeDETR import stvg
from persistence.repositories import paths


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
    """
    Function to Draw Bounding boxes when tracking
    :param img:
    :param bbox:
    :param identities:
    :param categories:
    :param confidences:
    :param names:
    :param colors:
    :return: image
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def get_stats(weights: str = [],
              source: str = 'vid'):
    
    basket_logs = paths.logs_path / weights[0].split('/')[1] 
    pose_logs = paths.logs_path / weights[1].split('/')[1] 
    actions_logs = paths.logs_path / weights[2].split('/')[1]

    df_basket_logs = pd.read_csv(str(basket_logs) + "/" + str(source.split('.')[0]) + "_logs.csv") 
    df_pose_logs = pd.read_csv(str(pose_logs) + "/" + str(source.split('.')[0]) + "_logs.csv") 
    df_actions_logs = pd.read_csv(str(actions_logs) + "/" + str(source.split('.')[0]) + "_logs.csv") 

    df_shooting = df_actions_logs.loc[df_actions_logs['labels']=='shooting']
    df_points = df_pose_logs.loc[df_shooting['frame']]

    '''
    Compare dates against a margin
    remove all occurances inside this margin
    continue for the new margin
    1 occurance for each margin
    '''

    # df_shooting.to_csv("datasets/logs/" + str(weights[2].split('/')[1]) + "/" + str(source.split('.')[0]) + "_logs.csv")

    print(df_points)

    return


def detect(weights: str = 'yolov7.pt',
           source: str = 'inference/images',
           img_size: int = 640,
           conf_thresh: float = 0.25,
           iou_thresh: float = 0.45,
           device: str = '',
           view_img: bool = False,
           dont_save: bool = False,
           augment: bool = False,
           track: bool = True,
           sampling: bool = False,
           temporal: bool = False,
           shots: bool = False,
           pose_est: bool = False) -> tuple:
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
    # print(source)
    ##################################################################################################################################################
    # Parameters
    ##################################################################################################################################################
    '''Variables for counting the shots made'''
    if (shots):
        frames_shot_made: list = []
        NUMBER_OF_FRAMES_AFTER_SHOT_MADE = 5
        shotmade = 0
        history = []
        for _ in range(NUMBER_OF_FRAMES_AFTER_SHOT_MADE):
            history.append(False)
    
    '''Sampling rate to generate labels'''
    if(sampling):
        NUMBER_OF_FRAMES_PER_LABEL = 20
    
    save_img = not dont_save and not source.endswith('.txt')  # save inference images

    weights_name = weights.split('/')[1] 

    '''Directories'''
    save_dir = paths.video_inferred_path / weights_name
    save_txt = paths.bbox_coordinates_path / weights_name
    save_label = paths.labels_path / weights_name
    save_logs = paths.logs_path / weights_name
    save_dir.mkdir(parents=True, exist_ok=True)  # create directory
    save_txt.mkdir(parents=True, exist_ok=True)  # create directory
    save_label.mkdir(parents=True, exist_ok=True)  # create directory
    save_logs.mkdir(parents=True, exist_ok=True)  # create directory

    '''Logs'''
    df_log = pd.DataFrame(columns=['timestamp', 'frame', 'video', 'labels'])

    '''Temporal Analysis'''
    if(temporal):
        temporal_model='utils/TubeDETR/models/checkpoints/res352/vidstg_k4.pth'
        out_dir = 'datasets/videos_input/'
        out_dir_frames = 'datasets/videos_input_frames/'
        source = stvg.analyze(source,temporal_model,out_dir,out_dir_frames)

    '''Initialize'''
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    sort_tracker = Sort(max_age=5,
                        min_hits=2,
                        iou_threshold=0.2)

    '''Load model'''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    if (not pose_est):
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
        if(pose_est):
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            t2 = time_synchronized()

            pred = non_max_suppression_kpt(pred, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            t3 = time_synchronized()
            
            output = output_to_keypoint(pred)

        else:                      
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            t2 = time_synchronized()

            '''Apply NMS'''
            pred = non_max_suppression(pred, conf_thresh, iou_thresh)
            t3 = time_synchronized()

        '''Dummy Equation of Arc'''
        arcUp = [-0.0105011, 0.0977268, -0.308306, 0.315377, -0.229249, 2.11325]
        arcDown = [0.000527976, 0.00386626, 0.0291599, 0.121282, -2.22398]

        polyUp = np.poly1d(arcUp)
        polyDown = np.poly1d(arcDown)

        '''Process detections'''
        for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                filename = (p.name.replace(" ", "_"))
                save_label_video = Path(save_label / (filename.split('.')[0]))
                save_label_video.mkdir(parents=True, exist_ok=True)  # make dir
                label_per_frame = str(save_label_video / (str(frame) + '.txt'))
                save_path = str(save_dir / (filename.split('.')[0] + "-out" + ".mp4"))  # img.jpg
                txt_path = str(save_txt / (filename.split('.')[0] + '.txt'))
                #if frame % NUMBER_OF_FRAMES_PER_LABEL == 0:
                cv2.imwrite(str(save_label_video / (str(frame) + ".jpg")), image)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                dets_to_sort = np.empty((0, 6))
                if len(det):
                    '''Rescale boxes from img_size to im0 size'''
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False).round()
                    if (pose_est): # Rescale keypoints to original image size
                        det_norm = det[:, 6:]
                        det[:, 6:] = scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)

                    '''Print results'''
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    '''Write results

                            pose_est: get bounding boxes and keypoints
                                    keypoints: det[:, 6:]

                            not pose_est: get bounding boxes

                            bounding box: xyxy (x1 y1 x2 y2)
                                    convert xyxy --> xywh : x1 y1 width height
                    '''
                    if (pose_est):
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            xywh_label = ' '.join(map(str, ['%.5f' % elem for elem in xywh]))
                            xywh = '\t'.join(map(str, ['%.5f' % elem for elem in xywh]))
                            line = [str(frame), names[int(cls)], xywh, str(round(float(conf), 5))]

                            # kpts_norm = det_norm[det_index, 6:]
                            kpts = det[det_index, 6:]
                            with open(txt_path, 'a') as f:
                                f.write(('\t'.join(line)) + '\n')

                            label = [str(int(cls)), xywh_label]
                            with open(label_per_frame, 'a') as f:
                                f.write((' '.join(label)) + '\n') 

                            if save_img or view_img:
                                label =  f'{names[int(cls)]} {conf:.2f}'
                                output = output_to_keypoint(pred)
                                
                                # print(output[det_index, 7:].T[-2]) # foot kpt Y Coord
                                # print(output[det_index, 7:].T[-3]) # foot kpt X Coord
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], 
                                             line_thickness=3, kpt_label=True, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                            
                            xy=[]
                            steps=3
                            num_kpts = len(kpts) // steps

                            for kid in range(num_kpts):
                                x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
                                if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                                    xy.append([int(x_coord), int(y_coord)])

                            # ratio set according to scale of axis
                            ratio_x = (image.shape[1])/20 
                            ratio_y = (image.shape[0])/5.8

                            x_center = (image.shape[1]/2)/ratio_x   # width
                            y_center = (image.shape[0]/2)/ratio_y   # height

                            x_new = (xy[-1][0])/ratio_x - x_center
                            y_new = (xy[-1][1])/ratio_y - y_center

                            yUp = polyUp(x_new)
                            yDown = polyDown(x_new)

                            position=""
                            if yUp > y_new and yDown < y_new:
                                position = "2_points"
                            else:
                                position = "3_points"


                            label = names[int(cls)]
                            data = { 'timestamp' : datetime.now(),
                                'frame': str(frame),
                                'video': filename.split('.')[0],
                                'labels': label,
                                'person' : "ID",
                                'feet_coord' : xy[-1],
                                'position' : position
                                }
                            
                            df_log = df_log.append(data, ignore_index = True)

                    else:
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
                    
                            if (shots):
                                cv2.putText(im0, f'Shots Made: {shotmade}', (25, 25), 0, 1, [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                                if names[int(cls)] == "netbasket":
                                    if any(history[-NUMBER_OF_FRAMES_AFTER_SHOT_MADE:]):
                                        history.append(False)
                                    else:
                                        shotmade += 1
                                        frames_shot_made.append(frame)
                                        history.append(True)
                                
                            if save_img or view_img:  # Add bbox to image
                                label = names[int(cls)]
                                print(label)
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1, kpt_label=False)

                            label = names[int(cls)]
                            data = {'timestamp' : datetime.now(),
                                'frame': str(frame),
                                'video': filename.split('.')[0],
                                'labels': label,
                                'bbox_coord' : xywh}
                            
                            df_log = df_log.append(data, ignore_index = True)

                        dets_to_sort = np.empty((0, 6))
                        # NOTE: We send in detected object class too
                        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                            if detclass == 0.0:
                                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Add frame with no detections in logs
                else:
                    data = {'timestamp' : datetime.now(),
                        'frame': str(frame),
                        'video': filename.split('.')[0],
                        'labels': ''}
                    
                    df_log = df_log.append(data, ignore_index = True)

                '''Perform Tracking'''
                if track:
                    tracked_dets = sort_tracker.update(dets_to_sort)
                    tracks = sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        '''loop over tracks'''
                        for t, track in enumerate(tracks):
                            track_color = colors[int(track.detclass)]

                            [cv2.line(im0, (int(track.centroidarr[i][0]),
                                            int(track.centroidarr[i][1])),
                                    (int(track.centroidarr[i + 1][0]),
                                    int(track.centroidarr[i + 1][1])),
                                    track_color, thickness=round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
                            for i, _ in enumerate(track.centroidarr)
                            if i < len(track.centroidarr) - 1]
                    else:
                        bbox_xyxy = dets_to_sort[:, :4]
                        identities = None
                        categories = dets_to_sort[:, 5]
                        confidences = dets_to_sort[:, 4]
                    im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)

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

                    df_log.to_csv("datasets/logs/" + weights_name + "/" + filename.split('.')[0] + "_logs.csv")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    if (shots):
        print(f'Total Made Shots: {shotmade}')
        return vid_path, txt_path, frames_shot_made, shotmade
    else:
        return vid_path, txt_path
        

def run_detect(data):
    weights = data[0]
    source = data[1]
    shots = data[2]
    pose_est = data[3]
    for vid in os.listdir(source):
        with torch.no_grad():
            video_path = detect(weights=weights, source=source + str(vid), shots=shots, pose_est=pose_est)
            strip_optimizer(weights)
        print(video_path)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    weights = ['weights/net_hoop_basket_2.pt','weights/yolov7-w6-pose.pt', 'weights/actions_1.pt']
    source = 'datasets/videos_input/'
    
    for vid in os.listdir(source):
        torch.cuda.empty_cache()
        with torch.no_grad():
            video_path = detect(weights='weights/yolov7-w6-pose.pt', source=source + str(vid), shots=False, pose_est=True, temporal=False)
            # strip_optimizer(weights)
        print(video_path)

    '''Get stats after processing'''
    # for vid in os.listdir(source):
    #     get_stats(weights=weights, source=str(vid))
        

    '''For parallel processing: '''

    # data = [[weights[0],source,True,False],[
    #     weights[1],source,False,True],
    #     [weights[2],source,False,False]]

    # pool = Pool(3)
    # # run_detect(data)
    # pool.map(run_detect, data)

    
