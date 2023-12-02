from pathlib import Path
import time
import random

import numpy as np

import torch
import cv2

from models.experimental import attempt_load

from utils.datasets import LoadImages
from utils.general import set_logging, non_max_suppression, scale_coords, xyxy2xywh, check_img_size
from utils.torch_utils import time_synchronized, select_device, TracedModel
from utils.plots import plot_one_box

from persistence.repositories import paths

import utils.handle_db.basket_db_handler as action_db

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