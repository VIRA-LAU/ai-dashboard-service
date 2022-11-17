import argparse
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from numpy import random
from models.experimental import attempt_load
from sort import Sort
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer, set_logging, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel
from shared.helper.json_helpers import parse_json
from PIL import Image

"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
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


NUMBER_OF_FRAMES_AFTER_SHOT_MADE = 5


def detect(weights='yolov7.pt',
           source='inference/images',
           img_size=640,
           conf_thresh=0.25,
           iou_thresh=0.45,
           device='',
           view_img=False,
           dont_save=False,
           augment=False,
           trace=True,
           track=True):
    print("weigths: ", weights)
    save_img = not dont_save and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(parse_json("assets/paths.json")["videos_inferred_path"])
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    save_txt = Path(parse_json("assets/paths.json")["bbox_coordinates_path"])
    save_txt.mkdir(parents=True, exist_ok=True)  # make dir
    save_label = Path(parse_json("assets/paths.json")["label_path"])
    save_label.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    sort_tracker = Sort(max_age=5,
                        min_hits=2,
                        iou_threshold=0.2)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()

    # madebasketball counter
    shotmade = 0
    history = []

    for _ in range(NUMBER_OF_FRAMES_AFTER_SHOT_MADE):
        history.append(False)
    for path, img, im0s, image, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thresh, iou_thresh)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            filename = (p.name.replace(" ", "_"))
            save_label_video = Path(Path(parse_json("assets/paths.json")["label_path"]) / (filename.split('.')[0]))
            save_label_video.mkdir(parents=True, exist_ok=True)  # make dir
            cv2.imwrite(str(save_label_video / (str(frame) + ".jpg")), image)
            label_per_frame = str(save_label_video / (str(frame) + '.txt'))
            save_path = str(save_dir / (filename.split('.')[0] + "-out" + ".mp4"))  # img.jpg
            txt_path = str(save_txt / (filename.split('.')[0] + '.txt'))

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results

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
                    cv2.putText(im0, f'Shots Made: {shotmade}', (25, 25), 0, 1, [0, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    if names[int(cls)] == "madebasketball":
                        if any(history[-NUMBER_OF_FRAMES_AFTER_SHOT_MADE:]):
                            history.append(False)
                        else:
                            shotmade += 1
                            history.append(True)

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                dets_to_sort = np.empty((0, 6))
                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    if detclass == 0.0:
                        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

                # Perform Tracking
            if track:
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()

                # draw boxes for visualization
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    confidences = None

                    # loop over tracks
                    # for t, track in enumerate(tracks):
                    #     track_color = colors[int(track.detclass)]
                    #
                    #     [cv2.line(im0, (int(track.centroidarr[i][0]),
                    #                     int(track.centroidarr[i][1])),
                    #                     (int(track.centroidarr[i + 1][0]),
                    #                     int(track.centroidarr[i + 1][1])),
                    #                     track_color, thickness=round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1)
                    #                     for i, _ in enumerate(track.centroidarr)
                    #                         if i < len(track.centroidarr) - 1]
                else:
                    bbox_xyxy = dets_to_sort[:, :4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]

                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
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
    print(f'Total Made Shots: {shotmade}')

    return vid_path, txt_path


if __name__ == '__main__':
    weights = 'yolo_v7_model/weights/best.pt'
    source = 'datasets/videos_input/57 - Copy.avi'

    with torch.no_grad():
        video_path = detect(weights=weights, source=source)
        strip_optimizer(weights)
    print(video_path)
