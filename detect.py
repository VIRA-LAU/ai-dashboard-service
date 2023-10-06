from pathlib import Path

import torch
import yaml

from bridge_wrapper import *
from detection_helpers import *
from dev_utils.paths.game import get_game_data
from persistence.repositories import paths
from tracking_helpers import *
from utils.TubeDETR import stvg
from utils.datasets import LoadImages
from utils.general import strip_optimizer, set_logging, non_max_suppression_kpt
from utils.google_utils import gdrive_download
from utils.plots import output_to_keypoint, plot_kpts
from utils.torch_utils import time_synchronized
import dev_utils.handle_db.action_db_handler as action_db
import dev_utils.handle_db.basket_db_handler as basket_db
import dev_utils.handle_db.pose_db_handler as pose_db


def draw_boxes_tracked(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
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
    x1, y1, x2, y2 = [int(i) for i in bbox]
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # cat = int(categories[i]) if categories is not None else 0
    cat = 0
    id = int(identities) if identities is not None else 0
    conf = confidences if confidences is not None else 0
    color = colors[cat]

    cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

    id = float(str(id) + '.0')
    tf = max(tl - 1, 1)  # font thickness
    label = str(id)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = x1 + t_size[0], y1 - t_size[1] - 3
    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def extract_key_frames(dataset_folder: str = "", video_max_len: int = 200, video_res: int = 224, device: str = "cuda"):
    '''
        Temporal Analysis
    '''
    if len(os.listdir('utils/TubeDETR/models/checkpoints/res352/')) == 0:
        ckpt = gdrive_download(id='1GqYjnad42-fri1lxSmT0vFWwYez6_iOv', file='utils/TubeDETR/models/checkpoints/res352/vidstg_k4.pth')

    temporal_model='utils/TubeDETR/models/checkpoints/res352/vidstg_k4.pth' # model that performs temporal analysis
    out_dir = paths.video_input_path # output video with less frames
    out_dir_frames = paths.temporal_frames  # output frames

    source = os.path.join(paths.temporal_videos_input_path, dataset_folder)
    for video in os.listdir(source):
        vid_path = os.path.join(source, video)
        stvg.analyze(vid_path, video_max_len, video_res, temporal_model, device, out_dir, out_dir_frames)


def detect_pose(weights: str = 'yolov7.pt',
           source: str = 'inference/images',
           img_size: int = 640,
           conf_thresh: float = 0.6,
           iou_thresh: float = 0.45,
           device: str = '',
           view_img: bool = False,
           dont_save: bool = False,
           augment: bool = False,
           track: bool = True) -> tuple:
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
    :return: vid_path, txt_path
    """
    time.sleep(5)
    print("weigths: ", weights)
    # print(source)
    ##################################################################################################################################################
    # Parameters
    ##################################################################################################################################################
    
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
    # save_logs.mkdir(parents=True, exist_ok=True)  # create directory

    '''Initialize'''
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    pose_logs={
        'pose_detection': {}
    }

    '''Initialize Tracker'''
    detector = Detector(classes = [0])
    detector.load_model(weights) 
    wrapper = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
    tracker = wrapper.tracker

    encoder = wrapper.encoder

    '''Load model'''
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
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
        pred = non_max_suppression_kpt(pred, conf_thresh, iou_thresh, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
        t3 = time_synchronized()
        
        output = output_to_keypoint(pred)

        '''Dummy Equation of Arc'''
        arcUp = [-0.0105011, 0.0977268, -0.308306, 0.315377, -0.229249, 2.11325]
        arcDown = [0.000527976, 0.00386626, 0.0291599, 0.121282, -2.22398]

        polyUp = np.poly1d(arcUp)
        polyDown = np.poly1d(arcDown)

        '''Extract frame attributes'''
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        '''Add empty entry to logs'''
        pose_logs['pose_detection'][frame] = {}

        '''Process detections'''
        for i, det in enumerate(pred):  # detections per image
                p = Path(p)  # to Path
                filename = (p.name.replace(" ", "_"))
                save_label_video = Path(save_label / (filename.split('.')[0]))
                save_label_video.mkdir(parents=True, exist_ok=True)  # make dir
                label_per_frame = str(save_label_video / (str(frame) + '.txt'))
                save_path = str(save_dir / (filename.split('.')[0] + "_pose_out" + ".mp4"))  # img.jpg
                txt_path = str(save_txt / (filename.split('.')[0] + '.txt'))
                #if frame % NUMBER_OF_FRAMES_PER_LABEL == 0:
                cv2.imwrite(str(save_label_video / (str(frame) + ".jpg")), image)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                dets_to_sort = np.empty((0, 6))
                if len(det):
                    # '''Remove empty entry'''
                    # pose_logs['pose_detection'][frame].remove(pose_logs['pose_detection'][frame][0])

                    '''Rescale boxes from img_size to im0 size'''
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False).round()
                    # Rescale keypoints to original image size
                    det[:, 6:] = scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=True, step=3)

                    '''Print results'''
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    '''Write results
                            get bounding boxes and keypoints
                                    keypoints: det[:, 6:]

                            bounding box: xyxy (x1 y1 x2 y2)
                                    convert xyxy --> xywh : x1 y1 width height
                    '''

                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh_label = ' '.join(map(str, ['%.5f' % elem for elem in xywh]))
                        xywh = '\t'.join(map(str, ['%.5f' % elem for elem in xywh]))
                        line = [str(frame), names[int(cls)], xywh, str(round(float(conf), 5))]

                        kpts = det[det_index, 6:]
                        with open(txt_path, 'a') as f:
                            f.write(('\t'.join(line)) + '\n')

                        label = [str(int(cls)), xywh_label]
                        with open(label_per_frame, 'a') as f:
                            f.write((' '.join(label)) + '\n') 

                        label = names[int(cls)]
                        
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

                        plot_kpts(im0, kpts=kpts, steps=3, orig_shape=im0.shape[:2])

                    # NOTE: We send in detected object class too
                        for element in det.cpu().detach().numpy():
                            if element[5] == 0.0:
                                dets_to_sort = np.vstack((dets_to_sort, np.array([element[0], element[1], element[2], element[3], element[4], element[5]])))

                    if track:
                        yolo_dets = det

                        if yolo_dets is None:
                            bboxes = []
                            scores = []
                            classes = []
                            num_objects = 0
                        
                        else:
                            bboxes = yolo_dets[:,:4]
                            bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                            scores = yolo_dets[:,4]
                            classes = yolo_dets[:,-1]
                            num_objects = bboxes.shape[0]

                        names = []
                        for i in range(num_objects):
                            # class_indx = int(classes[i])
                            class_name = "person"
                            names.append(class_name)

                        names = np.array(names)
                        count = len(names)

                        features = encoder(im0, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
                        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                        boxes = np.array([d.tlwh for d in detections])  # run non-maxima supression below
                        scores = np.array([d.confidence for d in detections])
                        classes = np.array([d.class_name for d in detections])
                        indices = preprocessing.non_max_suppression(boxes, classes, 1.0, scores)
                        detections = [detections[i] for i in indices]  

                        tracker.predict()
                        tracker.update(detections)
                        tracked_dets = tracker.tracks

                        # Check if there are tracked persons
                        if len(tracked_dets) > 0:
                            for track in tracked_dets:
                                identities = int(track.track_id)
                                bbox_xyxy = track.to_tlbr()     # Get current position in bounding box format (min x, miny, max x, max y)
                                class_name = track.get_class()
                                categories = class_name
                                confidences = None

                                player_id = "player_" + str(identities)

                                if(player_id not in pose_logs['pose_detection'][frame]):
                                    pose_logs['pose_detection'][frame][player_id] = []

                                # Add Tracked Person to Logs
                                player_entry = {
                                    "player_id": str(identities),
                                    "bbox_coords": bbox_xyxy.tolist(),
                                    "feet_coords": list(xy[-1]),
                                    "position": position
                                }
                                pose_db.insert_into_pose_table(
                                    frame_num=frame,
                                    player_num= int(identities),
                                    bbox_coords= bbox_xyxy.tolist(),
                                    feet_coords= list(xy[-1]),
                                    position= position
                                )
                                pose_logs['pose_detection'][frame][player_id].append(player_entry)

                                # Draw Boxes on Image
                                im0 = draw_boxes_tracked(im0, bbox_xyxy, identities, categories, confidences, names, colors)
                            

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

    return pose_logs


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
    # print(source)
    ##################################################################################################################################################
    # Parameters
    ##################################################################################################################################################
    '''Variables for counting the shots made'''
    frames_shot_made: list = []
    NUMBER_OF_FRAMES_AFTER_SHOT_MADE = 10
    shotmade = 0
    history = []
    for _ in range(NUMBER_OF_FRAMES_AFTER_SHOT_MADE):
        history.append(False)
    
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
    # save_logs.mkdir(parents=True, exist_ok=True)  # create directory

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
                #if frame % NUMBER_OF_FRAMES_PER_LABEL == 0:
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
    
    print(f'Total Made Shots: {shotmade}')
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
    # print(source)
    ##################################################################################################################################################
    # Parameters
    ##################################################################################################################################################
    
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
    # save_logs.mkdir(parents=True, exist_ok=True)  # create directory

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


def set_max_split_size_mb(model, max_split_size_mb):
    """
    Set the max_split_size_mb parameter in PyTorch to avoid fragmentation.

    Args:
        model (torch.nn.Module): The PyTorch model.
        max_split_size_mb (int): The desired value for max_split_size_mb in megabytes.
    """
    for param in model.parameters():
        param.requires_grad = False  # Disable gradient calculation to prevent unnecessary memory allocations

    # Dummy forward pass to initialize the memory allocator
    dummy_input = torch.randn(1, 1)
    model(dummy_input)

    # Get the current memory allocator state
    allocator = torch.cuda.memory._get_memory_allocator()

    # Update max_split_size_mb in the memory allocator
    allocator.set_max_split_size(max_split_size_mb * 1024 * 1024)

    for param in model.parameters():
        param.requires_grad = True 

       
def detect_all(game_id: str = ''):
    videoPath, dataLogFilePath = get_game_data(game_id=game_id)
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    with torch.no_grad():
        action_weights = 'weights/actions_2.pt'
        basket_weights = 'weights/net_hoop_basket_combined_april.pt'
        pose_weights = 'weights/yolov7-w6-pose.pt'

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

        # Pose
        pose_db.create_database(game_id)
        pose_db.create_pose_table()
        pose_logs = detect_pose(weights=pose_weights, source=videoPath, dont_save=False)
        strip_optimizer(pose_weights)
        pose_db.close_db()
        writeToLog(dataLogFilePath, pose_logs)


if __name__ == '__main__':
    # extract_key_frames(dataset_folder='HardwareDatasetOne', video_max_len = 500, video_res = 128, device = "cpu")
    
    detect_all(game_id='04183')

    
