import argparse
from pathlib import Path

import torch

def parse_segmentation(weights: str = 'yolov8.pt',
                        reid_model: Path = '',
                        tracking_method: str = 'deepocsort',
                        source: str='',
                        conf: float=0.6,
                        iou: float=0.45,
                        show: bool=True,
                        img_size: int=640,
                        device= torch.device("cuda:0"),
                        half: bool=True,
                        show_conf: bool=False,
                        save_txt: bool=False,
                        show_labels: bool=True,
                        save: bool=True,
                        save_mot: bool=False,
                        save_id_crops: bool=True,
                        verbose: bool=True,
                        exist_ok: bool=False,
                        save_dir: str = 'datasets/videos_inferred',
                        name: str = '',
                        classes: int = None,
                        per_class: bool = False,
                        vid_stride: int = 1,
                        line_width: int = 3):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=weights,
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=reid_model,
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default=tracking_method,
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default=source,
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[img_size],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=conf,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=iou,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default=device,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', default=show,
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true', default=save,
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=classes,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=save_dir,
                        help='save results to project/name')
    parser.add_argument('--name', default=name,
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=exist_ok,
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', default=half,
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=vid_stride,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false', default=show_labels,
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false', default=show_conf,
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true', default=save_txt,
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true', default=save_id_crops,
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true', default=save_mot,
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=line_width, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=per_class, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=verbose, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=vid_stride, type=int,
                        help='video frame-rate stride')

    opt = parser.parse_args()
    return opt
