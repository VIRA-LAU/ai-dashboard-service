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


def parse_detection(weights: str = 'yolov7.pt',
                    cfg: str = '',
                    data: str = 'data/coco.yaml',
                    hyp: str = 'data/hyp.scratch.p5.yaml',
                    epochs: int = 300,
                    batch_size: int = 16,
                    img_size: int = 640,
                    augment: bool = True,
                    rect: bool = False,
                    resume: bool = False,
                    nosave: bool = False,
                    notest: bool = False,
                    noautoanchor: bool = False,
                    evolve: bool = False,
                    bucket: str = '',
                    cache_images: bool = False,
                    image_weights: bool = False,
                    device: str = '',
                    multi_scale: bool = False,
                    single_cls: bool = False,
                    adam: bool = False,
                    sync_bn: bool = False,
                    local_rank: int = -1,
                    workers: int = 8,
                    project: str = 'runs/train',
                    entity: str = None,
                    name: str = 'exp',
                    exist_ok: bool = False,
                    quad: bool = False,
                    linear_lr: bool = False,
                    label_smoothing: float = 0.0,
                    upload_dataset: bool = False,
                    bbox_interval: int = -1,
                    save_period: int = -1,
                    artifact_alias: str = "latest",
                    freeze: list = [0],
                    v5_metric: bool = False):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=weights, help='initial weights path')
    parser.add_argument('--cfg', type=str, default=cfg, help='model.yaml path')
    parser.add_argument('--data', type=str, default=data, help='data.yaml path')
    parser.add_argument('--hyp', type=str, default=hyp, help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--batch-size', type=int, default=batch_size, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[img_size, img_size], help='[train, test] image sizes')
    parser.add_argument('--augment', type=bool, default=augment, help='augment images when loading dataset')
    parser.add_argument('--rect', action='store_true', default=rect, help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=resume, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', default=nosave, help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', default=notest, help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', default=noautoanchor, help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', default=evolve, help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default=bucket, help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', default=cache_images, help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', default=image_weights, help='use weighted image selection for training')
    parser.add_argument('--device', default=device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', default=multi_scale, help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', default=single_cls, help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', default=adam, help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', default=sync_bn, help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=local_rank, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=workers, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=project, help='save to project/name')
    parser.add_argument('--entity', default=entity, help='W&B entity')
    parser.add_argument('--name', default=name, help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=exist_ok, help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', default=quad, help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', default=linear_lr, help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=label_smoothing, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', default=upload_dataset, help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=bbox_interval, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=save_period, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default=artifact_alias, help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=freeze, help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', default=v5_metric, help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    return opt