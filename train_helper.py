import argparse
import logging
import os
from pathlib import Path

import torch.distributed as dist
import torch.utils.data
import yaml
from torch.utils.tensorboard import SummaryWriter

from utils.general import increment_path, get_latest_run, check_file, set_logging, colorstr
from utils.wandb_logging.wandb_utils import check_wandb_resume
from utils.torch_utils import select_device

from utils.args import *

from train.train_detection_handler import *
from train.train_segmentation_handler import *

logger = logging.getLogger(__name__)

def run_train_detection(weights: str = 'yolov7.pt',
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
                    generations: int = 300,
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
    
    opt = parse_detection(weights = weights,
                    cfg = cfg,
                    data = data,
                    hyp = hyp,
                    epochs = epochs,
                    batch_size = batch_size,
                    img_size = img_size,
                    augment = augment,
                    rect = rect,
                    resume = resume,
                    nosave = nosave,
                    notest  = notest,
                    noautoanchor = noautoanchor,
                    evolve = evolve,
                    generations = generations,
                    bucket = bucket,
                    cache_images = cache_images,
                    image_weights = image_weights,
                    device = device,
                    multi_scale = multi_scale,
                    single_cls = single_cls,
                    adam = adam,
                    sync_bn = sync_bn,
                    local_rank = local_rank,
                    workers = workers,
                    project = project,
                    entity = entity,
                    name = name,
                    exist_ok = exist_ok,
                    quad = quad,
                    linear_lr = linear_lr,
                    label_smoothing = label_smoothing,
                    upload_dataset = upload_dataset,
                    bbox_interval = bbox_interval,
                    save_period = save_period,
                    artifact_alias = artifact_alias,
                    freeze = freeze,
                    v5_metric = v5_metric)

    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train_detection(hyp, opt, device, tb_writer)
    else:
        train_detection_evolve(hyp, opt, device)


def run_auto_annotator_segmentation():
    auto_annotate_segmentation()


def run_train_segmentation():
    train_segmentation(weights='weights/yv8_person_seg.pt', data='datasets/training_datasets/instance_segmentation/PhoneDatasetThree_seg/data.yaml')
    return


if __name__ == "__main__":
    print(torch.cuda.is_available())

    '''Weights Location'''
    actions_weight = 'weights/yolov7.pt'
    netbasket_weight = 'weights/netbasket_3_5.pt'

    '''Dataset Location'''
    actions_dataset_path = 'datasets/training_datasets/object_detection/actions/VIP_Demo/data.yaml'
    netbasket_dataset_path = 'datasets/training_datasets/object_detection/netbasket/VIP_Demo/data.yaml'

    '''Hyperparameters Location'''
    hyp = 'train/cfg/object_detection/hyperparameters/hyp.scratch.p6.yaml'

    '''Run Training'''
    # run_train_detection(weights=actions_weight, device='0', data=actions_dataset_path, evolve=False, generations=30, epochs=100, batch_size=8, name='actions_VIP_Demo', exist_ok=True, hyp=hyp)
    run_train_detection(weights=netbasket_weight, device='0', data=netbasket_dataset_path, evolve=False, generations=30, epochs=100, batch_size=8, name='netbasket_VIP_Demo', exist_ok=True, hyp=hyp)
    # run_train_segmentation()
