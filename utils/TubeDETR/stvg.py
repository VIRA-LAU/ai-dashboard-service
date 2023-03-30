import torch
import sys
import os
import json
import random
import numpy as np
import ffmpeg
import argparse
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import utils.TubeDETR.util.dist as dist
import utils.TubeDETR.util.misc as utils
from utils.TubeDETR.models import build_model
from utils.TubeDETR.models.transformer import build_transformer
from utils.TubeDETR.models.backbone import build_backbone
from utils.TubeDETR.models.tubedetr import TubeDETR
from utils.TubeDETR.models.postprocessors import PostProcessSTVG, PostProcess
from utils.TubeDETR.datasets.video_transforms import prepare, make_video_transforms
from utils.TubeDETR.util.misc import NestedTensor

def get_args_parser(vid,model,out_dir,out_dir_frames):
    parser = argparse.ArgumentParser("Set TubeDETR", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Model parameters
    parser.add_argument(
        "--freeze_text_encoder",
        action="store_true",
        help="Whether to freeze the weights of the text encoder",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Whether to freeze the weights of the visual encoder",
    )
    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--num_queries",
        default=1,
        type=int,
        help="Number of object query slots per image",
    )
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # Run specific
    parser.add_argument(
        "--output-dir", default=out_dir, help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--output-dir-frames", default=out_dir_frames, help="path where to save frames, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    parser.add_argument(
        "--load",
        default=model,
        help="resume from checkpoint",
    )

    # Baselines
    parser.add_argument(
        "--no_fast",
        dest="fast",
        action="store_false",
        help="whether to use the fast branch in the encoder",
    )
    parser.add_argument(
        "--learn_time_embed",
        action="store_true",
        help="whether to learn time embeddings or use frozen sinusoidal ones",
    )
    parser.add_argument(
        "--no_time_embed",
        action="store_true",
        help="whether to deactivate the time encodings or not",
    )
    parser.add_argument(
        "--no_tsa",
        action="store_true",
        help="whether to deactivate the temporal self-attention in the decoder",
    )
    parser.add_argument(
        "--rd_init_tsa",
        action="store_true",
        help="whether to randomly initialize the temporal self-attention in the decoder",
    )
    parser.add_argument(
        "--fast_mode",
        type=str,
        default="",
        choices=["", "gating", "transformer", "pool", "noslow"],
        help="alternative implementations for the fast and aggregation modules",
    )
    parser.add_argument(
        "--caption", default="", type=str, help="caption example for STVG demo"
    )
    parser.add_argument(
        "--video",
        default="",
        type=str,
        help="path to a video example for STVG demo",
    )
    parser.add_argument("--stride", type=int, default=5, help="temporal stride k")
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="number of frames per second extracted from videos",
    )
    parser.add_argument(
        "--no_tmp_crop",
        dest="tmp_crop",
        action="store_false",
        help="whether to use random temporal cropping during training",
    )
    # Video parameters
    parser.add_argument(
        "--resolution", type=int, default=224, help="spatial resolution of the images"
    )
    parser.add_argument(
        "--video_max_len",
        type=int,
        default=200,
        help="maximum number of frames for a video",
    )

    '''Other Args'''

    parser.add_argument(
        "--no_guided_attn",
        dest="guided_attn",
        action="store_false",
        help="whether to use the guided attention loss",
    )
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=1,
        help="standard deviation for the quantized gaussian law used for the kullback leibler divergence loss",
    )
    parser.add_argument(
        "--no_sted",
        dest="sted",
        action="store_false",
        help="whether to use start end KL loss",
    )
    # Loss coefficients
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--sted_loss_coef", default=10, type=float)
    parser.add_argument("--guided_attn_loss_coef", default=1, type=float)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=False)
    parser.add_argument(
        "--combine_datasets",
        nargs="+",
        help="List of datasets to combine for training",
        required=False,
    )
    parser.add_argument(
        "--combine_datasets_val",
        nargs="+",
        help="List of datasets to combine for eval",
        required=False,
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="whether to use the second version of HC-STVG or not",
    )
    parser.add_argument(
        "--tb_dir", type=str, default="", help="eventual path to tensorboard directory"
    )

    # Training hyper-parameters
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr_drop", default=10, type=int)
    parser.add_argument(
        "--epoch_chunks",
        default=-1,
        type=int,
        help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk",
    )
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" epochs',
    )

    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument(
        "--fraction_warmup_steps",
        default=0.01,
        type=float,
        help="Fraction of total number of steps",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    return parser

def analyze(vid,model,out_dir,out_dir_frames):
    # args
    parser = argparse.ArgumentParser(
        "TubeDETR training and evaluation script", parents=[get_args_parser(vid,model,out_dir,out_dir_frames)]
    )
    args = parser.parse_args()
    device = args.device

 # Init distributed mode
    dist.init_distributed_mode(args)
    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.set_deterministic(True)
    torch.use_deterministic_algorithms(True)

    # Build the model
    model, criterion, weight_dict = build_model(args)
    model.to(device)

    # Get a copy of the model for exponential moving averaged version of the model
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Set up optimizers
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "text_encoder" in n and p.requires_grad
            ],
            "lr": args.text_encoder_lr,
        },
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")


    # load model
    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = TubeDETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        video_max_len=args.video_max_len,
        stride=args.stride,
        guided_attn=args.guided_attn,
        fast=args.fast,
        fast_mode=args.fast_mode,
        sted=args.sted,
    )
    model.to(device)
    print("model loaded")

    postprocessors = {"vidstg": PostProcessSTVG(), "bbox": PostProcess()}

    # load checkpoint
    assert args.load
    checkpoint = torch.load(args.load, map_location="cpu")
    if "model_ema" in checkpoint:
        if (
            args.num_queries < 100 and "query_embed.weight" in checkpoint["model_ema"]
        ):  # initialize from the first object queries
            checkpoint["model_ema"]["query_embed.weight"] = checkpoint["model_ema"][
                "query_embed.weight"
            ][: args.num_queries]
        if "transformer.time_embed.te" in checkpoint["model_ema"]:
            del checkpoint["model_ema"]["transformer.time_embed.te"]
        model.load_state_dict(checkpoint["model_ema"], strict=False)
    else:
        if (
            args.num_queries < 100 and "query_embed.weight" in checkpoint["model"]
        ):  # initialize from the first object queries
            checkpoint["model"]["query_embed.weight"] = checkpoint["model"][
                "query_embed.weight"
            ][: args.num_queries]
        if "transformer.time_embed.te" in checkpoint["model"]:
            del checkpoint["model"]["transformer.time_embed.te"]
        model.load_state_dict(checkpoint["model"], strict=False)
    print("checkpoint loaded")

    # load video (with eventual start & end) & caption demo examples
    captions = [args.caption]
    vid_path = vid #args.video
    # get video metadata
    probe = ffmpeg.probe(vid_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    """num, denum = video_stream["avg_frame_rate"].split("/")
    video_fps = int(num) / int(denum)"""
    clip_start = (
        float(video_stream["start_time"])
    )
    clip_end = (
        float(video_stream["start_time"]) + float(video_stream["duration"])
    )
    ss = clip_start
    t = clip_end - clip_start
    extracted_fps = (
        min((args.fps * t), args.video_max_len) / t
    )  # actual fps used for extraction given that the model processes video_max_len frames maximum
    cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=extracted_fps)
    out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        capture_stdout=True, quiet=True
    )
    w = int(video_stream["width"])
    h = int(video_stream["height"])
    images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
    assert len(images_list) <= args.video_max_len
    image_ids = [[k for k in range(len(images_list))]]

    # video transforms
    empty_anns = []  # empty targets as placeholders for the transforms
    placeholder_target = prepare(w, h, empty_anns)
    placeholder_targets_list = [placeholder_target] * len(images_list)
    transforms = make_video_transforms("test", cautious=True, resolution=args.resolution)
    images, targets = transforms(images_list, placeholder_targets_list)
    samples = NestedTensor.from_tensor_list([images], False)
    if args.stride:
        samples_fast = samples.to(device)
        samples = NestedTensor.from_tensor_list([images[:, :: args.stride]], False).to(
            device
        )
    else:
        samples_fast = None
    durations = [len(targets)]

    with torch.no_grad():  # forward through the model
        # encoder
        memory_cache = model(
            samples,
            durations,
            captions,
            encode_and_save=True,
            samples_fast=samples_fast,
        )
        # decoder
        outputs = model(
            samples,
            durations,
            captions,
            encode_and_save=False,
            memory_cache=memory_cache,
        )

        pred_steds = postprocessors["vidstg"](outputs, image_ids, video_ids=[0])[
            0
        ]  # (start, end) in terms of image_ids
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        vidstg_res = {}  # maps image_id to the coordinates of the detected box
        for im_id, result in zip(image_ids[0], results):
            vidstg_res[im_id] = {"boxes": [result["boxes"].detach().cpu().tolist()]}

        # create output dirs
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4])):
            os.makedirs(os.path.join(args.output_dir, vid_path.split("/")[-1][:-4]))
        # create output dirs frames
        if not os.path.exists(args.output_dir_frames):
            os.makedirs(args.output_dir_frames)
        if not os.path.exists(os.path.join(args.output_dir_frames, vid_path.split("/")[-1][:-4])):
            os.makedirs(os.path.join(args.output_dir_frames, vid_path.split("/")[-1][:-4]))
        # extract actual images from the video to process them adding boxes
        os.system(
            f'ffmpeg -i {vid_path} -ss {ss} -t {t} -qscale:v 2 -r {extracted_fps} {os.path.join(args.output_dir_frames, vid_path.split("/")[-1][:-4], str(vid_path.split("/")[-1][:-4]) + "%05d.jpg")}'
        )
        for img_id in image_ids[0]:
            # load extracted image
            img_path = os.path.join(
                args.output_dir_frames,
                vid_path.split("/")[-1][:-4],
                vid_path.split('/')[-1][:-4] + str(int(img_id) + 1).zfill(5) + ".jpg",
            )
            img = Image.open(img_path).convert("RGB")
            imgw, imgh = img.size
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.imshow(img, aspect="auto")

            if (
                pred_steds[0] <= img_id < pred_steds[1]
            ):  # add predicted box if the image_id is in the predicted start and end
                x1, y1, x2, y2 = vidstg_res[img_id]["boxes"][0]
                w = x2 - x1
                h = y2 - y1
                #rect = plt.Rectangle(
                #   (x1, y1), w, h, linewidth=2, edgecolor="#FAFF00", fill=False
                #)
                #ax.add_patch(rect)

            fig.set_dpi(100)
            fig.set_size_inches(imgw / 100, imgh / 100)
            fig.tight_layout(pad=0)

            # save image with eventual box
            fig.savefig(
                img_path,
                format="jpg",
            )
            plt.close(fig)

        # save video with tube
        os.system(
            f"ffmpeg -r {extracted_fps} -pattern_type glob -i '{os.path.join(args.output_dir_frames, vid_path.split('/')[-1][:-4])}/*.jpg' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r {extracted_fps} -crf 25 -c:v libx264 -pix_fmt yuv420p -movflags +faststart {os.path.join(args.output_dir, vid_path.split('/')[-1])}"
        )

        print (os.path.join(args.output_dir, vid_path.split('/')[-1]))
        return os.path.join(args.output_dir, vid_path.split('/')[-1])

