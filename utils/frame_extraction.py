import os

import numpy as np

import ffmpeg

def extract_frames(vid_path, fps, video_max_len, out_dir_frames):
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
            min((fps * t), video_max_len) / t
        )  # actual fps used for extraction given that the model processes video_max_len frames maximum
        cmd = ffmpeg.input(vid_path, ss=ss, t=t).filter("fps", fps=extracted_fps)
        out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
            capture_stdout=True, quiet=True
        )
        w = int(video_stream["width"])
        h = int(video_stream["height"])
        images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        assert len(images_list) <= video_max_len
        image_ids = [[k for k in range(len(images_list))]]

        os.system(
            f'ffmpeg -i {vid_path} -ss {ss} -t {t} -qscale:v 2 -r {extracted_fps} {os.path.join(out_dir_frames, vid_path.split("/")[-1][:-4], str(vid_path.split("/")[-1][:-4]) + "%05d.jpg")}'
        )