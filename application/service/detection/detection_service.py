from __future__ import annotations

from typing import Tuple, Any

import torch

from core.song_player import give_song
from core.video_concat import video_concat
from core.video_splitter import video_splitter
from detect import detect_all
from utils.general import strip_optimizer


class DetectionService:
    def __init__(self):
        self.weights = 'weights/yolov7-w6-pose.pt'

    def infer_detection(self, source: str) -> tuple[Any, Any, Any, Any]:

        with torch.no_grad():
            stats, video_path, frames_point_scored, shotsmade = detect_all([1], [2])
            strip_optimizer(self.weights)
        return stats, video_path, frames_point_scored, shotsmade

    def run_inference(self, path_input_video: str, filename: str) -> tuple[dict, str, str, str, str, int]:
        stats, video_inferred_path, frames_point_scored, shots_made = self.infer_detection(source=path_input_video)
        if shots_made > 0:
            videos_paths = video_splitter(path_to_video=path_input_video, frames_shot_made=frames_point_scored)
            concatenated, video = video_concat(videos_paths, filename)
            print(video.duration)
            concatenated_with_music = give_song(video_clip=video, duration=int(video.duration), filename=filename)
        print(frames_point_scored)
        return stats, video_inferred_path, videos_paths, concatenated, concatenated_with_music, shots_made
