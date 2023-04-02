from __future__ import annotations

from typing import Tuple, Any

import torch

from core.song_player import give_song
from core.video_concat import video_concat
from core.video_splitter import video_splitter
from detect import detect
from utils.general import strip_optimizer


class DetectionService:
    def __init__(self):
        self.weights = 'weights/yolov7-w6-pose.pt'

    def infer_detection(self, source: str, pose_est: bool) -> tuple[Any, Any, Any, Any]:
        if (pose_est):
            weights = 'weights/yolov7-w6-pose.pt'
        else:
            weights = self.weigts
        with torch.no_grad():
            video_path, txt_path = detect(weights=weights, source=source, shots=not pose_est, pose_est=pose_est)
            strip_optimizer(self.weights)
        return video_path, txt_path

    def run_inference(self, path_input_video: str, filename: str, pose_est: bool) -> tuple[str, list, str, str, Any]:
        if (pose_est):
            video_inferred_path, bbox_coordinated_path = self.infer_detection(source=path_input_video, pose_est=pose_est)
            return video_inferred_path
        else:
            video_inferred_path, bbox_coordinated_path, frames_made, shots_made = self.infer_detection(source=path_input_video)
            if shots_made > 0:
                videos_paths = video_splitter(path_to_video=path_input_video, frames_shot_made=frames_made)
                concatenated, video = video_concat(videos_paths, filename)
                print(video.duration)
                concatenated_with_music = give_song(video_clip=video, duration=int(video.duration), filename=filename)
            print(frames_made)
            return video_inferred_path, videos_paths, concatenated, concatenated_with_music, shots_made
