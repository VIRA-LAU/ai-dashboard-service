from __future__ import annotations

import os.path
from typing import Tuple, Any

import torch

from core.song_player import give_song
from core.video_concat import video_concat
from core.video_splitter import video_splitter
from detect import detect_all
import application.service.lock_handler.lock_service as lock_service
from upload_detected_video import upload_highlights_to_s3
from utils.general import strip_optimizer


class DetectionService:
    def __init__(self):
        self.weights = 'weights/yolov7-w6-pose.pt'

    def infer_detection(self, game_id: str) -> tuple[Any, Any, Any, Any]:
        with torch.no_grad():
            stats, video_path, frames_point_scored, shotsmade = detect_all(game_id, [1], [2])
            # strip_optimizer(self.weights)
        return stats, video_path, frames_point_scored, shotsmade

    def run_inference(self, game_id: str) -> tuple[dict, str, str, str, str, int]:
        stats, video_inferred_path, frames_point_scored, shots_made = self.infer_detection(game_id)
        filename = os.path.basename(video_inferred_path)
        if shots_made > 0:
            videos_paths = video_splitter(game_id=game_id, path_to_video=video_inferred_path,
                                          frames_shot_made=frames_point_scored)
            concatenated, video = video_concat(game_id, videos_paths, filename)
            print(video.duration)
            concatenated_with_music = give_song(game_id=game_id, video_clip=video, duration=int(video.duration),
                                                filename=filename)
        print(frames_point_scored)
        upload_highlights_to_s3(game_id)
        lock_service.LockService().deleteLockFile(game_id)
        print(stats)
        return stats, video_inferred_path, videos_paths, concatenated, concatenated_with_music, shots_made
