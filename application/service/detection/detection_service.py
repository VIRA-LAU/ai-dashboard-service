from __future__ import annotations

import json
import os.path
import os.path
from typing import Any

import requests
import torch

import application.service.lock_handler.lock_service as lock_service
from core.song_player import give_song
from core.video_concat import video_concat
from core.video_splitter import video_splitter
from detect import detect_all
from dev_utils.aws_conn.delete_downloaded_video import delete_downloaded_video
from dev_utils.aws_conn.download_hosted_video import download_video
from dev_utils.aws_conn.upload_detected_video import upload_highlights_to_s3
from getstats import getsShots
from dev_utils.handle_db.stats_handler import getAPIStats, getNetbasketCoordinatesFrames
from post_process import process_video
import persistence.repositories.paths as paths



class DetectionService:
    def __init__(self):
        self.weights = 'weights/yolov7-w6-pose.pt'
        self.weights = 'weights/yolov7-w6-pose.pt'

    def infer_detection(self, game_id: str) -> tuple[Any, Any, Any, Any]:
        with torch.no_grad():
            detect_all(game_id=game_id)
            # strip_optimizer(self.weights)
        return

    def run_inference(self, game_id: str) -> tuple[dict, str, str, str, str, int]:
        #download_video(game_id)
        #self.infer_detection(game_id)
        process_video(game_id)
        frames_point_scored, shotsmade = getNetbasketCoordinatesFrames(game_id)
        stats = getAPIStats(game_id, [1], [2])
        video_inferred_path = str(paths.post_process_path / f'{game_id}.mp4') # Post Process
        filename = os.path.basename(video_inferred_path)
        if shotsmade > 0:
            videos_paths = video_splitter(game_id=game_id, path_to_video=video_inferred_path,
                                          frames_shot_made=frames_point_scored)
            concatenated, video = video_concat(game_id, videos_paths, filename)
            print(video.duration)
            concatenated_with_music = give_song(game_id=game_id, video_clip=video, duration=int(video.duration),
                                                filename=filename)
        print(frames_point_scored)
        upload_highlights_to_s3(game_id)
        default_stats = {'game_id': 'f1tch41n',
                         'team_1': {'players': [{'player_1': {'scored': 3, 'missed': 0}}], 'points': 3,
                                    'possession': '100.0 %'},
                         'team_2': {'players': [], 'points': 0, 'possession': '0.0 %'}}
        #delete_downloaded_video(game_id)
        requests.patch(
            url=f'http://80.81.157.19:3000/games/video_processed/{game_id}',
            data=json.dumps(stats),
            headers= {'Content-Type': 'application/json'}
        )
        lock_service.LockService().deleteLockFile(game_id)
        print(stats)
        #return stats, video_inferred_path, videos_paths, concatenated, concatenated_with_music, shotsmade