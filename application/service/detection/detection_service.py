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
from detect_helper import detect_all
from utils.aws.delete_downloaded_video import delete_downloaded_video
from utils.aws.download_hosted_video import download_video
from utils.aws.upload_detected_video import upload_highlights_to_s3
from utils.handle_db.stats_db_handler import Stats_DB_Handler
from post_process import process_video
import persistence.repositories.paths as paths

class DetectionService:
    def infer_detection(self, game_id: str) -> tuple[Any, Any, Any, Any]:
        with torch.no_grad():
            detect_all(game_id=game_id)
        return

    def run_inference(self, game_id: str) -> tuple[dict, str, str, str, str, int]:
        # download_video(game_id)
        # self.infer_detection(game_id)
        game_db_stats = Stats_DB_Handler(game_id, [1], [2])
        process_video(game_id, game_db_stats.post_process_data)
        frames_point_scored, shotsmade = game_db_stats.getNetbasketCoordinatesFrames()
        stats = game_db_stats.getAPIStats()
        video_inferred_path = str(paths.post_process_path / f'{game_id}.mp4') # Post Process
        filename = os.path.basename(video_inferred_path)
        if shotsmade > 0:
            videos_paths = video_splitter(game_id=game_id, path_to_video=video_inferred_path,
                                          frames_shot_made=frames_point_scored)
            concatenated, video = video_concat(game_id, videos_paths, filename)
            print(video.duration)
            concatenated_with_music = give_song(game_id=game_id, video_clip=video, duration=int(video.duration),
                                                filename=filename)

        # upload_highlights_to_s3(game_id)
        # delete_downloaded_video(game_id)
        # requests.patch(
        #     url=f'http://ec2-16-170-232-235.eu-north-1.compute.amazonaws.com:3000/ai/video_processed/{game_id}',
        #     data=json.dumps(stats),
        #     headers= {'Content-Type': 'application/json'}
        # )
        # lock_service.LockService().deleteLockFile(game_id)
        return stats, video_inferred_path, videos_paths, concatenated, concatenated_with_music, shotsmade