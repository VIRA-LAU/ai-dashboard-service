import yaml

import torch
import yaml
import os

from utils.paths.game import get_game_data
from utils.general import strip_optimizer

from persistence.repositories import paths

import utils.handle_db.action_db_handler as action_db
import utils.handle_db.basket_db_handler as basket_db
import utils.handle_db.person_db_handler as pose_db

from utils.args import *
from utils.frame_extraction import extract_frames

from detect.detect_actions import detect_actions
from detect.detect_basketball import detect_basketball
from detect.detect_person import detect_person


def extract_key_frames(dataset_folder: str = "", video_max_len: int = 200, fps: int = 5):
    out_dir = paths.video_input_path # output video with less frames
    out_dir_frames = paths.temporal_frames  # output frames

    source = os.path.join(paths.temporal_videos_input_path, dataset_folder)
    for video in os.listdir(source):
        vid_path = os.path.join(source, video)
        extract_frames(vid_path, fps, video_max_len, out_dir_frames)

       
def detect_all(game_id: str = ''):
    videoPath, dataLogFilePath = get_game_data(game_id=game_id)
    print(torch.cuda.is_available())
    torch.cuda.empty_cache()
    with torch.no_grad():
        action_weights = 'weights/actions_3_5.pt'
        basket_weights = 'weights/netbasket_3_5.pt'
        instance_weights = 'weights/yv8_person_seg.pt'

        '''
            Detect
        '''
        # # Actions
        # action_db.create_database(game_id)
        # action_db.create_action_table()
        # actions_logs = detect_actions(weights=action_weights, source=videoPath, dont_save=False)
        # strip_optimizer(action_weights)
        # action_db.close_db()
        # writeToLog(dataLogFilePath, actions_logs)

        # Basketball
        basket_db.create_database(game_id)
        basket_db.create_basket_table()
        detect_basketball(weights=basket_weights, source=videoPath, dont_save=False)
        strip_optimizer(basket_weights)
        basket_db.close_db()

        # Instance Segmentation
        pose_db.create_database(game_id)
        pose_db.create_pose_table()
        detect_person(weights=instance_weights, source=videoPath, dont_save=False)
        pose_db.close_db()


if __name__ == '__main__':
    # extract_key_frames(dataset_folder='F1', video_max_len = 10000)
    # extract_key_frames(dataset_folder='F2', video_max_len = 10000)
    detect_all(game_id='IMG_3050_Demo')
    # TO FIX: Write the rescaled bounding boxes of person (and possibly action) to the db file