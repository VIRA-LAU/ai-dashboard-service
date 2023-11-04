import math
import yaml

import numpy as np

import cv2
import os
from getstats import *
from shared.helper.json_helpers import parse_json
from dev_utils.paths.game import get_game_data
from dev_utils.handle_db.post_processing_handler import getPostProcessingData
import persistence.repositories.paths as paths

'''
    Individual Stats
'''

def process_video(game_id: str):
    video_path = get_game_data(game_id)[0]
    data = getPostProcessingData(game_id)
    video = cv2.VideoCapture(video_path)
    framerate = math.ceil(video.get(cv2.CAP_PROP_FPS))
    os.makedirs(os.path.join(paths.post_process_path), exist_ok=True)
    output = cv2.VideoWriter(
        str(paths.post_process_path / f'{game_id}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), framerate, (1920, 1080))
    
    while(True):
        ret, frame = video.read()
        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        if(ret):
            tl = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
            tf = max(tl - 3, 1)
            if data[(frame_num)] is not None:
                player_with_ball = data[frame_num]['player_with_ball']
                label = None

                if player_with_ball is not None:
                    if data[frame_num][player_with_ball]['coord'] is not None:
                        p_x1, p_y1, p_x2, p_y2 = data[frame_num][player_with_ball]['coord']
                        cv2.rectangle(frame, (int(p_x1), int(p_y1)), (int(p_x2), int(p_y2)), (53, 103, 240), tl)

                        # Text
                        label = 'player ' + str(player_with_ball) + ': points: ' + str(data[frame_num][player_with_ball]['score'])
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 =  int(p_x1) + t_size[0]-180, int(p_y1) - t_size[1] - 3
                        cv2.rectangle(frame, (int(p_x1), int(p_y1)), c2, (42, 43, 42), -1, cv2.LINE_AA)
                        cv2.putText(frame, label, (int(p_x1), int(p_y1-6) - 2), 0, tl / 6, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    
                if data[frame_num]['basket'] is not None:
                    b_x1, b_y1, b_x2, b_y2 = data[frame_num]['basket']
                    cv2.rectangle(frame, (int(b_x1), int(b_y1)), (int(b_x2), int(b_y2)), (240, 103, 53), tl)

                if data[frame_num]['netbasket'] != []:
                    n_x1, n_y1, n_x2, n_y2 = data[frame_num]['netbasket']
                    cv2.rectangle(frame, (int(n_x1), int(n_y1)), (int(n_x2), int(n_y2)), (240, 103, 53), tl)
                    # Text
                    netbasket_label = 'score'
                    t_size = cv2.getTextSize(netbasket_label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 =  int(n_x1) + t_size[0]-50, int(n_y1) - t_size[1] - 3
                    cv2.rectangle(frame, (int(n_x1), int(n_y1)), c2, (42, 43, 42), -1, cv2.LINE_AA)
                    cv2.putText(frame, netbasket_label, (int(n_x1), int(n_y1-6) - 2), 0, tl / 6, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    
            output.write(frame)
        else:
            break
  
    output.release()
    video.release()

    return output


if __name__ == "__main__":
    '''
        Video, Logs
    '''
    # video_dir = 'datasets/videos_input/04181.mp4'
    # data_path = 'datasets/logs/data.json'
    # data = parse_json(data_path)

    process_video('04181')

