import math
import yaml

import numpy as np

import cv2

from moviepy.editor import *
from moviepy.video.fx.resize import resize

from getstats import *
from shared.helper.json_helpers import parse_json

'''
    Team Stats:
        1. Lineups
        2. Scoreboard
        3. Overall Stats
        4. All detections bbox
'''
def add_score(video, score_asset, scores):
    x = 948
    y = 845
    score_composites=[]
    frames = scores['frame']
    for i, frame in enumerate(frames):
        team1_score_text = TextClip(str(frames[frame]['team1']['score']),font="SpaceGrotesk-Bold", fontsize=50, color='white')
        team2_score_text = TextClip(str(frames[frame]['team2']['score']),font="SpaceGrotesk-Bold", fontsize=50, color='white')

        team1_score_text = team1_score_text.set_pos((x-150,y))
        team2_score_text = team2_score_text.set_pos((x+150,y))

        score_clip_composite = CompositeVideoClip([score_asset, team1_score_text, team2_score_text])
        score_clip_composite = score_clip_composite.set_start(int(frame)/60)
        score_composites.append(score_clip_composite)

        if(i == 0):
            clip_duration = video.duration - int(frame)/30

    score_clip = CompositeVideoClip(score_composites)
    score_clip.duration = clip_duration

    video_scores = CompositeVideoClip([video, score_clip.crossfadein(1)])
    video_scores.duration = video.duration
    video_scores.write_videofile("datasets/post_process/exported/test.mp4", fps=60,codec='libx264')

    return video_scores


def add_lineups(video, lineups, team):
    x=500
    y=700
    text_composites=[]

    x_num = 750
    y_first = 350

    logo = resize(logo, height=80, width=80)
    logo = logo.set_position((645, 190))
    txt_team = TextClip("FITCHAIN", font='Space-Grotesk-Bold', color='#FFFFFF',fontsize=40)
    txt_mov_team = txt_team.set_pos((750, 205))
    text_composites.append(logo, txt_mov_team)

    y_increment = 0
    for player in team["players"]:
        number_txt = TextClip(str(player["number"]),font="Space Grotesk", fontsize=30, color='black')
        first_name_txt = TextClip(str(player["first name"]),font="Space Grotesk", fontsize=30, color='black')
        last_name_txt = TextClip(str(player["last name"]),font="Space Grotesk", fontsize=30, color='black')

        number_txt = number_txt.set_pos(x_num,y)
        first_name_txt = first_name_txt.set_pos(x+number_txt.w+50,y)
        last_name_txt = last_name_txt.set_pos(x+last_name_txt.w+70,y)

        text_composites.append(number_txt, first_name_txt, last_name_txt)

    text_clip = CompositeVideoClip(text_composites)

    video_lineup = CompositeVideoClip([video, # starts at t=0
                            lineups.set_start(2).crossfadein(1),
                            text_clip.set_start(2).crossfadein(1.5)])
    
    video_lineup = video_lineup.set_duration(10).crossfadeout(2)
    
    return video_lineup


def load_clips(video_dir, lineups_asset, stats_asset, score_asset):
    video = VideoFileClip(video_dir)
    # lineups = VideoFileClip(lineups_asset)
    lineups = ''
    stats = ''
    # stats = VideoFileClip(stats_asset)
    score = ImageClip(score_asset)

    return video, lineups, stats, score


def draw_bbox():
    return


'''
    Individual Stats
'''

def draw_player_bbox(video_path, player_bbox):
    video = cv2.VideoCapture(video_path)
    framerate = math.ceil(video.get(cv2.CAP_PROP_FPS))

    output = cv2.VideoWriter(
        "datasets/post_process/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), framerate, (1920, 1080))
    
    while(True):
        ret, frame = video.read()
        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        if(ret):
            bbox = player_bbox[frame_num]
            if(bbox is not None):
                x1, y1, x2, y2 = [int(i) for i in bbox] #xmin, ymin, xmax, ymax

                half_w = int((x2-x1)/2)
                center=int(x1+half_w)
                left = center - 15
                right = center + 15
                top_c = y1-40
                top_lr = top_c - 15

                pt1 = (center, top_c) # Center Point
                pt2 = (left, top_lr) # Left Point
                pt3 = (right, top_lr) # Right Point

                triangle_cnt = np.array( [pt1, pt2, pt3] )

                # Draw Triangle
                cv2.drawContours(frame, [triangle_cnt], 0, (255,0,0), -1)

                # # Draw Triangle
                # cv2.circle(frame, pt1, 2, (0,0,255), -1)
                # cv2.circle(frame, pt2, 2, (0,0,255), -1)
                # cv2.circle(frame, pt3, 2, (0,0,255), -1)

            output.write(frame)
        else:
            break
  
    output.release()
    video.release()

    return output


def add_individual_score(scoring_players, player_id):
    return


if __name__ == "__main__":
    '''
        Video, Logs
    '''
    video_dir = 'datasets/videos_input/04181.mp4'
    logs_path = 'datasets/logs/04181_log.yaml'
    with open(logs_path, "r") as stream:
        logs = yaml.safe_load(stream)

    '''
        Team Stats
    '''
    team1 = [1]
    team2 = [2]
    teams = [team1, team2]

    lineups_asset_dir = 'assets/templates/' 
    stats_asset_dir = 'assets/templates/'
    score_asset_dir = 'assets/templates/Scoreboard.png'

    video, lineups_asset, stats_asset, score_asset = load_clips(video_dir, lineups_asset_dir, stats_asset_dir, score_asset_dir)
    shooting_players, scoring_players = getShootingPlayers(logs, teams, video_dir)
    scores = getScoreBoard(scoring_players)
    video_scores = add_score(video, score_asset, scores)

    '''
        Individual Stats
    '''
    with open(logs_path, "r") as stream:
        logs = yaml.safe_load(stream)

    player_bbox = getPlayerBbox(logs['pose_detection'], "player_1")
    draw_player_bbox(video_dir, player_bbox)