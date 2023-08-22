import math
import re
import yaml
import cv2
import os

def getPossessionPerTeam(logs):
    team1 = [1]
    team2 = [2]
    team1_frames = 0
    team2_frames = 0
    
    basketball_detections = logs['basketball_detections']
    pose_detections = logs['pose_detections']

    lastPlayerWithBall = None
    for frame in basketball_detections:
        basketball_coords = basketball_detections[frame]['bbox_coords']
        for player in pose_detections[frame]:
            player_id = pose_detections[frame][player]['player_id']
            if (checkIfPlayerHasBall(player['bbox_coords'], basketball_coords)):
                if player_id in team1:
                    team1_frames += 1
                    lastPlayerWithBall = player['player']
                elif player_id in team2:
                    team2_frames += 1
                    lastPlayerWithBall = player_id
                elif lastPlayerWithBall is not None:
                    if lastPlayerWithBall in team1:
                        team1_frames += 1
                    elif lastPlayerWithBall in team2:
                        team2_frames += 1

    total_frames = sum([team1_frames, team2_frames])
    possession_team_1 = round(team1_frames * 100 / total_frames, 1)
    possession_team_2 = round(team2_frames * 100 / total_frames, 1)

    return {
        'Team 1 possession' : f'{possession_team_1} %',
        'Team 2 possession' : f'{possession_team_2} %'
    }

def getShootingPlayers(logs):
    shooting_players = []
    # video_fps = get_video_framerate(logFileDirectory)
    NUMBER_OF_FRAMES_PER_SHOOTING = 30 * 2
    NETBASKET_FRAMES_AFTER_SHOOTING = 30 * 3
    PERSON_ACTION_PRECISION = 0.92

    pose_detections = logs['pose_detection']
    actions_detections = logs['action_detection']
    basketball_detections = logs['basketball_detection']
    for frame in actions_detections:
        playerNum = None
        player_position = None
        if(actions_detections[frame]['action']=='shooting'):
            shooting_coords = actions_detections[frame]['bbox_coords']
            for player in pose_detections[frame]:
                player_id = pose_detections[frame][player]['player_id']
                player_bbox = pose_detections[frame][player]['bbox_coords']
                

                playerCoords = player_bbox
                check = []
                for i in range(len(shooting_coords)):
                    check.append(min(playerCoords[i], shooting_coords[i]) /
                                max(playerCoords[i], shooting_coords[i]) >= PERSON_ACTION_PRECISION)
                    
                if len(check) == 4 and all(check):
                    player_position = pose_detections[frame][player]['position']
                    playerNum = player_id

        shot_frame = frame + NETBASKET_FRAMES_AFTER_SHOOTING
        if(basketball_detections[shot_frame] is not None):
            if(basketball_detections[shot_frame]['shot']=='netbasket'):
                entry = {
                    'frame': frame,
                    'player': playerNum,
                    'shot': 'netbasket',
                    'points': 2 if player_position == '2_points' else 3
                }
                shooting_players.append(entry)
            else:
                entry = {
                    'frame': frame,
                    'player': playerNum,
                    'shot': 'basket_missed',
                    'points': 0
                }
                shooting_players.append(entry)
            
    return shooting_players

def getIndividualStats(shooting_players):
    individ = {}
    for data in shooting_players:
        entry = {
            'scored': 0,
            'missed': 0,
            'total_points':0
        }
        individ[shooting_players[data]['player']].append(entry)

    for data in shooting_players:
        if(shooting_players[data]['show'] == 'basket_missed'):
            individ[shooting_players[data]['player']]['missed']+=1
        else:
            individ[shooting_players[data]['player']]['scored']+=1
            individ[shooting_players[data]['player']]['total_points']+=shooting_players[data]['points']

    return individ

def getPointsPerTeam(player_scores, team1_players, team2_players):
    team1Points = 0
    team2Points = 0
    for player in player_scores:
        playerNum = int(player[-1])
        if playerNum in team1_players:
            team1Points += player_scores[player]['scored']
        if playerNum in team2_players:
            team2Points += player_scores[player]['scored']
    return {
        'Team 1' : team1Points,
        'Team 2' : team2Points
    }


def checkIfPlayerHasBall(playerCoords, basketballCoords):
    if(all(element is None for element in basketballCoords)):
        return False
    check = []
    player_x1, player_y1, player_x2, player_y2 = playerCoords
    basket_x1, basket_y1, basket_x2, basket_y2 = basketballCoords
    basket_x = (basket_x1 + basket_x2) / 2
    basket_y = (basket_y1 + basket_y2) / 2
    if player_x1 <= basket_x and basket_x <= player_x2:
        check.append(True)
    elif getCoordsRatio(basket_x, player_x1) >= 0.92 or getCoordsRatio(basket_x, player_x2) >= 0.92:
        check.append(True)
    else:
        check.append(False)
    if player_y1 <= basket_y and basket_y <= player_y2:
        check.append(True)
    elif getCoordsRatio(basket_y, player_y1) >= 0.92 or getCoordsRatio(basket_y, player_y2) >= 0.92:
        check.append(True)
    else:
        check.append(False)
    if len(check) == 2 and all(check):
        return True
    else:
        return False


def getCoordsRatio(coord1, coord2):
    return min(coord1, coord2) / max(coord1, coord2)


def get_video_framerate(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0].split('_')[0]
    video_extension = None
    if video_name is not None:
        video_extension = get_video_name(video_name)
    if video_extension is not None:
        # Open the video file
        video = cv2.VideoCapture(video_extension)

        # Check if the video file is successfully opened
        if not video.isOpened():
            print("Error opening video file")
            return None

        # Get the framerate of the video
        framerate = video.get(cv2.CAP_PROP_FPS)

        # Release the video file
        video.release()

        return math.ceil(framerate)
    return 0


def get_video_name(video_title):
    for file_name in os.listdir('datasets/videos_input'):
        if file_name.startswith(video_title):
            # Found a file with the desired name
            file_path = os.path.join('datasets/videos_input', file_name)
            return file_path
    return None

if __name__ == "__main__":
    logs_path = 'datasets/logs/test_log.yaml'
    logs = yaml.safe_load(logs_path)

    possession = getPossessionPerTeam(logs)
    shooting_players = getShootingPlayers(logs)
    individual_stats = getIndividualStats(shooting_players)

    print(possession)
    print("#######################################")
    print(shooting_players)
    print("#######################################")
    print(individual_stats)
    print("#######################################")