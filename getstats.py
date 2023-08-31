import math
import yaml

import cv2


'''
    Team Stats
        Needs logs file:
            getPossessionPerTeam
            getPassesPerTeam
            getShootingPlayers
        Dependent Methods:
            getPointsPerTeam
            getScoreBoard
'''
def getPossessionPerTeam(logs, teams):
    team1, team2 = teams
    team1_frames = 0
    team2_frames = 0
    
    basketball_detections = logs['basketball_detection']
    pose_detections = logs['pose_detection']

    lastPlayerWithBall = None
    for frame in basketball_detections:
        # Get Basketball Coordinates
        for basket_dets in basketball_detections[frame]:
            if(basket_dets['shot']=='basketball'):
                basketball_coords = basket_dets['bbox_coords']
        for player in pose_detections[frame]:
            curr_player = pose_detections[frame][player][0]
            player_id = int(curr_player['player_id'])
            if (checkIfPlayerHasBall(curr_player['bbox_coords'], basketball_coords)):
                if player_id in team1:
                    team1_frames += 1
                    lastPlayerWithBall = player_id
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
        'team_1_possession' : f'{possession_team_1} %',
        'team_2_possession' : f'{possession_team_2} %'
    }


def getPassesPerTeam(logs, teams, video):
    team1, team2 = teams
    team1_passes = 0
    team2_passes = 0
    team1_assists = 0
    team2_assists = 0

    passes = {}
    assists = {}

    fps = get_video_framerate(video)

    if(fps==60):
        NUMBER_OF_FRAMES_PER_SHOOTING = int(fps / 1.8)
        NETBASKET_FRAMES_AFTER_PASSING = int(fps * 2.5)
    elif(fps==30):
        NUMBER_OF_FRAMES_PER_SHOOTING = int(fps * 1.5)
        NETBASKET_FRAMES_AFTER_PASSING = int(fps * 2.5)
    else:
        NUMBER_OF_FRAMES_PER_SHOOTING = int(fps * 1.5)
        NETBASKET_FRAMES_AFTER_PASSING = int(fps * 2.5)
    
    basketball_detections = logs['basketball_detection']
    pose_detections = logs['pose_detection']

    lastPlayerWithBall = None
    lastPlayerWithBallCoords = None
    for frame in basketball_detections:
        shot_frame = frame + NETBASKET_FRAMES_AFTER_PASSING
        # Get Basketball Coordinates
        for basket_dets in basketball_detections[frame]:
            if(basket_dets['shot']=='basketball'):
                basketball_coords = basket_dets['bbox_coords']
        for player in pose_detections[frame]:
            curr_player = pose_detections[frame][player][0]
            player_id = int(curr_player['player_id'])
            if lastPlayerWithBall is not None:
                if (checkIfPlayerHasBall(lastPlayerWithBallCoords, basketball_coords)):
                    continue
                else:
                    if (checkIfPlayerHasBall(curr_player['bbox_coords'], basketball_coords)):
                        print(frame)
                        if player_id in team1:
                            team1_passes += 1
                        elif player_id in team2:
                            team2_passes += 1
                        lastPlayerWithBall = player_id
                        lastPlayerWithBallCoords = curr_player['bbox_coords']
                        '''Assists'''
                        if(shot_frame in basketball_detections):
                            for basket_dets in basketball_detections[shot_frame]:
                                if(basket_dets['shot']=='netbasket'):
                                    if player_id in team1:
                                        team1_assists += 1
                                    elif player_id in team2:
                                        team2_assists += 1

            else:
                if (checkIfPlayerHasBall(curr_player['bbox_coords'], basketball_coords)):
                    lastPlayerWithBall = player_id
                    lastPlayerWithBallCoords = curr_player['bbox_coords']

    passes = {
        "Team 1 Passes": team1_passes,
        "Team 2 Passes": team2_passes
    }

    assists = {
        "Team 1 Assists": team1_assists,
        "Team 2 Assists": team2_assists
    }

    return passes, assists


def getShootingPlayers(logs, teams, video):
    shooting_players = []
    scoring_players = []

    team1, team2 = teams

    '''Get FPS to calculate frames to skip after detection'''
    fps = get_video_framerate(video)

    if(fps==60):
        NUMBER_OF_FRAMES_PER_SHOOTING = int(fps / 1.8)
        NETBASKET_FRAMES_AFTER_SHOOTING = int(fps * 1.3)
    elif(fps==30):
        NUMBER_OF_FRAMES_PER_SHOOTING = int(fps * 1.5)
        NETBASKET_FRAMES_AFTER_SHOOTING = int(fps * 2.5)
    else:
        NUMBER_OF_FRAMES_PER_SHOOTING = int(fps * 1.5)
        NETBASKET_FRAMES_AFTER_SHOOTING = int(fps * 2.5)

    PERSON_ACTION_PRECISION = 0.92

    '''Extract Detections from logs'''
    pose_detections = logs['pose_detection']
    actions_detections = logs['action_detection']
    basketball_detections = logs['basketball_detection']
    skip_frames = False
    skip_count = 0

    '''Iterate over frames in action detections'''
    for frame in actions_detections:
        # Check if needed to skip frames after shooting detection
        if(skip_frames):
            if(skip_count>0):
                skip_count -= 1
                continue # skip frames by using continue
            else:
                skip_frames = False

        # Get action detections
        playerNum = None
        player_position = None
        for action in actions_detections[frame]:
            if(action['action']=='shooting'):
                shooting_coords = action['bbox_coords']
                # Get the shooting player
                for player in pose_detections[frame]:
                    curr_player = pose_detections[frame][player][0]
                    player_id = curr_player['player_id']
                    player_bbox = curr_player['bbox_coords']
                    
                    playerCoords = player_bbox
                    check = []
                    for i in range(len(shooting_coords)):
                        check.append(min(playerCoords[i], shooting_coords[i]) /
                                    max(playerCoords[i], shooting_coords[i]) >= PERSON_ACTION_PRECISION)
                        
                    if len(check) == 4 and all(check):
                        player_position = curr_player['position']
                        xy_position = curr_player['feet_coords']
                        playerNum = int(player_id)
                        entry = {
                            'frame': frame,
                            'player': playerNum,
                            'action': 'shooting',
                            'position': player_position,
                            'xy_position': xy_position,
                            'team': "team1" if playerNum in team1 else "team2"
                        }
                        shooting_players.append(entry)

                shot_frame = frame + NETBASKET_FRAMES_AFTER_SHOOTING
                if(basketball_detections[shot_frame] is not None):
                    for basket_dets in basketball_detections[shot_frame]:
                        if(basket_dets['shot']=='netbasket'):
                            entry = {
                                'frame': shot_frame,
                                'player': playerNum,
                                'shot': 'scored',
                                'points': 2 if player_position == '2_points' else 3,
                                'xy_position': xy_position,
                                'team': "team1" if playerNum in team1 else "team2"
                            }
                            scoring_players.append(entry)
                        elif(basket_dets['shot']=='netempty'):
                            entry = {
                                'frame': shot_frame,
                                'player': playerNum,
                                'shot': 'missed',
                                'points': 0,
                                'xy_position': xy_position,
                                'team': "team1" if playerNum in team1 else "team2"
                            }
                            scoring_players.append(entry)

                skip_count = NUMBER_OF_FRAMES_PER_SHOOTING
                skip_frames = True
            
    return shooting_players, scoring_players


def getPointsPerTeam(scoring_players, teams):
    '''
        Parameters:
            scoring_players: list of dicts
            teams: list of lists
        Returns:
            dict
    '''
    team1, team2 = teams
    team1Points = 0
    team2Points = 0
    for entry in scoring_players:
        player_id = int(entry['player'])
        if player_id in team1:
            team1Points += entry['points']
        if player_id in team2:
            team2Points += entry['points']
    return {
        'team_1' : team1Points,
        'team_2' : team2Points
    }


def getScoreBoard(scoring_players):
    '''
    Get both teams scores in chronological order to add on scoreboard in video during post process
        Scorboard hierarchy:
            Frames:
                frame_num:
                    team1:
                        player
                        score
                    team2:
                        player
                        score
    '''

    score_board = {
        "frame": {}
    }
    team1_score = 0
    team2_score = 0

    for entry in scoring_players:
        # Get scoring player and team
        player = entry['player']
        if(entry['team'] == "team1"):
            team1_score+=entry['points']
        elif(entry['team'] == "team2"):
            team2_score+=entry['points']

        # Add entry to te scoreboard
        score_board['frame'][entry['frame']] = {
            "team1": {
                "player": player if entry['team'] == "team1" else None,
                "score": team1_score
            },
            "team2": {
                "player": player if entry['team'] == "team2" else None,
                "score": team2_score
            }
        }

    return score_board

'''
    Individual Stats
'''
def getIndividualStats(scoring_players):
    individ = {}
    for data in scoring_players:
        entry = {
            'scored': 0,
            'missed': 0,
            'total_points':0
        }
        individ[data['player']] = entry

    for data in scoring_players:
        if(data['shot'] == 'missed'):
            individ[data['player']]['missed']+=1
        else:
            individ[data['player']]['scored']+=1
            individ[data['player']]['total_points']+=data['points']

    return individ


def getPlayerBbox(pose_detection, player_id):
    player_bbox = {}
    for frame in pose_detection:
        player_bbox[frame] = pose_detection[frame][player_id][0]["bbox_coords"]
    return player_bbox

'''
    Utils
'''
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
    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Get the framerate of the video
        framerate = video.get(cv2.CAP_PROP_FPS)

        # Release the video file
        video.release()

        return math.ceil(framerate)
    except:
        print("Error opening video file")
        return 0



if __name__ == "__main__":
    video = 'datasets/videos_input/04181.mp4'
    logs_path = 'datasets/logs/04181_log.yaml'
    team1 = [1]
    team2 = [2]
    teams = [team1, team2]
    with open(logs_path, "r") as stream:
        logs = yaml.safe_load(stream)

    possession = getPossessionPerTeam(logs, teams)
    shooting_players, scoring_players = getShootingPlayers(logs, teams, video)
    score_board = getScoreBoard(scoring_players)
    individual_stats = getIndividualStats(scoring_players)
    points_per_team = getPointsPerTeam(scoring_players, teams)
    passes, assists = getPassesPerTeam(logs, teams, video)

    print(possession)
    print("1#######################################")
    print(shooting_players)
    print("2#######################################")
    print(scoring_players)
    print("3#######################################")
    print(points_per_team)
    print("4#######################################")
    print(score_board)
    print("5#######################################")
    print(individual_stats)
    print("6#######################################")
    print(passes)
    print("7#######################################")
    print(assists)
    print("7#######################################")