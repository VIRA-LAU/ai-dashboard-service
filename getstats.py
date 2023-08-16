import math
import re
import yaml
import cv2
import os

data = {}
team1 = [1]
team2 = [2]
team1_frames = 0
team2_frames = 0

framesWithShootingPlayer = []
pointsForAllPlayers = {}
PERSON_ACTION_PRECISION = 0.92

def getPossessionPerTeam(logFileDirectory: str):
    global data, team1_frames, team2_frames
    with open(logFileDirectory, 'r') as file:
        data = yaml.safe_load(file)
    lastPlayerWithBall = None
    for frame in data:
        basketball_coords = getBasketballCoords(frame)
        all_players_coords = getAllPlayersCoords(frame)
        for player in all_players_coords:
            if (checkIfPlayerHasBall(player['coords'], basketball_coords)):
                if player['playerNum'] in team1:
                    team1_frames += 1
                    lastPlayerWithBall = player['playerNum']
                elif player['playerNum'] in team2:
                    team2_frames += 1
                    lastPlayerWithBall = player['playerNum']
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

def getPointsPerPlayer(logFileDirectory: str):
    global data, framesWithShootingPlayer, pointsForAllPlayers
    video_fps = get_video_framerate(logFileDirectory)
    NUMBER_OF_FRAMES_PER_SHOOTING = video_fps * 2
    NETBASKET_FRAMES_AFTER_SHOOTING = video_fps * 3
    with open(logFileDirectory, 'r') as file:
        data = yaml.safe_load(file)
    for frame in data:
        for frameInfo in data[frame]:
            shooting_coords = getShootingCoords(frame, frameInfo)
            playerCoords, playerNum, index = [None, None, None]
            for stat in data[frame]:
                if playerCoords is not None and playerNum is not None and index is not None:
                    break
                playerCoords, playerNum, index = getShootingPlayer(frame, stat, shooting_coords)
            addShootingPlayerToGlobalMap(frame, index, playerCoords, playerNum, NUMBER_OF_FRAMES_PER_SHOOTING)
    for frameObject in framesWithShootingPlayer:
        frameNum = frameObject['frame']
        index = frameObject['index']
        pointLogged = False
        for frame in range(frameNum, frameNum + NETBASKET_FRAMES_AFTER_SHOOTING):
            playerIdentifier = 'Player ' + frameObject['playerNum']
            if pointLogged:
                break
            for frameInfo in data[frame]:
                if frameInfo.startswith('basketball_detection'):
                    pointLogged = addPointToPlayer(frame, frameInfo, playerIdentifier, index)
                    if pointLogged:
                        break
        addMissedPointToPlayer(playerIdentifier, pointLogged)
    return pointsForAllPlayers


def getPointsPerTeam(player_scores: dict, team1_players: list, team2_players: list):
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

def getBasketballCoords(frame: int):
    basketballIndex = 0
    for frameInfo in data[frame]:
        if frameInfo.startswith('basketball_detection'):
            if data[frame][frameInfo] == 'basketball':
                basketballIndex = frameInfo[-1]
                break
    basket_coords = 'basketball_bbox_coords_' + str(basketballIndex)
    if basket_coords in data[frame]:
        return data[frame][basket_coords]
    else:
        return [None, None, None, None]


def getAllPlayersCoords(frame: int):
    allPlayersCoords = []
    for frameInfo in data[frame]:
        pattern = r'player_\d+_bbox_coords_\d+'
        if re.match(pattern, frameInfo):
            playerNum = frameInfo[frameInfo.find('_') + 1]
            currentPlayerCoords = data[frame][frameInfo]
            allPlayersCoords.append(
                {
                    'playerNum': int(playerNum),
                    'coords': currentPlayerCoords
                }
            )
    return allPlayersCoords


def checkIfPlayerHasBall(playerCoords: list, basketballCoords: list):
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


def getCoordsRatio(coord1: int, coord2: int):
    return min(coord1, coord2) / max(coord1, coord2)







def getShootingCoords(frame: int, frameInfo: str):
    shooting_coords = []
    if frameInfo.startswith('action_detection'):
        if data[frame][frameInfo] == 'shooting':
            index = frameInfo[-1]
            shooting_coords = data[frame]['action_bbox_coords_' + index]
    return shooting_coords


def getShootingPlayer(frame: int, stat: str, shooting_coords: list):
    playerInfo = [None, None, None]
    pattern = r'player_\d+_bbox_coords_\d+'
    if re.match(pattern, stat):
        currentCoords = data[frame][stat]
        check = []
        for i in range(len(shooting_coords)):
            check.append(min(currentCoords[i], shooting_coords[i]) /
                         max(currentCoords[i], shooting_coords[i]) >= PERSON_ACTION_PRECISION)
        if len(check) == 4 and all(check):
            playerInfo[0] = currentCoords
            playerInfo[1] = stat[stat.find('_') + 1]
            playerInfo[2] = str(stat[-1])
    return playerInfo


def addShootingPlayerToGlobalMap(frame: int,
                                 index: str,
                                 playerCoords: list,
                                 playerNum: str,
                                 NUMBER_OF_FRAMES_PER_SHOOTING: int):
    if playerCoords is not None and playerNum is not None:
        if len(framesWithShootingPlayer) == 0:
            framesWithShootingPlayer.append(
                {'frame': frame, 'index': index, 'playerNum': playerNum})
        elif frame - framesWithShootingPlayer[-1]['frame'] >= NUMBER_OF_FRAMES_PER_SHOOTING:
            framesWithShootingPlayer.append(
                {'frame': frame, 'index': index, 'playerNum': playerNum})


def addPointToPlayer(frame: int, frameInfo: str, playerIdentifier: str, index: str):
    if frameInfo.startswith('basketball_detection'):
        if data[frame][frameInfo] == 'netbasket':
            playerPoints = 0
            for detection in data[frame]:
                if detection.endswith('_position_' + index):
                    playerPoints = 2 if data[frame][detection] == '2_points' else 3
            if playerPoints != 0:
                if playerIdentifier not in pointsForAllPlayers.keys():
                    pointsForAllPlayers[playerIdentifier] = {'scored': 0, 'missed': 0}
                pointsForAllPlayers[playerIdentifier]['scored'] += playerPoints
                return True
    return False


def addMissedPointToPlayer(playerIdentifier: str, pointLogged: bool):
    if not pointLogged:
        if playerIdentifier not in pointsForAllPlayers.keys():
            pointsForAllPlayers[playerIdentifier] = {'scored': 0, 'missed': 0}
        if playerIdentifier is not None and playerIdentifier in pointsForAllPlayers.keys():
            pointsForAllPlayers[playerIdentifier]['missed'] += 1


def get_video_framerate(video_path: str):
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


def get_video_name(video_title: str):
    for file_name in os.listdir('datasets/videos_input'):
        if file_name.startswith(video_title):
            # Found a file with the desired name
            file_path = os.path.join('datasets/videos_input', file_name)
            return file_path
    return None

if __name__ == '__main__':
    print(getPointsPerPlayer('datasets/logs/04181_log.yaml'))
    print(getPossessionPerTeam('datasets/logs/04181_log.yaml'))
