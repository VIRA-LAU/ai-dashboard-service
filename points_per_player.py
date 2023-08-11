import math
import re
import yaml
import cv2
import os

data = {}
framesWithShootingPlayer = []
pointsForAllPlayers = {}
PERSON_ACTION_PRECISION = 0.92


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
    print(getPointsPerPlayer('datasets/logs/04183_log.yaml'))
