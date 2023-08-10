import re
import yaml

data = {}
framesWithShootingPlayer = []
pointsForAllPlayers = {}
NUMBER_OF_FRAMES_PER_SHOOTING = 60
NETBASKET_FRAMES_AFTER_SHOOTING = 90
PERSON_ACTION_PRECISION = 0.92
with open('datasets/logs/04181_log.yaml', 'r') as file:
        data = yaml.safe_load(file)
def getPointsPerPlayer():
        for frame in data:
                for frameInfo in data[frame]:
                        shooting_coords = getShootingCoords(frame, frameInfo)
                        playerCoords, playerNum = [None, None]
                        for stat in data[frame]:
                                if playerCoords is not None and playerNum is not None:
                                        break
                                playerCoords, playerNum = getShootingPlayer(frame, stat, shooting_coords)
                        addShootingPlayerToGlobalMap(frame, frameInfo, playerCoords, playerNum)
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
                addMissedPointToPlayer(playerIdentifier, pointLogged)
                return pointsForAllPlayers

def getShootingCoords(frame: int, frameInfo: str):
        shooting_coords = []
        if frameInfo.startswith('action_detection'):
                key = frameInfo
                if data[frame][frameInfo] == 'shooting':
                        index = frameInfo[-1]
                        shooting_coords = data[frame]['action_bbox_coords_' + index]
        return shooting_coords
def getShootingPlayer(frame: int, stat: str, shooting_coords: list):
        playerInfo = [None, None]
        pattern = r'player_\d+_bbox_coords_\d+'
        if re.match(pattern, stat):
                currentCoords = data[frame][stat]
                check = []
                for i in range(len(shooting_coords)):
                        check.append(min(currentCoords[i], shooting_coords[i]) /
                                     max(currentCoords[i],shooting_coords[i]) >= PERSON_ACTION_PRECISION)
                if len(check) == 4 and all(check):
                        playerInfo[0] = currentCoords
                        playerInfo[1] = stat[stat.find('_') + 1]
        return playerInfo
def addShootingPlayerToGlobalMap(frame: int, frameInfo: str, playerCoords: list, playerNum: str):
        if playerCoords is not None and playerNum is not None:
                if len(framesWithShootingPlayer) == 0:
                        framesWithShootingPlayer.append(
                                {'frame': frame, 'index': frameInfo[-1], 'playerNum': playerNum})
                elif frame - framesWithShootingPlayer[-1]['frame'] >= NUMBER_OF_FRAMES_PER_SHOOTING:
                        framesWithShootingPlayer.append(
                                {'frame': frame, 'index': frameInfo[-1], 'playerNum': playerNum})
def addPointToPlayer(frame: int, frameInfo: str, playerIdentifier: str, index: str):
        if frameInfo.startswith('basketball_detection'):
                if data[frame][frameInfo] == 'netbasket':
                        playerPoints = 0
                        playerNum = 0
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
                if playerIdentifier is not None and playerIdentifier not in pointsForAllPlayers.keys():
                        pointsForAllPlayers[playerIdentifier]['missed'] += 1

if __name__ == '__main__':
        print(getPointsPerPlayer())