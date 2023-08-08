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
for frame in data:
        for frameInfo in data[frame]:
                if frameInfo.startswith('action_detection'):
                        if data[frame][frameInfo] == 'shooting':
                                index = frameInfo[-1]
                                shooting_coords = data[frame]['action_bbox_coords_' + index]
                                playerCoords = None
                                for stat in data[frame]:
                                        pattern = r'player_\d+_bbox_coords_\d+'
                                        if re.match(pattern, stat):
                                                currentCoords = data[frame][stat]
                                                print(currentCoords)
                                                print(shooting_coords)
                                                check = []
                                                for i in range(len(shooting_coords)):
                                                        check.append(min(currentCoords[i], shooting_coords[i])/ max(currentCoords[i], shooting_coords[i]) >= PERSON_ACTION_PRECISION)
                                                print(check)

                                # if len(framesWithShootingPlayer) == 0:
                                #         framesWithShootingPlayer.append({'frame' : frame, 'index' : frameInfo[-1]})
                                # elif frame - framesWithShootingPlayer[-1]['frame'] >= NUMBER_OF_FRAMES_PER_SHOOTING:
                                #         framesWithShootingPlayer.append(frame)
print(framesWithShootingPlayer)
for frameObject in framesWithShootingPlayer:
        frameNum = frameObject['frame']
        index = frameObject['index']
        pointLogged = False
        for frame in range(frameNum, frameNum + NETBASKET_FRAMES_AFTER_SHOOTING):
                if pointLogged:
                        break
                for frameInfo in data[frame]:
                        if frameInfo.startswith('basketball_detection'):
                                if data[frame][frameInfo] == 'netbasket':
                                        playerPoints = 0
                                        playerNum = 0
                                        for detection in data[frame]:
                                                if detection.endswith('_position_' + index):
                                                     playerPoints = 2 if data[frame][detection] == '2_points' else 3
                                                     playerNum = detection[detection.find('_') + 1]
                                        if playerPoints != 0 and playerNum != 0:
                                                playerIdentifier = 'Player ' + playerNum
                                                if playerIdentifier not in pointsForAllPlayers.keys():
                                                        pointsForAllPlayers[playerIdentifier] = 0
                                                pointsForAllPlayers[playerIdentifier] += playerPoints
                                                pointLogged = True
print(pointsForAllPlayers)







