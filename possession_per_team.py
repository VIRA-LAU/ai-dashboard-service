import yaml
import re

team1 = [1]
team2 = [2]
team1_frames = 0
team2_frames = 0

with open('datasets/logs/04181_log.yaml', 'r') as file:
    data = yaml.safe_load(file)
def getPossessionPerTeam():
    global team1_frames, team2_frames
    lastPlayerWithBall = None
    for frame in data:
        basketball_coords = getBasketballCoords(frame)
        all_players_coords = getAllPlayersCoords(frame)
        for player in all_players_coords:
            if(checkIfPlayerHasBall(player['coords'], basketball_coords)):
                if player['playerNum'] in team1:
                    team1_frames += 1
                    lastPlayerWithBall = player['playerNum']
                elif player['playerNum'] in team2:
                    team2_frames += 1
                    lastPlayerWithBall = player['playerNum']
                elif lastPlayerWithBall is not None:
                    if lastPlayerWithBall in team1: team1_frames += 1
                    elif lastPlayerWithBall in team2: team2_frames += 1
    print('Team 1 frames:',team1_frames)
    print('Team 2 frames:', team2_frames)


def getBasketballCoords(frame: int):
    basketballIndex = 0
    for frameInfo in data[frame]:
        if frameInfo.startswith('basketball_detection'):
            if data[frame][frameInfo] == 'basketball':
                basketballIndex = frameInfo[-1]
    return data[frame]['basketball_bbox_coords_' + str(basketballIndex)]
def getAllPlayersCoords(frame: int):
    allPlayersCoords = []
    for frameInfo in data[frame]:
        pattern = r'player_\d+_bbox_coords_\d+'
        if re.match(pattern, frameInfo):
            playerNum = frameInfo[frameInfo.find('_') + 1]
            currentPlayerCoords = data[frame][frameInfo]
            allPlayersCoords.append(
                {
                    'playerNum' : int(playerNum),
                    'coords' : currentPlayerCoords
                }
            )
    return allPlayersCoords
def checkIfPlayerHasBall(playerCoords: list, basketballCoords: list):
    check = []
    player_x1, player_y1, player_x2, player_y2 = playerCoords
    basket_x1, basket_y1, basket_x2, basket_y2 = basketballCoords
    basket_x = (basket_x1 + basket_x2)/2
    basket_y = (basket_y1 + basket_y2)/2
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

def getCoordsRatio(coord1 :int, coord2: int):
    return min(coord1, coord2)/max(coord1,coord2)

if __name__ == '__main__':
    getPossessionPerTeam()