import json
import sqlite3

import persistence.repositories.paths as path
import dev_utils.common.formatting as formatting

_conn = sqlite3.Connection
_cursor = sqlite3.Cursor
FRAMES_DIFF_FOR_SHOTS_MADE = 90
NUMBER_OF_FRAMES_AFTER_SHOOTING = 120
PERSON_ACTION_PRECISION = 0.92
PLAYER_WITH_BALL_PRECISION = 0.92


def connect_to_db(game_id: str):
    global _conn, _cursor
    _conn = sqlite3.connect(path.logs_path / f'{game_id}_logs.db')
    _cursor = _conn.cursor()
def getPostProcessingData():
    global _conn, _cursor
    availablePlayers = getAllPlayerIds()
    playerMap = getPlayerShotsPerFrame()
    basketCoords = getBasketCoordinatesPerFrame()
    netbasketCoords = getNetbasketCoordinatesPerFrame()

    playerCoords = {}
    for player in availablePlayers:
        playerCoords[player] = getPlayerCoordPerFrame(player)
    maxFrames = getMaxFrame()
    data = { 1 : {} }
    for player in availablePlayers:
        data[1][player] = {
            'score' : 0,
            'coord' : playerCoords[player][1]
        }
    data[1]['basket'] = basketCoords[1]
    data[1]['netbasket'] = netbasketCoords[1]
    data[1]['player_with_ball'] = getPlayerWithBall(1, availablePlayers, playerCoords, basketCoords[1])
    for i in range(2, maxFrames+1):
        data[i] = {}
        for player in availablePlayers:
            player_with_ball = getPlayerWithBall(i, availablePlayers, playerCoords, basketCoords[i])\
                if i in basketCoords.keys() else None
            data[i][player] = {
                'score' : data[i-1][player]['score'] + getAddedPlayerScoreInFrame(i, player, playerMap),
                'coord' : playerCoords[player][i] if i in playerCoords[player] else None
            }
        data[i]['basket'] = basketCoords[i] if i in basketCoords.keys() else None
        data[i]['netbasket'] = netbasketCoords[i] if i in netbasketCoords.keys() else None
        data[i]['player_with_ball'] = player_with_ball if player_with_ball is not None else data[i-1]['player_with_ball']
    return data


def getPlayerShotsPerFrame() -> dict:
    global _conn, _cursor
    _cursor.execute('''
        SELECT shots.frame_num, player_num, pose_db.bbox_coords,
        shots.bbox_coords, position FROM (SELECT frame_num,
        bbox_coords, ROW_NUMBER() OVER (ORDER BY frame_num)
        AS row_number from action_db where action = 'shooting')
        as shots JOIN pose_db ON shots.frame_num = pose_db.frame_num
    ''')
    rows = _cursor.fetchall()
    shooting_frames_with_players = []
    frames_added = []
    for row in rows:
        if len(frames_added) == 0 or row[0] - frames_added[-1] >= NUMBER_OF_FRAMES_AFTER_SHOOTING:
            shooting_frames_with_players.append({
                'frame': row[0],
                'player_num': row[1],
                'pose_coords': formatting.stringToListOfFloat(row[2]),
                'shot_coords': formatting.stringToListOfFloat(row[3]),
                'position': row[4]
            })
            frames_added.append(row[0])
    return getShotsPerPlayer(shooting_frames_with_players)


def getShotsPerPlayer(shooting_frames_with_players: list) -> dict:
    player_ids = getAllPlayerIds()
    playerMap = populatePlayerMap(player_ids)
    done_frames = []
    for element in shooting_frames_with_players:
        frame = element['frame']
        shooting_coords = element['shot_coords']
        player = element['player_num']
        player_bbox = element['pose_coords']
        check = []
        for i in range(len(shooting_coords)):
            check.append(min(player_bbox[i], shooting_coords[i]) /
                         max(player_bbox[i], shooting_coords[i]) >= PERSON_ACTION_PRECISION)
        if len(check) == 4 and all(check):
            if frame not in done_frames:
                shotmade, net_frame = checkForNetbasket(frame)
                added_points = 2 if element['position'] == '2_points' else 3
                if shotmade:
                    playerMap[player]['scorePerFrame'][net_frame] = added_points
                    playerMap[player]['recentScore'] += added_points
                done_frames.append(net_frame)
    return playerMap
def getBasketCoordinatesPerFrame() -> dict:
    global _conn, _cursor
    _cursor.execute('''
        SELECT frame_num, bbox_coords FROM basket_db WHERE shot = 'basketball'
    ''')
    rows = _cursor.fetchall()
    basket_coords = {}
    for row in rows:
        basket_coords[row[0]] = formatting.stringToListOfFloat(row[1])
    return basket_coords
def getAllPlayerIds() -> list[int]:
    global _conn, _cursor
    _cursor.execute('''
            SELECT player_num FROM pose_db GROUP BY player_num
        ''')
    rows = _cursor.fetchall()
    player_ids = []
    for row in rows:
        player_ids.append(row[0])
    return player_ids


def populatePlayerMap(player_ids: list) -> tuple[dict, dict]:
    playerMap = {}
    for player in player_ids:
        playerMap[player] = {
            'recentScore' : 0,
            'scorePerFrame' : {
                1 : 0
            }
        }
    return playerMap


def checkForNetbasket(frame_num: int) -> list:
    global _conn, _cursor
    table = _cursor.execute('''
        SELECT frame_num, shot FROM basket_db WHERE frame_num BETWEEN (?) AND (?)
    ''', [frame_num, frame_num + NUMBER_OF_FRAMES_AFTER_SHOOTING])
    rows = _cursor.fetchall()
    for row in rows:
        if (row[1]) == 'netbasket':
            return [True, row[0]]
    return [False, -1]
def getNetbasketCoordinatesPerFrame() -> dict:
    global _conn, _cursor
    max_frames = getMaxFrame()
    _cursor.execute('''
        SELECT frame_num, bbox_coords FROM basket_db WHERE shot = 'netbasket'
    ''')
    rows = _cursor.fetchall()
    frames_netbasket = set()
    coords_netbasket = dict()
    for row in rows:
        frames_netbasket.add(row[0])
        coords_netbasket[row[0]] = row[1]
    netbasket_coords = {}
    for i in range(1, max_frames+1):
        if i in frames_netbasket:
            netbasket_coords[i] = formatting.stringToListOfFloat(coords_netbasket[i])
        else:
            netbasket_coords[i] = []
    return netbasket_coords
def getPlayerCoordPerFrame(player_num: int):
    global _conn, _cursor
    _cursor.execute('''
        SELECT frame_num, bbox_coords FROM pose_db WHERE player_num = (?)
        ''', [player_num])
    rows = _cursor.fetchall()
    playerCoords = {}
    for row in rows:
        playerCoords[row[0]] = formatting.stringToListOfFloat(row[1])
    return playerCoords
def getMaxFrame():
    global _conn, _cursor
    _cursor.execute('''
        SELECT MAX(frame_num) from basket_db
    ''')
    rows = _cursor.fetchall()
    return rows[0][0]
def getAddedPlayerScoreInFrame(currentFrame: int, currentPlayer: int, playerMap: dict):
    if currentFrame in playerMap[currentPlayer]['scorePerFrame'].keys():
        return playerMap[currentPlayer]['scorePerFrame'][currentFrame]
    else:
        return 0
def getPlayerWithBall(frame: int, availablePlayers: list, playerCoords, basketCoordsAtFrame) -> int:
    for player in availablePlayers:
        currentPlayerCoords = playerCoords[player][frame] if frame in playerCoords[player] else None
        hasBall = False
        if currentPlayerCoords is not None:
            hasBall = checkIfPlayerHasBall(currentPlayerCoords, basketCoordsAtFrame)
        if hasBall:
            return player
    return None
def checkIfPlayerHasBall(playerCoords, basketballCoords):
    if (all(element is None for element in basketballCoords)):
        return False
    check = []
    player_x1, player_y1, player_x2, player_y2 = playerCoords
    basket_x1, basket_y1, basket_x2, basket_y2 = basketballCoords
    basket_x = (basket_x1 + basket_x2) / 2
    basket_y = (basket_y1 + basket_y2) / 2
    if player_x1 <= basket_x and basket_x <= player_x2:
        check.append(True)
    elif (getCoordsRatio(basket_x, player_x1) >= PLAYER_WITH_BALL_PRECISION
          or getCoordsRatio(basket_x, player_x2) >= PLAYER_WITH_BALL_PRECISION):
        check.append(True)
    else:
        check.append(False)
    if player_y1 <= basket_y and basket_y <= player_y2:
        check.append(True)
    elif (getCoordsRatio(basket_y, player_y1) >= PLAYER_WITH_BALL_PRECISION
          or getCoordsRatio(basket_y, player_y2) >= PLAYER_WITH_BALL_PRECISION):
        check.append(True)
    else:
        check.append(False)
    if len(check) == 2 and all(check):
        return True
    else:
        return False
def getCoordsRatio(coord1, coord2):
    return min(coord1, coord2) / max(coord1, coord2)

if __name__ == '__main__':
    connect_to_db('04181')
    data = getPostProcessingData()
    with open('data.json', 'w') as file:
        file.write(json.dumps(data))