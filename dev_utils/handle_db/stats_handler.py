import sqlite3

import persistence.repositories.paths as path
import dev_utils.common.formatting as formatting

_conn = sqlite3.Connection
_cursor = sqlite3.Cursor
FRAMES_DIFF_FOR_SHOTS_MADE = 90
NUMBER_OF_FRAMES_AFTER_SHOOTING = 120
PERSON_ACTION_PRECISION = 0.92


def connect_to_db(game_id: str):
    global _conn, _cursor
    _conn = sqlite3.connect(path.logs_path / f'{game_id}_logs.db')
    _cursor = _conn.cursor()


def getShotsPerTeam() -> tuple[dict, dict]:
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
    return getShotsPerPlayer(shooting_frames_with_players, [1], [2])


def getShotsPerPlayer(shooting_frames_with_players: list, team1: list, team2: list) -> tuple[dict, dict]:
    player_ids = getAllPlayerIds()
    team1_player_scores, team2_player_scores = populateTeamLists(player_ids, team1, team2)
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
                shotmade = checkForNetbasket(frame)
                added_points = 2 if element['position'] == '2_points' else 3
                if player in team1 and shotmade:
                    team1_player_scores[f'player_{player}']['scored'] += added_points
                elif player in team2 and shotmade:
                    team2_player_scores[f'player_{player}']['scored'] += added_points
                elif player in team1 and not shotmade:
                    team1_player_scores[f'player_{player}']['missed'] += 1
                elif player in team2 and not shotmade:
                    team2_player_scores[f'player_{player}']['missed'] += 1
                done_frames.append(frame)
    return team1_player_scores, team2_player_scores

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


def populateTeamLists(player_ids: list, team1: list, team2: list) -> tuple[dict, dict]:
    team1_player_scores = {}
    team2_player_scores = {}
    for player in player_ids:
        if player in team1:
            team1_player_scores[f'player_{player}'] = {
                'scored': 0,
                'missed': 0
            }
        elif player in team2:
            team2_player_scores[f'player_{player}'] = {
                'scored': 0,
                'missed': 0
            }
    return team1_player_scores, team2_player_scores


def checkForNetbasket(frame_num: int) -> bool:
    global _conn, _cursor
    table = _cursor.execute('''
        SELECT shot FROM basket_db WHERE frame_num BETWEEN (?) AND (?)
    ''', [frame_num, frame_num + NUMBER_OF_FRAMES_AFTER_SHOOTING])
    rows = _cursor.fetchall()
    for row in rows:
        if (row[0]) == 'netbasket':
            return True
    return False

def getTotalPerTeam(shotsPerTeam: dict):
    total = 0
    for player in shotsPerTeam.keys():
        total += shotsPerTeam[player]['scored']
    return total
# def getPossession():
def getPlayersAndBasket():
    global _conn, _cursor
    table = _cursor.execute('''
        SELECT pose_db.frame_num, player_num, pose_db.bbox_coords,
        basket_db.bbox_coords FROM pose_db
        JOIN basket_db on pose_db.frame_num = basket_db.frame_num
        WHERE  basket_db.shot = 'basketball'
    ''')
    rows = _cursor.fetchall()
    print(rows)


if __name__ == '__main__':
    connect_to_db('04183')
    team1, team2 = getShotsPerTeam()
    print(team1)
    print(team2)
