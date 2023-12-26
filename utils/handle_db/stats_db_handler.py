import json
import sqlite3
import math

import persistence.repositories.paths as path
import utils.common.formatting as formatting

import torch
from utils.general import scale_coords

FRAMES_DIFF_FOR_SHOTS_MADE = 90
NUMBER_OF_FRAMES_AFTER_SHOOTING = 60
NUMBER_OF_FRAMES_AFTER_ASSISTS = 90
PERSON_ACTION_PRECISION = 0.92
PLAYER_WITH_BALL_PRECISION = 0.92

class Stats_DB_Handler:
    def __init__(self, game_id: str, team1: list, team2: list):
        self.game_id = game_id
        self._conn = sqlite3.Connection
        self._cursor = sqlite3.Cursor

        self.team1 = team1
        self.team2 = team2

        self.connect_to_db()
        self.availablePlayers = self.getAllPlayerIds()
        self.playerMap, self.teamMap = self.getPlayerShotsPerFrame()
        self.basketCoords = self.getBasketCoordinatesPerFrame()
        self.netbasketCoords = self.getNetbasketCoordinatesPerFrame()
        self.team1_points = self.getTotalPerTeam(self.teamMap[0])
        self.team2_points = self.getTotalPerTeam(self.teamMap[1])
        
        self.post_process_data = self.getPostProcessingData()


    def connect_to_db(self):
        self._conn = sqlite3.connect(path.logs_path / f'{self.game_id}_logs.db')
        self._cursor = self._conn.cursor()


    def getPostProcessingData(self):
        playerCoords = {}
        for player in self.availablePlayers:
            playerCoords[player] = self.getPlayerCoordPerFrame(player)
        maxFrames = self.getMaxFrame()

        data = { 1 : {'players':{}} }
        for player in self.availablePlayers:
            player_team = 'team_1' if player in self.team1 else 'team_2'

            data[1]['players'][player] = {
                'score' : 0,
                'coord' : playerCoords[player][list(playerCoords[player].keys())[0]]
            }

        data[1]['basket'] = self.basketCoords[list(self.basketCoords.keys())[0]]
        data[1]['netbasket'] = self.netbasketCoords[list(self.netbasketCoords.keys())[0]]

        data[1]['player_with_ball'] = {
            'player': self.getPlayerWithBall(1, playerCoords, self.basketCoords[list(self.basketCoords.keys())[0]]),
            'team': player_team
        }

        for i in range(2, maxFrames+1):
            data[i] = {'players': {}}
            for player in self.availablePlayers:
                player_team = 'team_1' if player in self.team1 else 'team_2'
                player_with_ball = self.getPlayerWithBall(i, playerCoords, self.basketCoords[i]) if i in self.basketCoords.keys() else None

                data[i]['players'][player] = {
                    'score' : data[i-1]['players'][player]['score'] + self.getAddedPlayerScoreInFrame(i, player, self.playerMap),
                    'coord' : playerCoords[player][i] if i in playerCoords[player] else None
                }

            data[i]['basket'] = self.basketCoords[i] if i in self.basketCoords.keys() else None
            data[i]['netbasket'] = self.netbasketCoords[i] if i in self.netbasketCoords.keys() else None

            # COMMENT OUT
            # data[i]['player_with_ball'] = {
            # 'player': player_with_ball if player_with_ball is not None else data[i-1]['player_with_ball'],
            # 'team': player_team
            # }

            # TO REMOVE
            ##############################################################################################################
            if i < 400:
                data[i]['player_with_ball'] = {
                    'player': 1,
                    'team': 'team_1'
                }
            elif 400 < i < 460:
                data[i]['player_with_ball'] = {
                    'player': 2,
                    'team': 'team_1'
                }
            else:
                data[i]['player_with_ball'] = {
                    'player': 2,
                    'team': 'team_2'
                }
            ##############################################################################################################
        return data


    def getPlayerShotsPerFrame(self) -> dict:
        self._cursor.execute('''
            SELECT shots.frame_num, player_num, person_db.bbox_coords,
            shots.bbox_coords, position FROM (SELECT frame_num,
            bbox_coords, ROW_NUMBER() OVER (ORDER BY frame_num)
            AS row_number from action_db where action = 'shooting')
            as shots JOIN person_db ON shots.frame_num = person_db.frame_num
        ''')
        rows = self._cursor.fetchall()
        shooting_frames_with_players = []
        frames_added = []
        for row in rows:
            if len(frames_added) == 0 or row[0] - frames_added[-1] >= NUMBER_OF_FRAMES_AFTER_SHOOTING:
                shooting_frames_with_players.append({
                    'frame': row[0],
                    'player_num': row[1],
                    'person_coords': formatting.stringToListOfFloat(row[2]),
                    'shot_coords': formatting.stringToListOfFloat_Shot(row[3]),
                    'position': row[4]
                })
                frames_added.append(row[0])
        return self.getShotsPerPlayer(shooting_frames_with_players)


    def getShotsPerPlayer(self, shooting_frames_with_players: list) -> dict:
        playerMap = self.populatePlayerMap()
        team1_player_scores, team2_player_scores = self.populateTeamLists()
        done_frames = []
        for element in shooting_frames_with_players:
            frame = element['frame']
            shooting_coords = element['shot_coords']
            player = element['player_num']
            player_bbox = element['person_coords']
            check = []
            for i in range(len(shooting_coords)):
                check.append(min(player_bbox[i], shooting_coords[i]) /
                            max(player_bbox[i], shooting_coords[i]) >= PERSON_ACTION_PRECISION)
            if len(check) == 4:
                if frame not in done_frames:
                    shotmade, net_frame = self.checkForNetbasket(frame)
                    added_points = 2 if element['position'] == '2_points' else 3
                    if shotmade:
                        playerMap[player]['scorePerFrame'][net_frame] = added_points
                        playerMap[player]['recentScore'] += added_points
                        playerMap[player]['2points'] += 1 if element['position'] == '2_points' else playerMap[player]['2points']
                        playerMap[player]['3points'] += 1 if element['position'] == '3_points' else playerMap[player]['3points']
                        playerMap[player]['shotsmade'] += 1
                        if player in self.team1:
                            team1_player_scores[f'player_{player}']['points'] += added_points
                            team1_player_scores[f'player_{player}']['scored'] += 1 
                        elif player in self.team2:
                            team2_player_scores[f'player_{player}']['points'] += added_points
                            team2_player_scores[f'player_{player}']['scored'] += 1
                    else:
                        playerMap[player]['shotsmissed'] += 1
                        if player in self.team1:
                            team1_player_scores[f'player_{player}']['missed'] += 1
                        elif player in self.team2:
                            team2_player_scores[f'player_{player}']['missed'] += 1
                    playerMap[player]['shots_accuracy'] = round(playerMap[player]['shotsmade'] / (playerMap[player]['shotsmade'] + playerMap[player]['shotsmissed']), 2)
                    done_frames.append(net_frame)
        teamMap = (team1_player_scores, team2_player_scores)
        return playerMap, teamMap


    def getTotalPerTeam(self, shotsPerTeam: dict):
        total = 0
        for player in shotsPerTeam.keys():
            total += shotsPerTeam[player]['points']
        return total


    def getBasketCoordinatesPerFrame(self) -> dict:
        self._cursor.execute('''
            SELECT frame_num, bbox_coords FROM basket_db WHERE shot = 'basketball'
        ''')
        rows = self._cursor.fetchall()
        basket_coords = {}
        for row in rows:
            basket_coords[row[0]] = formatting.stringToListOfFloat_Shot(row[1])
        return basket_coords


    def getAllPlayerIds(self) -> 'list[int]':
        self._cursor.execute('''
                SELECT player_num FROM person_db GROUP BY player_num
            ''')
        rows = self._cursor.fetchall()
        player_ids = []
        for row in rows:
            player_ids.append(row[0])
        return player_ids


    def populatePlayerMap(self) -> 'tuple[dict, dict]':
        playerMap = {}
        for player in self.availablePlayers:
            playerMap[player] = {
                'recentScore' : 0,
                'scorePerFrame' : {1 : 0},
                '2points' : 0,
                '3points' : 0,
                'shotsmade' : 0,
                'shotsmissed' : 0,
                'shots_accuracy': 0.0
            }
        return playerMap


    def populateTeamLists(self) -> 'tuple[dict, dict]':
        team1_player_scores = {}
        team2_player_scores = {}
        for player in self.availablePlayers:
            if player in self.team1:
                team1_player_scores[f'player_{player}'] = {
                    'scored': 0,
                    'missed': 0,
                    'points': 0
                }
            elif player in self.team2:
                team2_player_scores[f'player_{player}'] = {
                    'scored': 0,
                    'missed': 0,
                    'points': 0
                }
        return team1_player_scores, team2_player_scores


    def getAPIStats(self):
        team_map = {
            'team1': {},
            'team2': {}
        }
        for player in self.playerMap:
            if player in self.team1:
                team_map['team1'][player] = self.playerMap[player]
            elif player in self.team2:
                team_map['team2'][player] = self.playerMap[player]

        passes, passes_frames = self.getPasses()
        assists = self.getAssists(passes_frames)
        possession = self.getPossession()

        api_stats = {
            'team_1' : {
                'players' : team_map['team1'],
                'points' : self.team1_points,
                'possession': possession['team_1_possession'],
            },
            'team_2' : {
                'players' : team_map['team2'],
                'points' : self.team2_points,
                'possession': possession['team_2_possession'],
            },
            'total_passes': passes,
            'total_assists': assists
        }

        return api_stats


    def checkForNetbasket(self, frame_num: int) -> list:
        self._cursor.execute('''
            SELECT frame_num, shot FROM basket_db WHERE frame_num BETWEEN (?) AND (?)
        ''', [frame_num, frame_num + NUMBER_OF_FRAMES_AFTER_SHOOTING])
        rows = self._cursor.fetchall()
        for row in rows:
            if (row[1]) == 'netbasket':
                return [True, row[0]]
        return [False, -1]


    def getNetbasketCoordinatesPerFrame(self) -> dict:
        max_frames = self.getMaxFrame()
        self._cursor.execute('''
            SELECT frame_num, bbox_coords FROM basket_db WHERE shot = 'netbasket'
        ''')
        rows = self._cursor.fetchall()
        frames_netbasket = set()
        coords_netbasket = dict()
        for row in rows:
            frames_netbasket.add(row[0])
            coords_netbasket[row[0]] = row[1]
        netbasket_coords = {}
        for i in range(1, max_frames+1):
            if i in frames_netbasket:
                netbasket_coords[i] = formatting.stringToListOfFloat_Shot(coords_netbasket[i])
            else:
                netbasket_coords[i] = []
        return netbasket_coords


    def getPlayerCoordPerFrame(self, player_num: int):
        self._cursor.execute('''
            SELECT frame_num, bbox_coords FROM person_db WHERE player_num = (?)
            ''', [player_num])
        rows = self._cursor.fetchall()
        playerCoords = {}
        for row in rows:
            playerCoords[row[0]] = formatting.stringToListOfFloat(row[1])
        return playerCoords


    def getMaxFrame(self):
        self._cursor.execute('''
            SELECT MAX(frame_num) from basket_db
        ''')
        rows = self._cursor.fetchall()
        return rows[0][0]


    def getAddedPlayerScoreInFrame(self, currentFrame: int, currentPlayer: int, playerMap: dict):
        if currentFrame in playerMap[currentPlayer]['scorePerFrame'].keys():
            return playerMap[currentPlayer]['scorePerFrame'][currentFrame]
        else:
            return 0
        

    def getPlayerWithBall(self, frame: int, playerCoords, basketCoordsAtFrame) -> int:

        # TO REMOVE
        ######################
        ratio = 1920/1088
        new_h = int(640/ratio)
        ######################

        for player in self.availablePlayers:
            currentPlayerCoords = playerCoords[player][frame] if frame in playerCoords[player] else None
            hasBall = False
            if currentPlayerCoords is not None:

                # TO REMOVE
                ##############################################################################################################
                r_bbox = scale_coords((new_h, 640), torch.Tensor([currentPlayerCoords]), (1088, 1920), kpt_label=False).round()
                p_x1, p_y1, p_x2, p_y2  = r_bbox.cpu().numpy()[0]
                p_y1 -= 30
                p_y2 -= 30
                currentPlayerCoords = [p_x1, p_y1, p_x2, p_y2]
                ##############################################################################################################

                check = []
                for i in range(len(currentPlayerCoords)):
                    check.append(min(currentPlayerCoords[i], basketCoordsAtFrame[i]) /
                                max(currentPlayerCoords[i], basketCoordsAtFrame[i]) >= PLAYER_WITH_BALL_PRECISION)
                if len(check) == 4:
                    hasBall = True
                # hasBall = checkIfPlayerHasBall(currentPlayerCoords, basketCoordsAtFrame)
            if hasBall:
                return player
        return None


    def getPossession(self):
        team1_frames = 0
        team2_frames = 0
        for frame in self.post_process_data:
            if self.post_process_data[frame]['player_with_ball']['team'] == 'team_1':
                team1_frames+=1
            elif self.post_process_data[frame]['player_with_ball']['team'] == 'team_2':
                team2_frames+=1
        team1_possession = math.floor(team1_frames/(team1_frames+team2_frames)*100)
        team2_possession = math.ceil(team2_frames/(team1_frames+team2_frames)*100)

        possession_data = {
            'team_1_possession': team1_possession,
            'team_2_possession': team2_possession
        }
        return possession_data
    

    def getPasses(self):
        passes = -1
        passes_frames = []
        last_player_with_ball = None
        for frame in self.post_process_data:
            if self.post_process_data[frame]['player_with_ball']['player'] != last_player_with_ball:
                passes +=1
                last_player_with_ball = self.post_process_data[frame]['player_with_ball']['player']
                passes_frames.append(frame)
        return passes, passes_frames


    def getAssists(self, passes_frames: list):
        assists = 0
        for frame in passes_frames:
            if frame + NUMBER_OF_FRAMES_AFTER_ASSISTS in self.post_process_data:
                if self.post_process_data[frame + NUMBER_OF_FRAMES_AFTER_ASSISTS]['netbasket'] != []:
                    assists += 1
        return assists
    

    def getNetbasketCoordinatesFrames(self) -> 'list[list, int]':
        self._cursor.execute('''
            SELECT frame_num FROM basket_db WHERE shot = 'netbasket'
        ''')
        rows = self._cursor.fetchall()
        frames_netbasket = []
        for row in rows:
            if len(frames_netbasket) == 0 or row[0] - frames_netbasket[-1] >= 60:
                frames_netbasket.append(row[0])
        return [frames_netbasket, len(frames_netbasket)]