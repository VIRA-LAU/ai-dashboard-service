import pandas as pd
import numpy as np
map = {}

actions = None
baskets = None
frames = []

NUMBER_OF_FRAMES_AFTER_SHOOTING = 90
NUMBER_OF_FRAMES_FOR_NETBASKET = 120
PLAYER_CONFIDENCE_LEVEL = 0.92


def getPointsPerPlayer(currentFrame : int, fileName : str):
    global actions, baskets, frames
    if actions is None and baskets is None and len(frames) == 0:
        actions = pd.read_csv(f'datasets/logs/actions_2.pt/{fileName}_logs.csv')
        baskets = pd.read_csv(f'datasets/logs/net_hoop_basket_5.pt/{fileName}_logs.csv')
        for idx, row in actions.loc[actions['labels'] == 'shooting'].iterrows():
            if len(frames) == 0:
                frames.append([row['frame'], row['bbox_coord']])
            elif row['frame'] - frames[-1][0] > NUMBER_OF_FRAMES_AFTER_SHOOTING:
                frames.append([row['frame'], row['bbox_coord']])
    player = pd.read_csv(f'datasets/logs/yolov7-w6-pose.pt/{fileName}_logs.csv')
    for frame_num, coord_action in frames:
        if currentFrame == frame_num:
            players = player.loc[player['frame'] == currentFrame][['player_coordinates', 'playerId', 'position']].values
            print(players)
            player_id = ''
            point_scored = 'None'
            for player_coord, playerId, position in players:
                player_coord = np.fromstring(player_coord[1:-2], sep = ',')
                coord_action = np.fromstring(coord_action[1:-2], sep = ',')
                check = []
                for i in range(len(player_coord)):
                    if min((player_coord[i],coord_action[i])/max(player_coord[i],coord_action[i])) > PLAYER_CONFIDENCE_LEVEL:
                        check.append(True)
                    else:
                        check.append(False)
                if all(check):
                    print(check)
                    player_id = playerId
                    for index, row in baskets[(baskets['frame'] >= frame_num) & (baskets['frame'] <= frame_num + NUMBER_OF_FRAMES_FOR_NETBASKET)].iterrows():
                        if row['labels'] == 'netbasket':
                            point_scored = position
                            print(point_scored)
                            break
                    if map.get(player_id) == None:
                        map[player_id] = 0 if point_scored == 'None' else 2 if point_scored == '2_points' else 3 if point_scored == '3_points' else 0
                    else:
                        add = 0 if point_scored == 'None' else 2 if point_scored == '2_points' else 3 if point_scored == '3_points' else 0
                        map[player_id] += add
                    print(map)
    return map

                
                


