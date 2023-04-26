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
        for idx, row in baskets.loc[baskets['labels'] == 'netbasket'].iterrows():
            if len(frames) == 0:
                frames.append(row['frame'])
            elif row['frame'] - frames[-1] > NUMBER_OF_FRAMES_FOR_NETBASKET:
                frames.append(row['frame'])
    player = pd.read_csv(f'datasets/logs/yolov7-w6-pose.pt/{fileName}_logs.csv')
    print(frames)

    for frame_num in frames:
        if currentFrame == frame_num:
            for index, row in actions[(actions['frame'] >= frame_num - NUMBER_OF_FRAMES_AFTER_SHOOTING) & (actions['frame'] <= frame_num)].iterrows():
                if row['labels'] == 'shooting':
                    coord_action = row['bbox_coord']
                    shooting_frame = row['frame']
                    break
            if coord_action is not None and shooting_frame is not None:
                players = player.loc[player['frame'] == shooting_frame][['player_coordinates', 'playerId', 'position']].values
                print(shooting_frame)
                player_id = ''
                point_scored = 'None'
                coord_action = np.fromstring(coord_action[1:-2], sep = ',')
                for player_coord, playerId, position in players:
                    player_coord = np.fromstring(player_coord[1:-2], sep = ',')
                    check = []
                    for i in range(len(player_coord)):
                        if min((player_coord[i],coord_action[i])/max(player_coord[i],coord_action[i])) > PLAYER_CONFIDENCE_LEVEL:
                            check.append(True)
                        else:
                            check.append(False)
                    if all(check):
                        player_id = playerId
                        point_scored = position
 
                        if map.get(player_id) == None:
                            map[player_id] = 0 if point_scored == 'None' else 2 if point_scored == '2_points' else 3 if point_scored == '3_points' else 0
                        else:
                            add = 0 if point_scored == 'None' else 2 if point_scored == '2_points' else 3 if point_scored == '3_points' else 0
                            map[player_id] += add
    return map

                
                


