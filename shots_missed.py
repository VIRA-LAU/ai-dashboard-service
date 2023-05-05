import pandas as pd
import numpy as np
map = {}

actions = None
baskets = None
frames = []



def getShotsMissedPerPlayer(currentFrame : int, fileName : str, shooting_frames: int = 90, netbasket_frames: int = 120, conf: float = 0.92):
    NUMBER_OF_FRAMES_AFTER_SHOOTING = shooting_frames
    NUMBER_OF_FRAMES_FOR_NETBASKET = netbasket_frames
    PLAYER_CONFIDENCE_LEVEL = conf
    global actions, baskets, frames, map
    if actions is None and baskets is None and len(frames) == 0:
        actions = pd.read_csv(f'datasets/logs/actions_2.pt/{fileName}_actions_logs.csv')
        baskets = pd.read_csv(f'datasets/logs/net_hoop_basket_combined_april.pt/{fileName}_nethoopbasket_logs.csv')
        for idx, row in actions.loc[actions['labels'] == 'shooting'].iterrows():
            if len(frames) == 0:
                frames.append([row['frame'], row['bbox_coord']])
            elif row['frame'] - frames[-1][0] > NUMBER_OF_FRAMES_AFTER_SHOOTING:
                frames.append([row['frame'], row['bbox_coord']])
    player = pd.read_csv(f'datasets/logs/yolov7-w6-pose.pt/{fileName}_pose_logs.csv')

    for frame_num, coord_action in frames:
        shotmade = False
        if currentFrame == frame_num:
            for index, row in baskets[(baskets['frame'] <= frame_num + NUMBER_OF_FRAMES_FOR_NETBASKET) & (baskets['frame'] >= frame_num)].iterrows():
                if row['labels'] == 'netbasket':
                    shotmade = True
                    break
            if coord_action is not None and shotmade == False:
                players = player.loc[player['frame'] == frame_num][['player_coordinates', 'playerId']].values
                
                players = player.loc[player['frame'] == currentFrame][['player_coordinates', 'playerId']].values
                player_id = ''
                for player_coord, playerId in players:
                    if player_coord is not None:
                        player_coord = np.fromstring(player_coord[1:-2], sep = ',')
                        coord_action = np.fromstring(coord_action[1:-2], sep = ',')
                        check = []
                        for i in range(len(player_coord)):
                            if min((player_coord[i],coord_action[i])/max(player_coord[i],coord_action[i])) > PLAYER_CONFIDENCE_LEVEL:
                                check.append(True)
                            else:
                                check.append(False)
                        if all(check):
                            player_id = playerId
    
                            if map.get(player_id) == None:
                                map[player_id] = 1
                            else:
                                map[player_id] += 1
    return map
                


