import pandas as pd

def getPointsPerPlayer():
    actions = pd.read_csv('datasets/logs/actions_1.pt/20230418_144556_logs.csv')
    player = pd.read_csv('datasets/logs/yolov7-w6-pose.pt/20230418_144556_logs.csv')
    baskets = pd.read_csv('datasets/logs/net_hoop_basket_3.pt/20230418_144556_logs.csv')
    frames = []
    index = 30
    map = {}
    NUMBER_OF_FRAMES_PER_SHOOTING_ACTION = 30
    NUMBER_OF_FRAMES_AFTER_SHOOTING_FOR_NETBASKET = 90
    PLAYER_CONFIDENCE_THRESHOLD = 0.95

    for element in actions.values:
        if element[-2] == 'shooting' and index >= NUMBER_OF_FRAMES_PER_SHOOTING_ACTION:
            frames.append([element[2], element[-1]])
            index = 0
        index += 1
    if len(frames) == 0:
        return {}
    for element in frames:
        players = []
        for entry in player.values:
            if(entry[2] == element[0]):
                players.append(entry)
        if len(players) == 0:
            return {}
        for scorer in players:
            coord_shooting = element[-1]
            coord_shooting = coord_shooting[1:-1].split(', ')
            coord_player = scorer[-3]
            coord_player = coord_player[1:-1].split(', ')
            frame_num = element[0]
            player_id = ''
            point_scored = 'None'
            check = []
            for i in range(len(coord_player)):
                if min(float(coord_player[i]), float(coord_shooting[i]))/max(float(coord_player[i]),float(coord_shooting[i])) > PLAYER_CONFIDENCE_THRESHOLD:
                    check.append(True)
                else:
                    check.append(False)
            if all(check):
                player_id = scorer[-1]
                index = 0
                for i in range(len(baskets.values)):
                    if baskets.values[i][2] == frame_num:
                        index = i
                        break
                for i in range(NUMBER_OF_FRAMES_AFTER_SHOOTING_FOR_NETBASKET):
                    if baskets.values[index][4] == 'netbasket' and baskets.values[index-1][4] == 'netbasket':
                        point_scored = scorer[-2]
                    index +=1
            if map.get(player_id) == None:
                map[player_id] = 0 if point_scored == 'None' else 2 if point_scored == '2_points' else 3 if point_scored == '3_points' else 0
            else:
                add = 0 if point_scored == 'None' else 2 if point_scored == '2_points' else 3 if point_scored == '3_points' else 0
                map[player_id] += add
            return map

            
            


