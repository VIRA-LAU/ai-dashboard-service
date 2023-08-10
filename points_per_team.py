player_scores = {
    'Player 1': {'scored': 2, 'missed': 0},
    'Player 2': {'scored': 8, 'missed': 0},
    'Player 3': {'scored': 5, 'missed': 0},
    'Player 4': {'scored': 10, 'missed': 0},
    'Player 5': {'scored': 7, 'missed': 0},
    'Player 6': {'scored': 12, 'missed': 0}
}
team1 = [1, 3, 6]
team2 = [2, 4, 5]


def getPointsPerTeam():
    team1Points = 0
    team2Points = 0
    for player in player_scores:
        playerNum = int(player[-1])
        if playerNum in team1:
            team1Points += player_scores[player]['scored']
        if playerNum in team2:
            team2Points += player_scores[player]['scored']
    print(team1Points)
    print(team2Points)


if __name__ == '__main__':
    getPointsPerTeam()
