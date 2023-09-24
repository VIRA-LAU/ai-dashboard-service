import os

def get_game_data(source: str = 'datasets/videos_input', game_id: str = ''):
    video = ''
    for vid in os.listdir(source):
        if (os.path.splitext(vid)[0] == game_id):
            video = vid

    videoPath = os.path.join(source, video)

    dataLogFilePath = os.path.join("datasets/logs", os.path.splitext(video)[0] + '_log.yaml')

    return videoPath, dataLogFilePath