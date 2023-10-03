import os
import persistence.repositories.paths as paths

def get_game_data(game_id: str = ''):
    source: str = paths.video_input_path
    video = ''
    for vid in os.listdir(source):
        if (os.path.splitext(vid)[0] == game_id):
            video = vid

    videoPath = os.path.join(source, video)

    dataLogFilePath = paths.logs_path / f'{os.path.splitext(video)[0]}_log.yaml'

    return videoPath, dataLogFilePath
if __name__ == '__main__':
    print(get_game_data(game_id='04181'))