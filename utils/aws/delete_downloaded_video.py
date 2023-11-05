import os
from persistence.repositories import paths
def delete_downloaded_video(game_id: str):
    for vid in os.listdir(paths.video_input_path):
        if vid.startswith(game_id):
            os.remove(paths.video_input_path / vid)
if __name__ == '__main__':
    delete_downloaded_video('f1tch41n')