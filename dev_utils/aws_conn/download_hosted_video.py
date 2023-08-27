import requests
import persistence.repositories.paths as paths
import dev_utils.paths.dirs as dirs

def download_video(game_id: str):
    response = requests.get(paths.hosted_videos_input_path + game_id + '.mp4')
    file_path = dirs.getRootDir() / f'datasets/videos_input/{game_id}.mp4'
    with open(file_path, 'wb') as file:
        file.write(response.content)
    print("Video downloaded successfully!")
if __name__ == '__main__':
    print('main')
