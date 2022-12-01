import urllib.request
import os
from persistence.repositories import paths


def download_video(video_url_input: str):
    # filename = video_url_input.rsplit('/', 1)[1]
    filename = "video.mp4"
    path = os.path.join(paths.video_input_path, filename)
    print(path)
    try:
        print("Downloading starts...\n")
        urllib.request.urlretrieve(video_url_input, path)
        print("Download completed..!!")
        return path
    except Exception as e:
        print(e.__str__())