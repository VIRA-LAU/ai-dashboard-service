import os.path
import shutil

from fastapi import UploadFile


def save_video(video: UploadFile, destination: str) -> str:
    """
    Saves the video uploaded to the endpoint locally
    :param video:
    :param destination:
    :return:
    """
    video_path=os.path.join(destination,video.filename)
    with open(video_path, 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

        return video_path
