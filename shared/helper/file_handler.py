import shutil

from fastapi import UploadFile


def save_video(video: UploadFile, destination: str) -> str:
    """
    Saves the video uploaded to the endpoint locally
    :param video:
    :param destination:
    :return:
    """
    with open(destination + '{}'.format(video.filename), 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

        return destination + '{}'.format(video.filename)
