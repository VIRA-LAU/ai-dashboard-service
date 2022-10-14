import shutil

from fastapi import UploadFile


def save_video(video: UploadFile, destination: str) -> str:
    with open(destination + '{}'.format(video.filename), 'wb') as buffer:
        shutil.copyfileobj(video.file, buffer)

        return destination + '{}'.format(video.filename)
