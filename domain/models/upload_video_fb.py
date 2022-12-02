import os

from domain.models.firebase_connection import bucket
from persistence.repositories import paths


def upload_video(id_user: str,
                 source_video: str = str(os.path.join(paths.concatenated_with_music, "movie.mp4"))) -> str:
    print(source_video)
    destination = "highlights/" + id_user + ".mp4"
    blob = bucket.blob(destination)  # parameter is the location where we want to store the file on firebase
    # where we want to store the videos
    # ask maria about it
    blob.upload_from_filename(source_video, timeout=200)
    blob.make_public()
    print("Firebase URL:", blob.public_url)
    return str(blob.public_url)
