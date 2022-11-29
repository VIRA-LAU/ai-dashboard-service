import os

from domain.models.firebase_connection import bucket
from persistence.repositories import paths


def upload_video(destination: str, source_video: str = paths.concatenated_path):
    blob = bucket.blob(destination)  # parameter is the location where we want to store the file on firebase
    # where we want to store the videos
    # ask maria about it
    blob.upload_from_filename(source_video, timeout=200)
    blob.make_public()
    print("Firebase URL:", blob.public_url)
