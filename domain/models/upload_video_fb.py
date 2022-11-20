import os

from domain.models.firebase_connection import bucket
from persistence.repositories import paths


def upload_video(inferred_video_name: str):
    inferred_video_dir = paths.video_inferred_path
    blob = bucket.blob(
        'classified_videos/' + inferred_video_name)  # classified videos is the location
    # where we want to store the videos
    # ask maria about it
    blob.upload_from_filename(os.path.join(inferred_video_dir, inferred_video_name), timeout=200)
    blob.make_public()
    print("Firebase URL:", blob.public_url)
