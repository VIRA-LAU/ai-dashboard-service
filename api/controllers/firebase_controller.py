import firebase_admin
from fastapi import APIRouter, Response
from shared.helper.json_helpers import parse_json
from persistence.repositories.paths import paths
from firebase_admin import credentials, initialize_app, storage
from google.cloud import storage
from google.oauth2 import service_account

router = APIRouter()

cred = credentials.Certificate("../keys.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'fyp-interface.appspot.com'})

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    credentials = service_account.Credentials.from_service_account_file("../keys.json")
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
upload_blob(firebase_admin.storage.bucket().name, 'sample_image_file.jpg', 'images/beatiful_picture.jpg')
# connecting to firebase

# bucket = storage.bucket() # storage bucket
# blob = bucket.blob(file_path) Do these 2 lines if we are uploading

credentials = service_account.Credentials.from_service_account_file("../keys.json")
storage.Client(credentials=credentials).bucket(firebase_admin.storage.bucket().name).blob(
    'text_docs/sample_text_file.txt').download_to_filename('downloaded_file.txt')

@router.get('/video')
async def get_video(url_path: str):
    storage.child(url_path).download(paths["video_input_path"])


@router.post("/video")
async def post_videos(path_on_cloud: str):
    # TODO
    # correct the local path of the infered video
    storage.child(path_on_cloud).put(paths["video_inferred_path"])
