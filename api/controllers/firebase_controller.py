from fastapi import APIRouter, Response
from shared.helper.json_helpers import parse_json
from persistence.repositories.paths import paths

router = APIRouter()
config_key = parse_json("./keys.json")
firebase = pyrebase.initialize_app(config_key)
storage = firebase.storage()


@router.get('/video')
async def(url_path: str):
    storage.child(url_path).download(paths["video_input_path"])


@router.post("/video")
async def(path_on_cloud: str):
    # TODO
    # correct the local path of the infered video
    storage.child(path_on_cloud).put(paths["video_inferred_path"])
