from fastapi import APIRouter, UploadFile, File

from containers import Services
from core.song_player import give_song
from core.video_splitter import video_splitter
from domain.models.download_video_fb import download_video
from persistence.repositories.api_response import ApiResponse
from persistence.repositories import paths
from shared.helper.file_handler import save_video
from core.video_concat import video_concat
from domain.models.upload_video_fb import upload_video
from application.service.highlights_handler import highlights_service
from pydantic import BaseModel
from typing import Union
from application.service.mailing_sender import email_service

router = APIRouter()

detection_service = Services.detection_service()
highlights_service = highlights_service.HighlightsService


@router.post('/Detection_Inference_video')
async def run_inference_on_video(video: UploadFile = File(...)) -> ApiResponse:
    videos_paths = ''
    concatenated = ''
    concatenated_with_music = ''
    path_input_video = save_video(video=video, destination=paths.video_input_path)
    video_inferred_path, bbox_coordinated_path, frames_made, shots_made = detection_service.infer_detection(source=path_input_video)
    if shots_made > 0:
        videos_paths = video_splitter(path_to_video=path_input_video, frames_shot_made=frames_made)
        concatenated, video = video_concat(videos_paths)
        print(video.duration)
        concatenated_with_music = give_song(video_clip=video, duration=int(video.duration))
    print(frames_made)
    return ApiResponse(success=True, data={
        "video_inferred": video_inferred_path,
        "highlights": videos_paths,
        "final": concatenated,
        "Concatenated With Music": concatenated_with_music,
        "shots_made": shots_made
    })


@router.get("/Detection_Inference")
async def fetch_run_inference(path: str) -> ApiResponse:
    videos_paths = ''
    concatenated = ''
    concatenated_with_music = ''
    path_input_video = download_video(video_url_input=path)
    video_inferred_path, bbox_coordinated_path, frames_made, shots_made = detection_service.infer_detection(source=path_input_video)
    if shots_made > 0:
        videos_paths = video_splitter(path_to_video=path_input_video, frames_shot_made=frames_made)
        concatenated, video = video_concat(videos_paths)
        print(video.duration)
        concatenated_with_music = give_song(video_clip=video, duration=int(video.duration))
    # upload_video(destination="", source_video=concatenated)
    # videos_paths, concatenated = highlights_service.split_concat_send(path_input_video=path_input_video,
    #                                                                   frames_made=frames_made, destination="")
    print(frames_made)
    return ApiResponse(success=True, data={
        "video_inferred": video_inferred_path,
        "highlights": videos_paths,
        "final": concatenated,
        "Concatenated With Music": concatenated_with_music,
        "shots_made": shots_made
    })


#####################################################################################################################################################
# To be used at a later stage to communicate with the mobile app
# Current functionality upload video and send mail
#####################################################################################################################################################
class Videos(BaseModel):
    id: str
    path: Union[str, None] = None


@router.post("/Detection_Inference")
async def fetch_run_inference_send_mail(video: Videos) -> ApiResponse:
    videos_paths = ''
    concatenated = ''
    concatenated_with_music = ''
    path_input_video = download_video(video_url_input=video.path)
    video_inferred_path, bbox_coordinated_path, frames_made, shots_made = detection_service.infer_detection(source=path_input_video)
    if int(shots_made) > 0:
        videos_paths = video_splitter(path_to_video=path_input_video, frames_shot_made=frames_made)
        concatenated, video_without_music = video_concat(videos_paths)
        concatenated_with_music = give_song(video_clip=video_without_music, duration=int(video_without_music.duration))
    url_video = upload_video(id_user=video.id, source_video=concatenated_with_music)
    # videos_paths, concatenated = highlights_service.split_concat_send(path_input_video=path_input_video,
    #                                                                   frames_made=frames_made, destination="")
    mail_success = email_service.send_mail(userId=video.id, link=url_video)

    # print(frames_made)
    return ApiResponse(success=True, data={
        "video_inferred": video_inferred_path,
        "highlights": videos_paths,
        "video_concatenated": concatenated,
        "video_concatenated_with_music": concatenated_with_music,
        "shots_made": shots_made,
        "mail_success": mail_success
    })
