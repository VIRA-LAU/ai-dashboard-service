from fastapi import APIRouter, UploadFile, File, BackgroundTasks

import detect
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
#from application.service.mailing_sender import email_service

router = APIRouter()

detection_service = Services.detection_service()
highlights_service = highlights_service.HighlightsService


@router.post('/Detection_Inference_video')
async def run_inference_on_video(video: UploadFile = File(...)) -> ApiResponse:
    path_input_video = save_video(video=video, destination=str(paths.video_input_path))
    filename = video.filename
    video_inferred_path, videos_paths, concatenated, concatenated_with_music, shots_made = detection_service.run_inference(path_input_video, filename)
    return ApiResponse(success=True, data={
        "video Inferred": video_inferred_path,
        "Highlights": videos_paths,
        "Concatenated": concatenated,
        "Concatenated With Music": concatenated_with_music,
        "Number of Shots Made": shots_made
    })

@router.get('/Run_Inference_On_Existing_Input_Videos')
async def run_inference_on_existing_input_videos() -> ApiResponse:
    #path_input_video, filename = download_video(video_url_input=path)
    stats, video_inferred_path, videos_paths, concatenated, concatenated_with_music, shots_made = detection_service.run_inference('datasets/videos_input/04183.mp4', '04183.mp4')
    return ApiResponse(success=True, data= stats)
@router.get('/Dummy_Endpoint')
async def get_dummy_stats() -> ApiResponse:
    return ApiResponse(success=True, data= {
        'Player_1' : {
            '2_points' : 2,
            '3_points' : 1
        },
        'Team_1_points' : 10,
        'Team_2_points' : 5
    })

@router.get("/Detection_Inference")
async def fetch_run_inference(path: str) -> ApiResponse:
    path_input_video, filename = download_video(video_url_input=path)
    video_inferred_path, videos_paths, concatenated, concatenated_with_music, shots_made = detection_service.run_inference(path_input_video, filename)
    return ApiResponse(success=True, data={
        "video Inferred": video_inferred_path,
        "Highlights": videos_paths,
        "Concatenated": concatenated,
        "Concatenated With Music": concatenated_with_music,
        "Number of Shots Made": shots_made
    })

@router.get("/Pose_Estimation")
async def fetch_run_inference(path: str) -> ApiResponse:
    path_input_video, filename = download_video(video_url_input=path)
    pose_est = True
    video_inferred_path = detection_service.run_inference(path_input_video, filename, pose_est)
    return ApiResponse(success=True, data={
        "video Inferred": video_inferred_path
    })


#####################################################################################################################################################
# To be used at a later stage to communicate with the mobile app
# Current functionality upload video and send mail
#####################################################################################################################################################
class Videos(BaseModel):
    id: str
    path: Union[str, None] = None


# @router.post("/Detection_Inference")
# async def fetch_run_inference_send_mail(video: Videos) -> ApiResponse:
#     path_input_video, filename = download_video(video_url_input=video.path)
#     video_inferred_path, videos_paths, concatenated, concatenated_with_music, shots_made = detection_service.run_inference(path_input_video, filename)
#     url_video = upload_video(id_user=video.id, source_video=concatenated_with_music)
#     # videos_paths, concatenated = highlights_service.split_concat_send(path_input_video=path_input_video,
#     #                                                                   frames_made=frames_made, destination="")
#     mail_success = email_service.send_mail(userId=video.id, link=url_video)
#
#     # print(frames_made)
#     return ApiResponse(success=True, data={
#         "video_inferred": video_inferred_path,
#         "highlights": videos_paths,
#         "video_concatenated": concatenated,
#         "video_concatenated_with_music": concatenated_with_music,
#         "shots_made": shots_made,
#         "mail_success": mail_success
#     })
