from fastapi import APIRouter, UploadFile, File

from containers import Services
from core.video_splitter import video_splitter
from domain.models.download_video_fb import download_video
from persistence.repositories.api_response import ApiResponse
from persistence.repositories.paths import paths
from shared.helper.file_handler import save_video
from core.video_concat import video_concat
from domain.models.upload_video_fb import upload_video

router = APIRouter()

detection_service = Services.detection_service()


@router.post('/Detection_Inference')
async def run_inference(video: UploadFile = File(...)) -> ApiResponse:
    path_to_input_video = save_video(video=video,
                                     destination=paths["video_input_path"])
    video_inferred_path, bbox_coordinated_path, frames_made = detection_service.infer_detection(
        source=path_to_input_video)

    return ApiResponse(success=True,
                       data={'Inferred Video': video_inferred_path,
                             'BBox Coordinates': bbox_coordinated_path})


@router.get("/Detection_Inference")
async def fetch_run_inference(path: str) -> ApiResponse:
    path_input_video = download_video(video_url_input=path)
    video_inferred_path, bbox_coordinated_path, frames_made = detection_service.infer_detection(source=path_input_video)
    videos_paths = video_splitter(path_to_video=path_input_video, frames_shot_made=frames_made)
    concatenated = video_concat(videos_paths)
    upload_video(destination="", source_video=concatenated)

    print(frames_made)
    return ApiResponse(success=True, data={
        "video_inferred": video_inferred_path,
        "highlights": videos_paths,
        "final": concatenated

    })
