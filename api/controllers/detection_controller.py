from fastapi import APIRouter, UploadFile, File

from containers import Services
from persistence.repositories.api_response import ApiResponse
from persistence.repositories.paths import paths
from shared.helper.file_handler import save_video

router = APIRouter()

detection_service = Services.character_extractor_service()


@router.post('/Detection_Inference')
async def get_characters_bounding_box(video: UploadFile = File(...)) -> ApiResponse:
    path_to_input_video = save_video(video=video,
                                     destination=paths["video_input_path"])
    video_inferred_path = detection_service.infer_detection(source=path_to_input_video)

    return ApiResponse(success=True,
                       data=video_inferred_path)
