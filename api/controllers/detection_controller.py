from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from persistence.repositories.api_response import ApiResponse
from application.service.detection.detection_service import DetectionService
from shared.helper.file_handler import save_video
from shared.helper.json_helpers import parse_json

router = APIRouter()


@router.post('/Detection_Inference')
async def get_characters_bounding_box(video: UploadFile = File(...)) -> ApiResponse:
    path_to_input_video = save_video(video=video,
                                     destination=str(parse_json("../../assets/paths.json")["videos_input_path"]))
    video_inferred_path = DetectionService.infer_detection(source=path_to_input_video)

    return ApiResponse(success=True,
                       data=video_inferred_path)
