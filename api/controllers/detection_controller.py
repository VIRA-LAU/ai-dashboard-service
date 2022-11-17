from fastapi import APIRouter, UploadFile, File
from shared.helper.json_helpers import parse_json
from containers import Services
from persistence.repositories.api_response import ApiResponse
from persistence.repositories.paths import paths
from shared.helper.file_handler import save_video

router = APIRouter()

detection_service = Services.detection_service()


@router.post('/Detection_Inference')
async def run_inference(video: UploadFile = File(...)) -> ApiResponse:
    path_to_input_video = save_video(video=video,
                                     destination=paths["video_input_path"])
    video_inferred_path, bbox_coordinated_path = detection_service.infer_detection(source=path_to_input_video)

    return ApiResponse(success=True,
                       data={'Inferred Video': video_inferred_path,
                             'BBox Coordinates': bbox_coordinated_path})
