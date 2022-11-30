from core.video_concat import video_concat
from core.video_splitter import video_splitter
from domain.models.upload_video_fb import upload_video
from persistence.repositories.api_response import ApiResponse


class HighlightsService:

    def split_concat_send(self, path_input_video: str, frames_made: str, destination: str):
        videos_paths = video_splitter(path_to_video=path_input_video, frames_shot_made=frames_made)
        concatenated = video_concat(videos_paths)
        upload_video(destination=destination, source_video=concatenated)
        return videos_paths, concatenated
