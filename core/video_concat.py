import os

from core.video_splitter import video_splitter
from moviepy.editor import VideoFileClip, concatenate_videoclips

from persistence.repositories import paths


def video_concat(path_to_highlights: list, filename: str) -> tuple:
    """
    Concatenate videos of shots made
    :param filename:
    :param path_to_highlights:
    :return:
    """
    clips = []
    for path in path_to_highlights:
        clips.append(VideoFileClip(path))
        print("appended")

    final_clip = concatenate_videoclips(clips)
    print(final_clip.duration)
    os.makedirs(paths.concatenated_path, exist_ok=True)
    path_to_return = os.path.join(paths.concatenated_path, filename)
    final_clip.write_videofile(path_to_return, codec="libx264")
    return path_to_return, final_clip
