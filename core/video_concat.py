from core.video_splitter import video_splitter
from moviepy.editor import VideoFileClip, concatenate_videoclips


def video_concat(path_to_highlights: list)->tuple:
    """
    Concatenate videos of shots made
    :param path_to_highlights:
    :return:
    """
    clips = []
    for path in path_to_highlights:
        clips.append(VideoFileClip(path))
        print("appended")

    final_clip = concatenate_videoclips(clips)
    print(final_clip.duration)
    final_clip.write_videofile("datasets/concatenated/videos.mp4")
    return "datasets/concatenated/videos.mp4", final_clip
