# Import everything needed to edit video clips
from moviepy.editor import *
from persistence.repositories import paths


def give_song(video_clip, duration: int)->str:
    """
    Adds music to a video
    :param video_clip:
    :param duration:
    :return:
    """
    audioclip = AudioFileClip(paths.song_path).subclip(13, 13 + duration)
    videoclip = video_clip.set_audio(audioclip)
    videoclip.write_videofile("datasets/concatenated_with_music/videos.mp4")
    return "datasets/concatenated_with_music/videos.mp4"
