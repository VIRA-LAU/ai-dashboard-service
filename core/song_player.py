# Import everything needed to edit video clips
from moviepy.editor import *
from persistence.repositories import paths


def give_song(video_clip, duration: int, filename: str) -> str:
    """
    Adds music to a video
    :param filename:
    :param video_clip:
    :param duration:
    :return:
    """
    audioclip = AudioFileClip(paths.song_path).subclip(13, 13 + duration)
    videoclip = video_clip.set_audio(audioclip)
    to_return_path = os.path.join(paths.concatenated_with_music, filename)
    videoclip.write_videofile(to_return_path, codec="libx264")
    return to_return_path
