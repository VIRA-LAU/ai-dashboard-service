from moviepy.editor import *

from persistence.repositories import paths

before = 3
after = 3


# TODO
# Recheck the timing: before and after might give out of bound values
def video_splitter(path_to_video: str, frames_shot_made: list) -> list:
    """
    Splits video at certain frame numbers
    :param path_to_video:
    :param frames_shot_made:
    :return: paths of the split videos
    """
    to_return: list = []
    video = VideoFileClip(path_to_video)
    fps = video.fps
    print(fps)
    filename = os.path.split(path_to_video)[1]
    filename = filename.split('.', 1)[0]
    os.makedirs(paths.highlights_path, exist_ok=True)
    for counter, frame in enumerate(frames_shot_made):
        filename_with_counter = filename + "_" + str(counter) + ".mp4"
        clip = video.subclip(max(int(frame / fps - before), 0), min(int(frame / fps + after), video.duration))
        clip.write_videofile(os.path.join(paths.highlights_path, filename_with_counter))
        to_return.append(os.path.join(paths.highlights_path, filename_with_counter))
    print(to_return)
    return to_return
