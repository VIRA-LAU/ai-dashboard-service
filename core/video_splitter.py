from moviepy.editor import *

from persistence.repositories import paths

before = 3
after = 3


# TODO
# Recheck the timing: before and after might give out of bound values
def video_splitter(path_to_video, frames_shot_made) -> list:
    to_return: list = []
    video = VideoFileClip(path_to_video)
    fps = video.fps
    print(fps)
    for counter, frame in enumerate(frames_shot_made):
        filename = path_to_video.rsplit('/', 1)[1]
        filename = filename.split('.', 1)[0]
        filename = filename + "_" + str(counter) + ".mp4"
        clip = video.subclip(max(int(frame/fps - before), 0), min(int(frame/fps + after), video.duration))
        clip.write_videofile(os.path.join(paths.highlights_path, filename))
        to_return.append(os.path.join(paths.highlights_path, filename))

    return to_return
