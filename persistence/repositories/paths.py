from pathlib import Path
from dev_utils import dirs
from shared.helper.json_helpers import parse_json
root_dir = dirs.getRootDir()
paths_file = str(root_dir / "assets/paths.json")

video_input_path = root_dir / Path(parse_json(paths_file)['videos_input_path'])
video_inferred_path = root_dir / Path(parse_json(paths_file)["videos_inferred_path"])

temporal_videos_input_path = root_dir / Path(parse_json(paths_file)["temporal_videos_input_path"])
temporal_frames = root_dir / Path(parse_json(paths_file)["temporal_frames"])

bbox_coordinates_path = root_dir / Path(parse_json(paths_file)["bbox_coordinates_path"])
keys_path = root_dir / Path(parse_json(paths_file)["keys_path"])
highlights_path = root_dir / Path(parse_json(paths_file)["highlights_path"])
concatenated_path = root_dir / Path(parse_json(paths_file)["concatenated_path"])
song_path = str(Path(root_dir) / Path(parse_json(paths_file)["song_path"]))
concatenated_with_music = root_dir / Path(parse_json(paths_file)["concatenated_with_music"])
labels_path = root_dir / Path(parse_json(paths_file)["label_path"])

logs_path = root_dir / Path(parse_json(paths_file)["logs_path"])
locks_path = root_dir / Path(parse_json(paths_file)["locks_path"])