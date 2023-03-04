from pathlib import Path

from shared.helper.json_helpers import parse_json

paths_file = "assets/paths.json"

video_input_path = Path(parse_json(paths_file)['videos_input_path'])
video_inferred_path = Path(parse_json(paths_file)["videos_inferred_path"])
bbox_coordinates_path = Path(parse_json(paths_file)["bbox_coordinates_path"])
keys_path = Path(parse_json(paths_file)["keys_path"])
highlights_path = Path(parse_json(paths_file)["highlights_path"])
concatenated_path = Path(parse_json(paths_file)["concatenated_path"])
song_path = parse_json(paths_file)["song_path"]
concatenated_with_music = Path(parse_json(paths_file)["concatenated_with_music"])
labels_path = Path(parse_json(paths_file)["label_path"])
