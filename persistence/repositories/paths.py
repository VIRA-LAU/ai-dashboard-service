from pathlib import Path

from shared.helper.json_helpers import parse_json

video_input_path = Path(parse_json("assets/paths.json")['videos_input_path'])
video_inferred_path = Path(parse_json("assets/paths.json")["videos_inferred_path"])
bbox_coordinates_path = Path(parse_json("assets/paths.json")["bbox_coordinates_path"])
keys_path = Path(parse_json("assets/paths.json")["keys_path"])
highlights_path = Path(parse_json("assets/paths.json")["highlights_path"])
concatenated_path = Path(parse_json("assets/paths.json")["concatenated_path"])
song_path = parse_json("assets/paths.json")["song_path"]
concatenated_with_music = Path(parse_json("assets/paths.json")["concatenated_with_music"])
