from shared.helper.json_helpers import parse_json

video_input_path = parse_json("assets/paths.json")['videos_input_path']
video_inferred_path = parse_json("assets/paths.json")["videos_inferred_path"]
bbox_coordinates_path = parse_json("assets/paths.json")["bbox_coordinates_path"]

paths = {
    "video_input_path": video_input_path,
    "video_inferred_path": video_inferred_path,
    "bbox_coordinates_path": bbox_coordinates_path

}
