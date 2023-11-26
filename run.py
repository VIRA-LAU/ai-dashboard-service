import sys
import os

sys.path.append(os.path.join(sys.path[0], '.'))
from containers import Services
detection_service = Services.detection_service()

from utils.handle_db.stats_db_handler import Stats_DB_Handler

# stuff to remove from:
    # post_process.py
    # post_processing_handler.py
    # stats_handler.py

# stuff to comment out:
    # detection_service.py

if __name__ == '__main__':
    game_id = 'IMG_3050_Demo'
    stats, video_inferred_path, videos_paths, concatenated, concatenated_with_music, shotsmade = detection_service.run_inference(game_id)
    print(stats, shotsmade)