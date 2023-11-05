import os
from pathlib import Path

project_title = 'ai-dashboard-service'
def getRootDir() -> Path:
    global project_title
    current_dir = os.path.abspath(__file__)

    while True:
        if os.path.basename(current_dir) == project_title:
            return Path(current_dir)

        current_dir = os.path.dirname(current_dir)

        if current_dir == os.path.dirname(current_dir):
            break

    return Path()
