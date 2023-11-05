import persistence.repositories.paths as paths
from utils.paths import dirs
import os


class LockService:
    def __init__(self):
        self.root_dir = dirs.getRootDir()
        self.lock_dir = paths.locks_path
        os.makedirs(self.lock_dir, exist_ok=True)

    def createLockFile(self, game_id: str):
        file_ext = f'{game_id}.processing'
        with open(self.lock_dir / file_ext, 'w'):
            pass

    def lockFileExists(self, game_id: str) -> bool:
        file_ext = f'{game_id}.processing'
        return os.path.exists(self.lock_dir / file_ext)

    def deleteLockFile(self, game_id: str):
        if self.lockFileExists(game_id):
            file_ext = f'{game_id}.processing'
            os.remove(self.lock_dir / file_ext)


if __name__ == '__main__':
    lock_service = LockService()
    mystr = 'vid.mp4'
    print(os.path.join('1', '2', '3'))
