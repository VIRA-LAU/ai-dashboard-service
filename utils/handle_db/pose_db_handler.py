import json
import sqlite3
import persistence.repositories.paths as path

_conn = sqlite3.Connection
_cursor = sqlite3.Cursor

def create_database(game_id: str):
    global _conn, _cursor
    _conn = sqlite3.connect(path.logs_path / f'{game_id}_logs.db')
    _cursor = _conn.cursor()
def create_pose_table():
    global _conn, _cursor
    _cursor.execute('''
        CREATE TABLE IF NOT EXISTS pose_db (
            id INTEGER PRIMARY KEY,
            frame_num INTEGER,
            player_num INTEGER,
            bbox_coords TEXT,
            feet_coords TEXT,
            position TEXT
        )
    ''')

    _conn.commit()
def insert_into_pose_table(
        frame_num: int,
        player_num: int,
        bbox_coords: list,
        feet_coords: list,
        position: str
):
    global _conn, _cursor
    _cursor.execute("INSERT INTO pose_db (frame_num, player_num, bbox_coords,"
                    "feet_coords, position) VALUES (?, ?, ?, ?, ?)",
                    (frame_num, player_num, json.dumps(bbox_coords),
                     json.dumps(feet_coords), position))
    _conn.commit()
def close_db():
    global _conn
    _conn.close()

if __name__ == '__main__':
    create_database('test')
    create_pose_table()
    insert_into_pose_table(
        10, 1, [1,2,3,4], [1,2], '2_points'
    )