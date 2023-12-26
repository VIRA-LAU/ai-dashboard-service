import json
import sqlite3
import persistence.repositories.paths as path

_conn = sqlite3.Connection
_cursor = sqlite3.Cursor

def create_database(game_id: str):
    global _conn, _cursor
    _conn = sqlite3.connect(path.logs_path / f'{game_id}_logs.db')
    _cursor = _conn.cursor()


def create_person_table():
    global _conn, _cursor
    _cursor.execute('''
        CREATE TABLE IF NOT EXISTS person_db (
            id INTEGER PRIMARY KEY,
            frame_num INTEGER,
            player_num INTEGER,
            bbox_coords TEXT,
            feet_coords TEXT,
            position TEXT,
            action TEXT,
            player_with_basketball TEXT
        )
    ''')

    _conn.commit()


def insert_into_person_table(
        frame_num: int,
        player_num: int,
        bbox_coords: list,
        feet_coords: list,
        position: str,
        action: str,
        player_with_basketball: str
):
    global _conn, _cursor
    _cursor.execute("INSERT INTO person_db (frame_num, player_num, bbox_coords,"
                    "feet_coords, position, action, player_with_basketball) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (frame_num, player_num, json.dumps(bbox_coords),
                     json.dumps(feet_coords), position, action, player_with_basketball))
    _conn.commit()

def update_person_pwb_table(
        frame_num: int,
        player_num: int,
        player_with_basketball: str
):
    global _conn, _cursor
    _cursor.execute("UPDATE person_db SET player_with_basketball = (?) WHERE frame_num = (?) AND player_num = (?)",
                    (player_with_basketball, frame_num, player_num))
    _conn.commit()


def close_db():
    global _conn
    _conn.close()