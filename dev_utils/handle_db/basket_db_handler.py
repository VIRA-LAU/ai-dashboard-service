import json
import sqlite3
import persistence.repositories.paths as path

_conn = sqlite3.Connection
_cursor = sqlite3.Cursor

def create_database(game_id: str):
    global _conn, _cursor
    _conn = sqlite3.connect(path.logs_path / f'{game_id}_logs.db')
    _cursor = _conn.cursor()
def create_basket_table():
    global _conn, _cursor
    _cursor.execute('''
        CREATE TABLE IF NOT EXISTS basket_db (
            id INTEGER PRIMARY KEY,
            frame_num INTEGER,
            shot TEXT,
            bbox_coords TEXT
        )
    ''')

    _conn.commit()
def insert_into_basket_table(
        frame_num: int,
        shot: str,
        bbox_coords: list
):
    global _conn, _cursor
    _cursor.execute("INSERT INTO basket_db (frame_num, shot, bbox_coords) VALUES (?, ?, ?)",
                    (frame_num, shot, json.dumps(bbox_coords)))
    _conn.commit()
def close_db():
    global _conn
    _conn.close()

if __name__ == '__main__':
    create_database('test')
    create_basket_table()
    insert_into_basket_table(
        1, 'netbasket', [1,2,3,4]
    )