
import sqlite3 as lite
from datetime import datetime as dt


def add_session(id_: str):
    with lite.connect('Logs/Observations.db') as db:
        cur = db.cursor()
        cur.execute('INSERT INTO sessions(id) VALUES (?);', (id_,))
        db.commit()

def update_session(id_: str, update_dict: dict):
    with lite.connect('Logs/Observations.db') as db:
        cur = db.cursor()
        for key in update_dict:
            cmd = f'UPDATE sessions SET "{key}"="{update_dict[key]}" WHERE id="{id_}";'
            cur.execute(cmd)
        db.commit()

def get_last_session_id():
    with lite.connect('Logs/Observations.db') as db:
        cur = db.cursor()
        cur.execute('SELECT id FROM sessions ORDER BY id DESC LIMIT 1;')
        id_ = cur.fetchall()[0][0]
    return id_

def get_entry(id_: str, column: str):
    with lite.connect('Logs/Observations.db') as db:
        cur = db.cursor()
        cur.execute(f'SELECT "{column}" FROM sessions WHERE id="{id_}";')
        entry = cur.fetchall()[0][0]
    return entry
