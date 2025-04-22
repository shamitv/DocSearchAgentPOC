import os
import sqlite3

def init_main_db(db_path=None):
    """
    Initialize the main application database and return (conn, cursor).
    If db_path is None, uses the INTERMEDIATE_DB_PATH env or 'intermediate_results.db'.
    """
    if db_path is None:
        db_path = os.getenv('INTERMEDIATE_DB_PATH', 'intermediate_results.db')
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        question TEXT,
        start_time TEXT
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS query_generations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        iteration INTEGER,
        timestamp TEXT,
        prompt TEXT,
        response TEXT
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS search_queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        iteration INTEGER,
        query TEXT,
        timestamp TEXT
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS search_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        iteration INTEGER,
        query TEXT,
        results TEXT,
        timestamp TEXT
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        iteration INTEGER,
        timestamp TEXT,
        prompt TEXT
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        iteration INTEGER,
        result_index INTEGER,
        timestamp TEXT,
        response TEXT
    );
    ''')
    conn.commit()
    return conn, cursor
