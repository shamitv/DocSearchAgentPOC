import time
from datetime import datetime, timezone

def log_run_start(cursor, conn, generate_run_id, question: str) -> str:
    run_id = generate_run_id()
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO runs (id, question, start_time) VALUES (?, ?, ?)',
        (run_id, question, ts)
    )
    conn.commit()
    return run_id

def log_query_generation(cursor, conn, run_id: int, iteration: int, prompt: str, response: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO query_generations (run_id, iteration, timestamp, prompt, response) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, ts, prompt, response)
    )
    conn.commit()

def log_search_query(cursor, conn, run_id: int, iteration: int, query: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO search_queries (run_id, iteration, query, timestamp) VALUES (?, ?, ?, ?)',
        (run_id, iteration, query, ts)
    )
    conn.commit()

def log_search_result(cursor, conn, run_id: int, iteration: int, query: str, results: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO search_results (run_id, iteration, query, results, timestamp) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, query, results, ts)
    )
    conn.commit()

def log_analysis_prompt(cursor, conn, run_id: int, iteration: int, result_index: int, prompt: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO analysis_prompts (run_id, iteration, result_index, timestamp, prompt) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, result_index, ts, prompt)
    )
    conn.commit()

def log_analysis_result(cursor, conn, run_id: int, iteration: int, result_index: int, response: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO analysis_results (run_id, iteration, result_index, timestamp, response) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, result_index, ts, response)
    )
    conn.commit()
