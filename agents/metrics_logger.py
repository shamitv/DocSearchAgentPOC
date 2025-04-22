# metrics_logger.py
# Contains metric logging functions and related DB setup for LLM metrics
import os
import sqlite3
import time
from datetime import datetime, timezone
import logging

# Setup logger (reuse if already configured)
logger = logging.getLogger("advanced_knowledge_agent.metrics_logger")
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def init_metrics_db(db_path=None):
    """
    Initialize the metrics logging database and return (conn, cursor).
    If db_path is None, uses the INTERMEDIATE_DB_PATH env or 'intermediate_results.db'.
    """
    if db_path is None:
        db_path = os.getenv('INTERMEDIATE_DB_PATH', 'intermediate_results.db')
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS query_llm_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        iteration INTEGER,
        timestamp TEXT,
        model_name TEXT,
        execution_time_seconds REAL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        raw_prompt TEXT,
        raw_content TEXT,
        error_message TEXT
    );
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_llm_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT,
        iteration INTEGER,
        result_index INTEGER,
        timestamp TEXT,
        model_name TEXT,
        execution_time_seconds REAL,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        raw_prompt TEXT,
        raw_content TEXT,
        error_message TEXT
    );
    ''')
    conn.commit()
    return conn, cursor

# SQLite DB initialization
DB_PATH = os.getenv('INTERMEDIATE_DB_PATH', 'intermediate_results.db')
conn, cursor = init_metrics_db(DB_PATH)

def log_llm_metrics(response, start_time, model_name="Unknown", is_query=True, run_id=None, iteration=None, result_index=None, raw_prompt=None):
    """
    Log metrics from an LLM response including tokens and execution time.
    Also stores the metrics in the database.
    """
    elapsed_time = time.time() - start_time
    operation = "Query generation" if is_query else "Result analysis"
    logger.info(f"{operation} execution time: {elapsed_time:.2f} seconds")
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    error_message = None
    raw_content = None
    try:
        if hasattr(response, 'content'):
            raw_content = str(response.content)[:1000]
        if hasattr(response, 'usage'):
            prompt_tokens = getattr(response.usage, 'prompt_tokens', None)
            completion_tokens = getattr(response.usage, 'completion_tokens', None)
            if hasattr(response.usage, 'total_tokens') and response.usage.total_tokens is not None:
                total_tokens = response.usage.total_tokens
            elif prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            logger.info(f"{operation} token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")
        else:
            logger.info(f"{operation} completed, but token usage information not available")
            error_message = "Token usage information not available"
    except Exception as e:
        error_message = str(e)
        logger.warning(f"Could not extract token usage information: {str(e)}")
    if run_id is not None:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            if is_query:
                cursor.execute('''
                INSERT INTO query_llm_metrics 
                (run_id, iteration, timestamp, model_name, execution_time_seconds, 
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', 
                (run_id, iteration, ts, model_name, elapsed_time, 
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message))
            else:
                cursor.execute('''
                INSERT INTO analysis_llm_metrics 
                (run_id, iteration, result_index, timestamp, model_name, execution_time_seconds, 
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (run_id, iteration, result_index, ts, model_name, elapsed_time,
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to log {operation} metrics to database: {str(e)}")
