import time
import sys
import os
# Ensure project root is in PYTHONPATH for utils import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import logging
from indexed_bzip2 import IndexedBzip2File
from utils import LoggerConfig, EnvLoader

# Initialize logger via utils
logger = LoggerConfig.configure_logging()

def measure_dump_read_performance(dump_file_path: str):
    """
    Measures the performance of reading lines from an indexed Bzip2 XML dump file.

    Args:
        dump_file_path: The path to the Wikipedia XML dump file (.bz2).
    """
    if not os.path.exists(dump_file_path):
        logger.error(f"Dump file not found: {dump_file_path}")
        return

    logger.info(f"Starting performance test for reading: {dump_file_path}")
    start_time = time.time()
    line_count = 0
    log_interval = 200000*100 # Log every 20 million lines

    try:
        # Use IndexedBzip2File for potentially faster seeking and parallel decompression
        # Set parallelization based on CPU count or a reasonable default
        num_threads = os.cpu_count() or 4 # Default to 4 if cpu_count is None
        logger.info(f"Using {num_threads} threads for decompression.")

        with IndexedBzip2File(dump_file_path, parallelization=num_threads) as file:
            # Iterate line by line
            for line in file:
                line_count += 1
                if line_count % log_interval == 0:
                    elapsed_time = time.time() - start_time
                    logger.info(f"Read {line_count:,} lines in {elapsed_time:.2f} seconds.")
                # We don't need to process the line content for this test
                pass # Explicitly do nothing with the line

    except FileNotFoundError:
        logger.error(f"Dump file not found during processing: {dump_file_path}")
        return
    except Exception as e:
        logger.error(f"An error occurred while reading the dump file: {e}", exc_info=True)
        return
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Finished reading dump file.")
        logger.info(f"Total lines read: {line_count:,}")
        logger.info(f"Total time taken: {total_time:.2f} seconds")
        if line_count > 0:
             logger.info(f"Average lines per second: {line_count / total_time:,.2f}")

if __name__ == "__main__":
    # Determine dump file path from argument or environment
    if len(sys.argv) == 2:
        wiki_dump_path = sys.argv[1]
    else:
        try:
            wiki_dump_path = EnvLoader.get_dump_file_path()
        except EnvironmentError as e:
            logger.error(str(e))
            print("Usage: python test_read_dump_perf.py <path_to_wikipedia_dump.xml.bz2>")
            sys.exit(1)
    measure_dump_read_performance(wiki_dump_path)
