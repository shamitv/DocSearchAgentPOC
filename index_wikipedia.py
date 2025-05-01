import sys
import logging # Keep for potential top-level script logging if needed

# Import necessary components from the new package
from wiki_es_indexer.indexer import process_dump, logger # Import logger for script-level messages
from utils import EnvLoader # Keep EnvLoader for getting dump path

# Configure logging for this script specifically (optional, if different from package)
# logger = logging.getLogger(__name__) # Example if you want a separate logger instance

if __name__ == "__main__":
    # Try to get the dump file path from environment variable via utils
    dump_path = EnvLoader.get_dump_file_path()

    # Override with command line argument if provided
    if len(sys.argv) == 2:
        dump_path = sys.argv[1]
    elif len(sys.argv) > 2:
         print(f"Usage: python {sys.argv[0]} [<dump_file_path>]")
         sys.exit(1)


    if not dump_path:
        logger.error("Dump file path not provided. Specify via command line argument or set WIKI_DUMP_PATH environment variable.")
        print(f"Usage: python {sys.argv[0]} [<dump_file_path>]")
        sys.exit(1)

    logger.info("Starting Wikipedia dump processing...")
    try:
        total_docs = process_dump(dump_path)
        logger.info(f"Processing completed. Total documents processed: {total_docs}")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during processing: {e}", exc_info=True)
        sys.exit(1)