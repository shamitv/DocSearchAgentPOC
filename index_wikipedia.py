import sys
import logging # Keep for potential top-level script logging if needed
import cProfile
import pstats
import io
import argparse

# Import necessary components from the new package
from wiki_es_indexer.indexer import process_dump, logger # Import logger for script-level messages
from utils import EnvLoader # Keep EnvLoader for getting dump path

# Configure logging for this script specifically (optional, if different from package)
# logger = logging.getLogger(__name__) # Example if you want a separate logger instance

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process Wikipedia dump and index articles")
    parser.add_argument("dump_path", nargs="?", default=None, help="Path to Wikipedia dump file (bz2 format)")
    parser.add_argument("--profile", action="store_true", help="Enable detailed profiling")
    parser.add_argument("--profile-output", default="wiki_indexer_profile.prof", 
                        help="Output file for profiling data (for use with snakeviz or similar tool)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Try to get the dump file path from environment variable or command line
    dump_path = args.dump_path or EnvLoader.get_dump_file_path()

    if not dump_path:
        logger.error("Dump file path not provided. Specify via command line argument or set WIKI_DUMP_PATH environment variable.")
        print(f"Usage: python {sys.argv[0]} [<dump_file_path>]")
        sys.exit(1)

    logger.info("Starting Wikipedia dump processing...")
    
    try:
        if args.profile:
            logger.info(f"Running with profiling enabled, output will be saved to {args.profile_output}")
            profiler = cProfile.Profile()
            profiler.enable()
            
            total_docs = process_dump(dump_path)
            
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # Show top 30 time-consuming functions
            logger.info(f"Profile results:\n{s.getvalue()}")
            
            # Save profiling data for external analysis
            ps.dump_stats(args.profile_output)
            logger.info(f"Detailed profiling data saved to {args.profile_output}")
            logger.info("You can analyze this file with tools like snakeviz (pip install snakeviz)")
            logger.info(f"Run: snakeviz {args.profile_output}")
        else:
            total_docs = process_dump(dump_path)
            
        logger.info(f"Processing completed. Total documents processed: {total_docs}")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during processing: {e}", exc_info=True)
        sys.exit(1)