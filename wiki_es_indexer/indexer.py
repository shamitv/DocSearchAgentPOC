import traceback
import os
from datetime import datetime, timezone
import time
import sys
import cProfile
import pstats
import io
from functools import wraps
import concurrent.futures
from itertools import islice

import mwparserfromhell
from indexed_bzip2 import IndexedBzip2File
import mwxml

from wiki_es_indexer.es_handler import ElasticsearchHandler

# Assuming utils.py is in the parent directory relative to this package
try:
    from utils import EnvLoader, LoggerConfig # Remove ElasticsearchClient if not used directly
except ImportError:
    # Handle cases where the script might be run directly or utils is elsewhere
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import EnvLoader, LoggerConfig # Remove ElasticsearchClient


# --- Configuration Loading ---
env_vars = EnvLoader.load_env()
es_host = env_vars.get("ES_HOST")
es_port = env_vars.get("ES_PORT")
es_search_index = env_vars.get("ES_SEARCH_INDEX")
INDEX_NAME = es_search_index
MAX_WORKERS = env_vars.get("INDEXER_MAX_WORKERS", 20)

# --- Logging Setup ---
logger = LoggerConfig.configure_logging()

# --- Elasticsearch Handler Initialization ---
if not INDEX_NAME:
    logger.error("ES_SEARCH_INDEX environment variable not set.")
    sys.exit(1)

# --- Profiling Utilities ---
def profile(output_file=None, sort_by='cumulative', lines_to_display=20):
    """
    Decorator for profiling functions to find performance bottlenecks.
    
    Args:
        output_file: Optional file path to save profiling results
        sort_by: Stat to sort results by (cumulative, time, calls, etc.)
        lines_to_display: Number of results to display
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            
            try:
                result = func(*args, **kwargs)
            finally:
                profiler.disable()
                
                # Print profiling results to console
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
                ps.print_stats(lines_to_display)
                logger.info(f"Profile results for {func.__name__}:\n{s.getvalue()}")
                
                # Save to file if requested
                if output_file:
                    ps.dump_stats(output_file)
                    logger.info(f"Profile data written to {output_file}")
            
            return result
        return wrapper
    return decorator

# Instantiate the handler
es_handler = ElasticsearchHandler(
    host=es_host,
    port=es_port,
    index_name=INDEX_NAME,
    max_workers=MAX_WORKERS
)

# --- Wikipedia Article Parser Class ---
class WikipediaArticleParser:
    """Handles parsing of individual Wikipedia article markup."""

    def _process_templates(self, wikicode):
        """Helper function to process templates within wikicode recursively."""
        templates_to_process = list(wikicode.filter_templates(recursive=False))

        for template in templates_to_process:
            params = []
            for param in template.params:
                value_wikicode = param.value
                if hasattr(value_wikicode, 'filter_templates'):
                    self._process_templates(value_wikicode) # Use self._process_templates

                name = str(param.name).strip()
                value_str = str(value_wikicode).strip()
                value_str = value_str.replace('<br>', '\\n').replace('<br/>', '\\n').replace('<br />', '\\n')
                params.append(f"{name}: {value_str}")

            replacement = "\\n".join(params)
            try:
                wikicode.replace(template, replacement)
            except ValueError:
                # Log this specific instance, maybe with template name?
                logger.warning(f"Could not replace template {template} - ValueError.")
                pass # Continue processing other templates

    def parse_article_text(self, raw_text):
        """Convert raw markup text to plain text, preserving template parameters."""
        if not raw_text:
            return ""
        try:
            wikicode = mwparserfromhell.parse(raw_text)
            self._process_templates(wikicode) # Use self._process_templates
            return wikicode.strip_code()
        except Exception as e:
            logger.error(f"Error processing text starting with '{raw_text[:50]}...': {e}")
            return "" # Return empty string on error


# --- Helper Function for Page Processing ---

# Dictionary to track timing metrics for page processing
page_processing_metrics = {
    "total_count": 0,
    "parsing_time": 0,
    "max_parsing_time": 0,
    "max_parsing_title": "",
    "skipped_count": 0
}

def extract_page(page):
    """Extracts raw data from a page for processing.

    Args:
        page: The mwxml page object.

    Returns:
        A dictionary containing raw page data, or None if the page should be skipped.
    """
    if page.redirect or page.namespace > 0:
        return None

    # Process the latest revision
    revision = next(page, None) # Get the first (often only) revision
    if not revision or not revision.text:
        return None

    # Extract the essential data
    page_data = {
        "page_id": page.id,
        "title": page.title,
        "raw_text": revision.text,
        "revision_id": revision.id,
        "timestamp": revision.timestamp
    }
    
    return page_data

def process_page_for_mp(page_data, parser):
    """Processes a single page from the dump for multiprocessing and directly indexes it.
    This function must be at module level to be properly serialized.

    Args:
        page_data: Dictionary containing the extracted page data.
        parser: An instance of WikipediaArticleParser.

    Returns:
        Boolean indicating whether the document was successfully processed and indexed.
    """
    global page_processing_metrics
    page_processing_metrics["total_count"] += 1
    process_start = time.time()
    
    try:
        if not page_data:
            page_processing_metrics["skipped_count"] += 1
            return False

        title = page_data["title"]
        raw_text = page_data["raw_text"]
        
        # Time the parsing operation specifically
        parsing_start = time.time()
        plain_text = parser.parse_article_text(raw_text)
        parsing_time = time.time() - parsing_start
        
        # Update parsing metrics
        page_processing_metrics["parsing_time"] += parsing_time
        if parsing_time > page_processing_metrics["max_parsing_time"]:
            page_processing_metrics["max_parsing_time"] = parsing_time
            page_processing_metrics["max_parsing_title"] = title
            
        # Log slow parsing operations
        if parsing_time > 1.0:  # Log pages that take more than 1 second to parse
            logger.warning(f"Slow parsing: '{title}' took {parsing_time:.2f}s to parse")

        if not plain_text: # Skip if text extraction failed or resulted in empty
            page_processing_metrics["skipped_count"] += 1
            return False

        metadata = {
            "id": page_data["page_id"],
            "revision_id": page_data["revision_id"],
            "timestamp": str(page_data["timestamp"]) if page_data["timestamp"] else None
        }
        document = {
            "title": title,
            "text": plain_text,
            "metadata": metadata,
            "indexed_on": datetime.now(timezone.utc).isoformat()
        }
        
        # Directly submit to Elasticsearch
        indexing_start = time.time()
        es_handler.index_document(document)
        indexing_time = time.time() - indexing_start
        
        if indexing_time > 0.5:  # Log slow indexing operations
            logger.warning(f"Slow indexing: '{title}' took {indexing_time:.2f}s to index")
        
        # Periodically log parsing metrics (every 1000 pages)
        if page_processing_metrics["total_count"] % 1000 == 0:
            avg_parsing_time = page_processing_metrics["parsing_time"] / page_processing_metrics["total_count"]
            logger.info(f"Parsing metrics - Processed: {page_processing_metrics['total_count']}, "
                      f"Skipped: {page_processing_metrics['skipped_count']}, "
                      f"Avg parsing time: {avg_parsing_time:.4f}s, "
                      f"Max parsing time: {page_processing_metrics['max_parsing_time']:.2f}s "
                      f"('{page_processing_metrics['max_parsing_title']}')")
            
        return True

    except Exception as e:
        title_str = page_data.get('title', 'Unknown Page')
        logger.error(f"Error processing page '{title_str}' (ID: {page_data.get('page_id', 'N/A')}): {e}\n{traceback.format_exc()}")
        return False
    finally:
        # Record total processing time for the page if needed for advanced metrics
        process_time = time.time() - process_start
        # For very slow operations, log them regardless of success
        if process_time > 2.0:  # Log pages that take more than 2 seconds to process completely
            title_str = page_data.get('title', 'Unknown Page')
            logger.warning(f"Slow page processing: '{title_str}' took {process_time:.2f}s to process completely")


# Function removed as we now index documents directly in process_page_for_mp

# --- Serializable functions for multiprocessing ---

def noop_function():
    """Empty function for worker initialization in ProcessPoolExecutor."""
    return None

# --- Core Indexing Functions ---

@profile(output_file="indexer_profile.prof", lines_to_display=30)
def process_dump(file_path):
    """Parse and process the Wikipedia dump sequentially using ElasticsearchHandler."""
    doc_count = 0
    start_time = time.time()
    # Make sure the parser is simple enough to be pickled for multiprocessing
    # If there are issues with pickling, you may need to modify the parser class
    parser = WikipediaArticleParser() # Instantiate the parser
    
    # Timing statistics
    timings = {
        "file_opening": 0,
        "page_processing": 0,
        "total_processing": 0
    }
    
    logger.info(f"Opening dump file: {file_path}")
    try:
        file_open_start = time.time()
        #with bz2.open(file_path, "rb") as file:
        with IndexedBzip2File(file_path, parallelization = 12) as file:
            dump = mwxml.Dump.from_file(file)
            timings["file_opening"] = time.time() - file_open_start

            logger.info(f"File opening completed in {timings['file_opening']:.2f}s")

            # Process pages in parallel with a maximum of N workers
            num_workers = min(20, os.cpu_count()*2 or 1)  # Use all available cores or a max of 20
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # spin up all workers up front
                futures = [executor.submit(noop_function) for _ in range(num_workers)]
                for f in futures: f.result()  # wait for them to start (and immediately do nothing)


                # Process the dump in batches to avoid memory issues
                batch_size = 20  # Process pages in batches
                page_count = 0
                
                while True:
                    # Get the next batch of pages
                    batch = list(islice(dump, batch_size))
                    if not batch:
                        break
                        
                    # If you're submitting tasks to process pages, use the named function:
                    # futures = [executor.submit(process_page, page) for page in batch]
                    
                    # Extract page data first
                    page_data_list = []
                    for page in batch:
                        page_data = extract_page(page)
                        if page_data:
                            page_data_list.append(page_data)
                    
                    # Submit batch for parallel processing
                    page_start = time.time()
                    future_to_page = {executor.submit(process_page_for_mp, page_data, parser): page_data for page_data in page_data_list}
                    
                    # Process completed futures as they finish
                    for future in concurrent.futures.as_completed(future_to_page):
                        document = future.result()
                        page_count += 1
                        

    except FileNotFoundError:
        logger.error(f"Dump file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process dump file {file_path}: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    logger.info("Waiting for all indexing tasks to complete...")
    shutdown_start = time.time()
    es_handler.shutdown(wait=True)
    shutdown_time = time.time() - shutdown_start
    
    # Final timing statistics
    total_time = time.time() - start_time
    timings["total_processing"] = total_time
    
    logger.info(f"All tasks completed. Total documents processed: {doc_count}.")
    logger.info(f"Performance summary:")
    logger.info(f"- Total processing time: {total_time:.2f}s")
    logger.info(f"- File opening: {timings['file_opening']:.2f}s ({timings['file_opening']/total_time*100:.1f}%)")
    logger.info(f"- Page processing: {timings['page_processing']:.2f}s ({timings['page_processing']/total_time*100:.1f}%)")
    logger.info(f"- Final shutdown: {shutdown_time:.2f}s ({shutdown_time/total_time*100:.1f}%)")
    logger.info(f"- Avg time per document: {total_time/doc_count if doc_count else 0:.6f}s")
    
    return doc_count
