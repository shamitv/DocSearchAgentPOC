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
import xml.etree.ElementTree as ET  # For parsing XML strings
import threading

import mwparserfromhell
from indexed_bzip2 import IndexedBzip2File
import mwxml

from wiki_es_indexer.es_handler import ElasticsearchHandler
from .bounded_process_pool_executor import BoundedProcessPoolExecutor

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
                # Convert HTML line breaks to actual newlines
                value_str = value_str.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
                params.append(f"{name}: {value_str}")

            # Join parameters using actual newline character
            replacement = "\n".join(params)
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

# Helper to parse a <page> XML string and index via existing logic
def process_page_str(page_xml, parser):
    """
    Parse a single <page> XML string and index it using process_page_for_mp.
    """
    try:
        root = ET.fromstring(page_xml)
        # Skip redirects and non-article namespaces
        if root.find('redirect') is not None:
            return False
        ns = root.find('ns')
        if ns is not None and int(ns.text or 0) > 0:
            return False
        # Extract fields
        page_id = int(root.find('id').text)
        title = root.find('title').text or ''
        rev = root.find('revision')
        if rev is None or rev.find('text') is None:
            return False
        raw_text = rev.find('text').text or ''
        page_data = {
            'page_id': page_id,
            'title': title,
            'raw_text': raw_text,
            'revision_id': int(rev.find('id').text) if rev.find('id') is not None else None,
            'timestamp': rev.find('timestamp').text if rev.find('timestamp') is not None else None
        }
        return process_page_for_mp(page_data, parser)
    except Exception as e:
        logger.error(f"XML parse error during process_page_str: {e}")
        return False

# --- Core Indexing Functions ---

#@profile(output_file="indexer_profile.prof", lines_to_display=30)
def process_dump(file_path):
    """"""
    return process_dump_stream(file_path)


# Alternative stream-based dump processor: line-by-line collection of <page> elements
#@profile(output_file="indexer_profile.prof", lines_to_display=30)
def process_dump_stream(file_path):
    """
    Stream process a Wikipedia XML dump: collect each <page>...</page> block and send to a thread pool.
    """
    parser = WikipediaArticleParser()
    num_workers = min(int(MAX_WORKERS), os.cpu_count() * 2 or 1)
    max_queue_size = num_workers * 2
    futures = []
    try:
        with IndexedBzip2File(file_path, parallelization=12) as bz2_file:
            text_file = io.TextIOWrapper(bz2_file, encoding='utf-8')
            with BoundedProcessPoolExecutor(max_workers=num_workers, max_queue_size=max_queue_size) as executor:
                buffer = []
                collecting = False
                for line in text_file:
                    if '<page>' in line:
                        collecting = True
                        buffer = [line]
                    elif '</page>' in line and collecting:
                        buffer.append(line)
                        page_xml = ''.join(buffer)
                        futures.append(executor.submit(process_page_str, page_xml, parser))
                        collecting = False
                    elif collecting:
                        buffer.append(line)
                # Wait for all tasks
                for fut in concurrent.futures.as_completed(futures):
                    _ = fut.result()
        # Ensure remaining tasks flushed and shutdown handler
        es_handler.shutdown(wait=True)
        return len(futures)
    except FileNotFoundError:
        logger.error(f"Dump file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Streaming processing failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    # CLI entry: streaming is default; use --batch for batch+multiprocess
    import sys
    args = sys.argv[1:]
    use_stream = True
    # switch to batch mode if requested
    if args and args[0] == '--batch':
        use_stream = False
        args = args[1:]
    # determine dump file path: CLI arg or environment
    from utils import EnvLoader
    if args:
        file_path = args[0]
    else:
        file_path = EnvLoader.get_dump_file_path()
    # execute
    if use_stream:
        count = process_dump_stream(file_path)
    else:
        count = process_dump(file_path)
    logger.info(f"Indexed {count} pages from {file_path}")
