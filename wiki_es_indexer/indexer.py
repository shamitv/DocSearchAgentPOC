import bz2
import traceback
import logging
import mwxml
import mwparserfromhell
from datetime import datetime, timezone
import time
import sys

# Import the new handler
from .es_handler import ElasticsearchHandler

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
logger = LoggerConfig.configure_logging(log_file='wiki_es_indexer.log')

# --- Elasticsearch Handler Initialization ---
if not INDEX_NAME:
    logger.error("ES_SEARCH_INDEX environment variable not set.")
    sys.exit(1)

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


# --- Core Indexing Functions ---

def process_dump(file_path):
    """Parse and process the Wikipedia dump sequentially using ElasticsearchHandler."""
    doc_count = 0
    start_time = time.time()
    bulk_documents = []
    bulk_size = 500  # Consider making this configurable
    parser = WikipediaArticleParser() # Instantiate the parser

    logger.info(f"Opening dump file: {file_path}")
    try:
        with bz2.open(file_path, "rb") as file:
            dump = mwxml.Dump.from_file(file)
            for page in dump:
                try:
                    if page.redirect or page.namespace > 0:
                        continue

                    # Process the latest revision
                    revision = next(page, None) # Get the first (often only) revision
                    if not revision or not revision.text:
                        continue

                    title = page.title
                    raw_text = revision.text
                    # Use the parser instance to extract text
                    plain_text = parser.parse_article_text(raw_text)

                    if not plain_text: # Skip if text extraction failed or resulted in empty
                        continue

                    metadata = {
                        "id": page.id,
                        "revision_id": revision.id,
                        "timestamp": str(revision.timestamp) if revision.timestamp else None
                    }
                    document = {
                        "title": title,
                        "text": plain_text,
                        "metadata": metadata,
                        "indexed_on": datetime.now(timezone.utc).isoformat()
                    }
                    bulk_documents.append(document)

                    if len(bulk_documents) >= bulk_size:
                        logger.info(f"Queueing bulk index for {len(bulk_documents)} documents via handler...")
                        # Use the handler to submit the bulk index task
                        es_handler.submit_bulk_index(list(bulk_documents)) # Pass a copy
                        doc_count += len(bulk_documents)
                        bulk_documents = [] # Clear the list

                        # Log progress periodically
                        if doc_count % (bulk_size * 10) == 0: # Log every 10 bulks
                             elapsed_time = time.time() - start_time
                             if doc_count > 0:
                                 avg_time_per_doc = elapsed_time / doc_count
                                 logger.info(f"Processed {doc_count} documents. Avg time/doc: {avg_time_per_doc:.6f}s.")
                             else:
                                 logger.info(f"Processed {doc_count} documents.")


                except Exception as e:
                    title_str = getattr(page, 'title', 'Unknown Page')
                    logger.error(f"Error processing page '{title_str}' (ID: {getattr(page, 'id', 'N/A')}): {e}\n{traceback.format_exc()}")


            # Index any remaining documents
            if bulk_documents:
                logger.info(f"Queueing final bulk index for {len(bulk_documents)} documents via handler...")
                # Use the handler for the final bulk
                es_handler.submit_bulk_index(list(bulk_documents))
                doc_count += len(bulk_documents)

    except FileNotFoundError:
        logger.error(f"Dump file not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to process dump file {file_path}: {e}\n{traceback.format_exc()}")
        sys.exit(1)

    logger.info("Waiting for all indexing tasks to complete...")
    # Use the handler to shut down the executor
    es_handler.shutdown(wait=True)
    logger.info(f"All tasks completed. Total documents processed: {doc_count}.")
    return doc_count
