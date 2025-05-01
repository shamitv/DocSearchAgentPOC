import bz2
import traceback
import logging  # Keep the import for potential direct use if needed

import mwxml

import mwparserfromhell
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor

from utils import EnvLoader, ElasticsearchClient, LoggerConfig # Import LoggerConfig
import sys

# Load environment variables
env_vars = EnvLoader.load_env()
es_host = env_vars.get("ES_HOST")
es_port = env_vars.get("ES_PORT")
es_dump_index = env_vars.get("ES_DUMP_INDEX")
es_search_index = env_vars.get("ES_SEARCH_INDEX")

# Define INDEX_NAME using the loaded environment variable
INDEX_NAME = es_search_index

# Configure logging using LoggerConfig from utils
logger = LoggerConfig.configure_logging()

# Initialize Elasticsearch client
try:
    es = ElasticsearchClient.get_client(es_host, es_port)
except Exception as e:
    logger.error(f"Failed to initialize Elasticsearch client: {str(e)}") # Use logger
    sys.exit(1) # Exit if ES connection fails

# Configure logging - REMOVED, now handled by LoggerConfig
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create a ThreadPoolExecutor for asynchronous bulk indexing
executor = ThreadPoolExecutor(max_workers=20)

def index_document(title, text, metadata):
    """Index a document into Elasticsearch."""
    document_id = metadata["id"]  # Use Wikipedia ID as the document ID
    document = {
        "title": title,
        "text": text,
        "metadata": metadata,
        "indexed_on": datetime.now(timezone.utc).isoformat()
    }
    # Use the globally defined es and INDEX_NAME
    es.index(index=INDEX_NAME, id=document_id, document=document)

def _process_templates(wikicode):
    """Helper function to process templates within wikicode recursively."""
    # Process templates depth-first: find templates, recursively process their params, then replace.
    templates_to_process = list(wikicode.filter_templates(recursive=False)) # Get top-level templates first

    for template in templates_to_process:
        params = []
        for param in template.params:
            # Recursively process templates within the parameter value first
            value_wikicode = param.value # Get the Wikicode object for the value
            if hasattr(value_wikicode, 'filter_templates'): # Check if value contains more wikicode
                _process_templates(value_wikicode) # Process templates within the value

            name = str(param.name).strip()
            # Convert the processed parameter value to string
            value_str = str(value_wikicode).strip()
            # Manually replace <br> tags (and variants) with actual newline characters
            value_str = value_str.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
            params.append(f"{name}: {value_str}")

        # Join parameters with actual newline characters
        replacement = "\n".join(params)
        try:
            # Replace the original template instance
            wikicode.replace(template, replacement)
        except ValueError:
            # This might happen if the template structure changed unexpectedly or was already replaced
            logger.warning(f"Could not replace template {template} - ValueError. Might be due to complex nesting or prior modification.") # Use logger
            pass

def extract_plain_text(raw_text):
    """Convert raw markup text to plain text, preserving template parameters."""
    wikicode = mwparserfromhell.parse(raw_text)
    # Preserve templates content by replacing each template with its parameter key-value pairs
    try:
        _process_templates(wikicode)
    except Exception as e:
        # Catch potential errors during the parsing/replacement itself
        logger.error(f"Error processing templates in text starting with '{raw_text[:50]}...': {e}") # Use logger
    return wikicode.strip_code()

def index_documents_bulk(documents):
    """Index multiple documents into Elasticsearch in bulk."""
    actions = []
    for doc in documents:
        # Use the globally defined INDEX_NAME
        action = {"index": {"_index": INDEX_NAME, "_id": doc["metadata"]["id"]}}
        actions.append(action)
        actions.append(doc)
    # Use the globally defined es
    es.bulk(operations=actions)

def index_documents_bulk_async(documents):
    """Index multiple documents into Elasticsearch in bulk asynchronously."""
    actions = []
    for doc in documents:
        # Use the globally defined INDEX_NAME
        action = {"index": {"_index": INDEX_NAME, "_id": doc["metadata"]["id"]}}
        actions.append(action)
        actions.append(doc)

    # Submit the bulk operation to the executor using the global es
    executor.submit(es.bulk, operations=actions)

def process_dump(file_path):
    """Parse and process the Wikipedia dump sequentially with bulk indexing."""
    doc_count = 0
    start_time = time.time()
    bulk_documents = []
    bulk_size = 500  # Number of documents to index in a single bulk request

    logger.info(f"Opening dump file: {file_path}") # Use logger
    with bz2.open(file_path, "rb") as file:
        dump = mwxml.Dump.from_file(file)
        for page in dump:
            try:
                if not page.redirect:
                    # Skip pages with namespace > 0
                    if page.namespace > 0:
                        continue
                    for revision in page:
                        title = page.title
                        raw_text = revision.text or ""
                        plain_text = extract_plain_text(raw_text)
                        metadata = {
                            "id": page.id,
                            "revision_id": revision.id,
                            "timestamp": str(revision.timestamp)  # Convert to string
                        }
                        document = {
                            "title": title,
                            "text": plain_text,
                            "metadata": metadata,
                            "indexed_on": datetime.now(timezone.utc).isoformat()
                        }
                        bulk_documents.append(document)

                        if len(bulk_documents) >= bulk_size:
                            logger.info(f"Indexing bulk of {len(bulk_documents)} documents...") # Use logger
                            index_documents_bulk_async(bulk_documents)
                            doc_count += len(bulk_documents)
                            bulk_documents = []

                            elapsed_time = time.time() - start_time
                            avg_time_per_doc = elapsed_time / doc_count
                            logger.info(f"Processed {doc_count} documents so far. Average time per document: {avg_time_per_doc:.6f} seconds.") # Use logger
            except Exception as e:
                logger.error(f"Error processing {title}: {e}") # Use logger

        # Index any remaining documents
        if bulk_documents:
            logger.info(f"Indexing final bulk of {len(bulk_documents)} documents...") # Use logger
            index_documents_bulk_async(bulk_documents)
            doc_count += len(bulk_documents)

    logger.info(f"All tasks completed. Total documents processed: {doc_count}.") # Use logger

if __name__ == "__main__":
    if len(sys.argv) != 2:
        #try to get the dump file path from environment variable via utils
        dump_path = EnvLoader.get_dump_file_path()
        if not dump_path:
            logger.error("Dump file path not provided. Please specify the dump file path as a command line argument or set it in the environment.") # Use logger
            print(f"Usage: python {sys.argv[0]} <dump_file_path>")
            sys.exit(1)
    else:
        dump_path = sys.argv[1]
    logger.info("Starting Wikipedia dump processing...") # Use logger
    process_dump(dump_path)
    logger.info("Processing completed.") # Use logger