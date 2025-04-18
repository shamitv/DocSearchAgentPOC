import bz2
import mwxml
from elasticsearch import Elasticsearch
import mwparserfromhell
import logging
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configuration
ES_URL = os.getenv("ES_URL")
DUMP_FILE_PATH = os.getenv("DUMP_FILE_PATH")

# Initialize Elasticsearch client
es = Elasticsearch([ES_URL])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    es.index(index="wikipedia", id=document_id, document=document)

def extract_plain_text(raw_text):
    """Convert raw markup text to plain text."""
    wikicode = mwparserfromhell.parse(raw_text)
    return wikicode.strip_code()

def index_documents_bulk(documents):
    """Index multiple documents into Elasticsearch in bulk."""
    actions = []
    for doc in documents:
        action = {"index": {"_index": "wikipedia", "_id": doc["metadata"]["id"]}}
        actions.append(action)
        actions.append(doc)

    es.bulk(operations=actions)

def index_documents_bulk_async(documents):
    """Index multiple documents into Elasticsearch in bulk asynchronously."""
    actions = []
    for doc in documents:
        action = {"index": {"_index": "wikipedia", "_id": doc["metadata"]["id"]}}
        actions.append(action)
        actions.append(doc)

    # Submit the bulk operation to the executor
    executor.submit(es.bulk, operations=actions)

def process_dump(file_path):
    """Parse and process the Wikipedia dump sequentially with bulk indexing."""
    doc_count = 0
    start_time = time.time()
    bulk_documents = []
    bulk_size = 500  # Number of documents to index in a single bulk request

    logging.info(f"Opening dump file: {file_path}")
    with bz2.open(file_path, "rb") as file:
        dump = mwxml.Dump.from_file(file)
        for page in dump:
            if not page.redirect:
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
                        logging.info(f"Indexing bulk of {len(bulk_documents)} documents...")
                        index_documents_bulk_async(bulk_documents)
                        doc_count += len(bulk_documents)
                        bulk_documents = []

                        elapsed_time = time.time() - start_time
                        avg_time_per_doc = elapsed_time / doc_count
                        logging.info(f"Processed {doc_count} documents so far. Average time per document: {avg_time_per_doc:.6f} seconds.")

        # Index any remaining documents
        if bulk_documents:
            logging.info(f"Indexing final bulk of {len(bulk_documents)} documents...")
            index_documents_bulk_async(bulk_documents)
            doc_count += len(bulk_documents)

    logging.info(f"All tasks completed. Total documents processed: {doc_count}.")

if __name__ == "__main__":
    logging.info("Starting Wikipedia dump processing...")
    process_dump(DUMP_FILE_PATH)
    logging.info("Processing completed.")