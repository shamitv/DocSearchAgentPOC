import bz2
import mwxml

import mwparserfromhell
import logging
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor

from utils import EnvLoader, ElasticsearchClient
import sys

# Load environment and initialize Elasticsearch client
es_host, es_port, es_dump_index, es_search_index = EnvLoader.load_env()
es = ElasticsearchClient.get_client(es_host, es_port)
INDEX_NAME = es_search_index

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
    es.index(index=INDEX_NAME, id=document_id, document=document)

def extract_plain_text(raw_text):
    """Convert raw markup text to plain text, preserving template parameters."""
    wikicode = mwparserfromhell.parse(raw_text)
    # Preserve templates content by replacing each template with its parameter key-value pairs
    for template in wikicode.filter_templates():
        params = []
        for param in template.params:
            name = str(param.name).strip()
            value = str(param.value).strip()
            params.append(f"{name}: {value}")
        replacement = "\n".join(params)
        wikicode.replace(template, replacement)
    return wikicode.strip_code()

def index_documents_bulk(documents):
    """Index multiple documents into Elasticsearch in bulk."""
    actions = []
    for doc in documents:
        action = {"index": {"_index": INDEX_NAME, "_id": doc["metadata"]["id"]}}
        actions.append(action)
        actions.append(doc)

    es.bulk(operations=actions)

def index_documents_bulk_async(documents):
    """Index multiple documents into Elasticsearch in bulk asynchronously."""
    actions = []
    for doc in documents:
        action = {"index": {"_index": INDEX_NAME, "_id": doc["metadata"]["id"]}}
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
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <dump_file_path>")
        sys.exit(1)
    dump_path = sys.argv[1]
    logging.info("Starting Wikipedia dump processing...")
    process_dump(dump_path)
    logging.info("Processing completed.")