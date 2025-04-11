import bz2
import mwxml
from elasticsearch import Elasticsearch
import mwparserfromhell
import logging
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Configuration
ES_URL = "http://i3tiny1.local:7020/"
DUMP_FILE_PATH = "./data/wikidump/2/enwiki-20250401-pages-articles-multistream.xml.bz2"

# Initialize Elasticsearch client
es = Elasticsearch([ES_URL])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress Elasticsearch client logs
logging.getLogger("elasticsearch").setLevel(logging.ERROR)

# Suppress urllib3 logs
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Suppress http.client logs
logging.getLogger("http.client").setLevel(logging.WARNING)

# Configure root logger to suppress all other loggers
class NullHandler(logging.Handler):
    def emit(self, record):
        pass

logging.getLogger().addHandler(NullHandler())

# Temporarily set the root logger level to CRITICAL to suppress all logs
logging.getLogger().setLevel(logging.CRITICAL)

# Adjust logging configuration to allow INFO-level logs for custom messages
logging.getLogger().setLevel(logging.INFO)

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

# Function to process a single page
def process_page(page):
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
            index_document(title, plain_text, metadata)

def process_dump(file_path):
    """Parse and process the Wikipedia dump in parallel using a thread-safe queue."""
    doc_count = 0
    start_time = time.time()
    page_queue = Queue(maxsize=500)  # Increased queue size to reduce producer blocking

    def producer():
        logging.info("Producer started.")
        with bz2.open(file_path, "rb") as file:
            dump = mwxml.Dump.from_file(file)
            for page in dump:
                page_queue.put(page)
                logging.info("Page added to queue.")
            # Signal that production is done
            for _ in range(30):  # Number of workers
                page_queue.put(None)
        logging.info("Producer finished.")

    def consumer():
        nonlocal doc_count
        logging.info("Consumer started.")
        while True:
            page = page_queue.get()
            if page is None:
                logging.info("Consumer received termination signal.")
                break
            logging.info(f"Consumer processing page: {page.title}")
            process_page(page)
            doc_count += 1

            if doc_count % 10000 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_doc = elapsed_time / doc_count
                logging.info(f"Processed {doc_count} documents so far. Average time per document: {avg_time_per_doc:.6f} seconds.")
        logging.info("Consumer finished.")

    # Start producer and consumer threads
    with ThreadPoolExecutor(max_workers=31) as executor:  # 30 consumers + 1 producer
        executor.submit(producer)
        futures = [executor.submit(consumer) for _ in range(30)]

        # Wait for all consumers to complete
        for future in futures:
            future.result()

    logging.info("All tasks completed.")

if __name__ == "__main__":
    print("Starting Wikipedia dump processing...")
    process_dump(DUMP_FILE_PATH)
    print("Processing completed.")