import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import logging
from utils import EnvLoader, LoggerConfig

# Configure logging
LoggerConfig.configure_logging()

# Load environment variables
ES_HOST, ES_PORT, ES_INDEX = EnvLoader.load_env()

DUMP_DIR = "./data/wikidump/"
DUMP_URL = "https://dumps.wikimedia.org/other/incr/enwiki/"
CUTOFF_DATE = datetime(2025, 4, 1)

# Directory to store dumps (set via environment variable)
DUMP_SUBDIR = os.getenv("DUMP_SUBDIR", "./data/wikidump/2/")
os.makedirs(DUMP_SUBDIR, exist_ok=True)

# Ensure dump directory exists
os.makedirs(DUMP_DIR, exist_ok=True)

# Connect to Elasticsearch
es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}"])

# Create index if it doesn't exist
if not es.indices.exists(index=ES_INDEX):
    es.indices.create(index=ES_INDEX)

# Scrape the incremental dumps page
resp = requests.get(DUMP_URL)
soup = BeautifulSoup(resp.text, "html.parser")

# Find all links to directories
links = soup.find_all('a')
dump_files = []

# Visit each directory for a date and find files
for link in links:
    href = link.get('href', '')
    if href.endswith('/'):
        try:
            date_str = href.strip('/')
            dump_date = datetime.strptime(date_str, "%Y%m%d")
            if dump_date > CUTOFF_DATE:
                # Visit the directory for this date
                date_url = DUMP_URL + href
                date_resp = requests.get(date_url)
                date_soup = BeautifulSoup(date_resp.text, "html.parser")
                date_links = date_soup.find_all('a')
                for date_link in date_links:
                    file_href = date_link.get('href', '')
                    if file_href.endswith('.bz2'):
                        dump_files.append((href + file_href, dump_date))
        except Exception:
            continue


# Ignore 404 errors when checking if a document exists in Elasticsearch.
# This is because a 404 simply means the document is not present, which is expected
# for new dumps that have not been processed yet. This avoids unnecessary exceptions
# and allows the script to proceed with downloading and indexing the dump.
es = es.options(ignore_status=[404])


for filename, dump_date in dump_files:
    file_url = DUMP_URL + filename
    local_path = os.path.join(DUMP_SUBDIR, filename.replace('/', '_'))  # Replace slashes for local storage
    # Check if already processed in ES
    es_id = filename
    try:
        doc = es.get(index=ES_INDEX, id=es_id)
        if doc.get('found') and doc['_source'].get('status') == 'processed':
            logging.info(f"Already processed: {filename}")
            continue
        # Download if not present
        if not os.path.exists(local_path):
            logging.info(f"Downloading {filename}...")
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        else:
            logging.info(f"Already downloaded: {filename}")
        # Index metadata in ES
        es.index(index=ES_INDEX, id=es_id, document={
            'filename': filename,
            'date': dump_date.isoformat(),
            'status': 'downloaded',
            'local_path': local_path,
            'url': file_url
        })
        logging.info(f"Indexed {filename} in ES as 'downloaded'.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {file_url}: {e}")
    except Exception as e:
        logging.error(f"Failed to process {filename}: {e}")

logging.info("Done.")
