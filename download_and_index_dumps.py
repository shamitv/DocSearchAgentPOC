import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_and_index_dumps.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Debugging: Log the absolute path of the .env file
env_file_path = os.path.abspath(".env")
logging.info(f"Absolute path of .env file: {env_file_path}")

# Debugging: Check if environment variables are loaded
current_dir = os.getcwd()
env_file_exists = os.path.exists(".env")
env_file_status = "found" if env_file_exists else "not found"
logging.info(f"Current directory: {current_dir}")
logging.info(f".env file status: {env_file_status}")

# Debugging: Log the values of environment variables after loading
logging.info(f"ES_HOST: {os.getenv('ES_HOST')}")
logging.info(f"ES_PORT: {os.getenv('ES_PORT')}")
logging.info(f"ES_DUMP_INDEX: {os.getenv('ES_DUMP_INDEX')}")

# Raise exception if required environment variables are not set
if not os.getenv("ES_HOST") or not os.getenv("ES_PORT") or not os.getenv("ES_DUMP_INDEX"):
    raise EnvironmentError(f"Required environment variables ES_HOST, ES_PORT, or ES_DUMP_INDEX are not set.\n"
                           f"Current directory: {current_dir}\n"
                           f".env file status: {env_file_status}. Exiting.")

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = os.getenv("ES_PORT", "7020")
ES_INDEX = os.getenv("ES_DUMP_INDEX", "wikipedia_dumps")
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
