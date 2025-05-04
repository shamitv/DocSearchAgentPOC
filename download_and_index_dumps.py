import traceback

import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
import logging
from utils import EnvLoader, LoggerConfig, ElasticsearchClient
from index_wikipedia import process_dump

def main():
    print("Script starting...")

    # Configure logging and obtain logger instance
    logger = LoggerConfig.configure_logging()

    logger.info("Script started")
    print(f"Current logging level: {logging.getLevelName(logger.getEffectiveLevel())}")

    # Load environment variables using EnvLoader
    env_vars = EnvLoader.load_env()
    ES_HOST = env_vars.get("ES_HOST")
    ES_PORT = env_vars.get("ES_PORT")
    ES_DUMP_INDEX = env_vars.get("ES_DUMP_INDEX")
    ES_INDEX = env_vars.get("ES_SEARCH_INDEX") # Assuming ES_INDEX corresponds to ES_SEARCH_INDEX

    DUMP_DIR = "./data/wikidump/"
    DUMP_URL = "https://dumps.wikimedia.org/other/incr/enwiki/"
    CUTOFF_DATE = datetime(2025, 4, 1)

    # Directory to store dumps (set via environment variable)
    DUMP_SUBDIR = os.getenv("DUMP_SUBDIR", "./data/wikidump/2/")
    os.makedirs(DUMP_SUBDIR, exist_ok=True)

    # Ensure dump directory exists
    os.makedirs(DUMP_DIR, exist_ok=True)

    # Use the ElasticsearchClient utility to get the Elasticsearch client
    try:
        es = ElasticsearchClient.get_client(ES_HOST, ES_PORT)
    except Exception as e:
        logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
        raise

    # Create index if it doesn't exist
    if not es.indices.exists(index=ES_DUMP_INDEX):
        es.indices.create(index=ES_DUMP_INDEX)

    # Scrape the incremental dumps page
    logger.info(f"Fetching dump list from {DUMP_URL}...")
    try:
        resp = requests.get(DUMP_URL)
        resp.raise_for_status()
        logger.info(f"Successfully fetched dump list from {DUMP_URL}.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch dump list from {DUMP_URL}: {e}")
        raise
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find all links to directories
    links = soup.find_all('a')
    logger.info(f"Found {len(links)} links on the page.")
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

    print(f"Current logging level: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")

    for filename, dump_date in dump_files:
        file_url = DUMP_URL + filename
        local_path = os.path.join(DUMP_SUBDIR, filename.replace('/', '_'))  # Replace slashes for local storage
        # Check if already processed in ES
        es_id = filename
        try:
            doc = es.get(index=ES_DUMP_INDEX, id=es_id)
            if doc.get('found') and doc['_source'].get('status') == 'processed':
                logger.info(f"Already processed: {filename}")
                continue
            # Download if not present
            if not os.path.exists(local_path):
                logger.info(f"Downloading {filename}...")
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
            else:
                logger.info(f"Already downloaded: {filename}")
            # Index metadata in ES
            es.index(index=ES_DUMP_INDEX, id=es_id, document={
            'filename': filename,
            'date': dump_date.isoformat(),
            'status': 'downloaded',
            'local_path': local_path,
            'url': file_url
            })
            logger.info(f"Indexed {filename} in ES as 'downloaded'.")

            # Process downloaded dump and update status
            try:
                logger.info(f"Processing dump file: {local_path}...")
                process_dump(local_path)
                es.update(index=ES_DUMP_INDEX, id=es_id, body={'doc': {'status': 'processed'}})
                logger.info(f"Updated {filename} in ES as 'processed'.")
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Failed to process dump {filename}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {file_url}: {e}")
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    logger.info("Done.")

if __name__ == "__main__":
    main()