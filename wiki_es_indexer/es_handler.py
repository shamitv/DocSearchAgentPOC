import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

# Assuming utils.py is in the parent directory relative to this package
try:
    from utils import EnvLoader, ElasticsearchClient, LoggerConfig
except ImportError:
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import EnvLoader, ElasticsearchClient, LoggerConfig

logger = LoggerConfig.configure_logging() # Use the same log file for consistency

class ElasticsearchHandler:
    """Handles Elasticsearch client initialization and bulk indexing operations."""

    def __init__(self, host, port, index_name, max_workers=20):
        """
        Initializes the Elasticsearch client and ThreadPoolExecutor.

        Args:
            host (str): Elasticsearch host.
            port (str): Elasticsearch port.
            index_name (str): The name of the Elasticsearch index.
            max_workers (int): Maximum number of worker threads for bulk indexing.
        """
        self.index_name = index_name
        self.es_client = self._initialize_es_client(host, port)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"ElasticsearchHandler initialized for index '{index_name}' with {max_workers} workers.")

    def _initialize_es_client(self, host, port):
        """Initializes and returns the Elasticsearch client."""
        try:
            client = ElasticsearchClient.get_client(host, port)
            logger.info(f"Elasticsearch client connected to {host}:{port}")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
            sys.exit(1) # Exit if ES connection fails

    def submit_bulk_index(self, documents):
        """
        Formats documents and submits a bulk indexing task to the ThreadPoolExecutor.

        Args:
            documents (list): A list of document dictionaries to index.
        """
        if not documents:
            return

        actions = []
        for doc in documents:
            # Ensure metadata and id exist, log warning if not (though should be guaranteed by caller)
            doc_id = doc.get("metadata", {}).get("id")
            if not doc_id:
                logger.warning(f"Document missing metadata.id, skipping: {str(doc)[:100]}...")
                continue

            action = {"index": {"_index": self.index_name, "_id": doc_id}}
            actions.append(action)
            # Add indexed_on timestamp if not present (though process_dump adds it)
            if "indexed_on" not in doc:
                 doc["indexed_on"] = datetime.now(timezone.utc).isoformat()
            actions.append(doc)

        if not actions:
            logger.debug("No valid actions generated for bulk submission.")
            return

        try:
            # Submit the bulk operation to the executor
            self.executor.submit(self.es_client.bulk, operations=actions, index=self.index_name)
            logger.debug(f"Submitted bulk task for {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Failed to submit bulk indexing task: {e}")

    def shutdown(self, wait=True):
        """Shuts down the ThreadPoolExecutor."""
        logger.info("Shutting down ElasticsearchHandler's executor...")
        self.executor.shutdown(wait=wait)
        logger.info("Executor shutdown complete.")

    def get_client(self):
        """Returns the underlying Elasticsearch client instance."""
        return self.es_client

    def index_document(self, document):
        """
        Indexes a single document in Elasticsearch.
    
        Args:
            document (dict): Document dictionary to index.
    
        Returns:
            bool: True if indexing was successful, False otherwise.
        """
        if not document:
            logger.warning("Empty document provided for indexing.")
            return False
    
        # Ensure metadata and id exist
        doc_id = document.get("metadata", {}).get("id")
        if not doc_id:
            logger.warning(f"Document missing metadata.id, cannot index: {str(document)[:100]}...")
            return False
    
        # Add indexed_on timestamp if not present
        if "indexed_on" not in document:
            document["indexed_on"] = datetime.now(timezone.utc).isoformat()
    
        try:
            # Index the document directly
            self.es_client.index(
                index=self.index_name,
                id=doc_id,
                document=document
            )
            logger.debug(f"Successfully indexed document with id: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to index document with id {doc_id}: {str(e)}")
            return False
