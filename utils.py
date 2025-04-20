# utils.py
import os
import logging
import time
import json
import sys
import random
import string
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

class EnvLoader:
    @staticmethod
    def load_env():
        load_dotenv()
        env_file_path = os.path.abspath(".env")
        logging.info(f"Absolute path of .env file: {env_file_path}")

        current_dir = os.getcwd()
        env_file_exists = os.path.exists(".env")
        env_file_status = "found" if env_file_exists else "not found"
        logging.info(f"Current directory: {current_dir}")
        logging.info(f".env file status: {env_file_status}")

        es_host = os.getenv("ES_HOST")
        es_port = os.getenv("ES_PORT")
        es_dump_index = os.getenv("ES_DUMP_INDEX")
        es_search_index = os.getenv("ES_SEARCH_INDEX", "wikipedia")  # Default to "wikipedia" if not set
        
        logging.info(f"ES_HOST: {es_host}")
        logging.info(f"ES_PORT: {es_port}")
        logging.info(f"ES_DUMP_INDEX: {es_dump_index}")
        logging.info(f"ES_SEARCH_INDEX: {es_search_index}")
        
        if not es_host or not es_port or not es_dump_index:
            raise EnvironmentError(f"Required environment variables ES_HOST, ES_PORT, or ES_DUMP_INDEX are not set.\n"
                                   f"Current directory: {current_dir}\n"
                                   f".env file status: {env_file_status}. Exiting.")
        
        return es_host, es_port, es_dump_index, es_search_index

class LoggerConfig:
    @staticmethod
    def configure_logging():
        # Determine log filename based on the main script filename
        main_file = os.path.basename(sys.argv[0])
        log_filename = os.path.splitext(main_file)[0] + ".log"
        # Create a dedicated logger with INFO level and attach file and stream handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        logger_name = os.path.splitext(main_file)[0]
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

class ElasticsearchClient:
    @staticmethod
    def get_client(es_host: str, es_port: str):
        try:
            es_client = Elasticsearch([f"http://{es_host}:{es_port}"])
            if es_client.ping():
                logging.info("Successfully connected to Elasticsearch")
            else:
                logging.error("Could not connect to Elasticsearch")
            return es_client
        except Exception as e:
            logging.error(f"Error connecting to Elasticsearch: {str(e)}")
            raise

def generate_run_id(length: int = 20) -> str:
    """
    Generate a random alphanumeric string of given length for use as a run primary key.
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))

def search_knowledge_base(query, max_results=5):
    """
    Search the Elasticsearch Wikipedia index for articles matching the query.
    
    Args:
        query (str): The search query
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
        
    Returns:
        dict: JSON serializable dictionary containing search results
    """
    logger = logging.getLogger("wiki_search")
    logger.info(f"Searching knowledge base for: '{query}' (max results: {max_results})")
    start_time = time.time()
    
    try:
        # Load environment variables
        es_host, es_port, es_dump_index, es_search_index = EnvLoader.load_env()
        
        # Get Elasticsearch client
        es_client = ElasticsearchClient.get_client(es_host, es_port)
        
        # Log the index being used
        logger.info(f"Using Elasticsearch index: {es_search_index}")
        
        # Log search query details
        logger.info(f"Executing search on index '{es_search_index}' with query: '{query}', max_results: {max_results}")
        
        # Execute search
        response = es_client.search(
            index=es_search_index,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "title^5"],  # The title field has 5x higher weight
                        "type": "best_fields"
                    }
                },
                "size": max_results
            }
        )
        
        hits = response.get("hits", {}).get("hits", [])
        total_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        
        if not hits:
            logger.warning(f"No results found for query: '{query}'")
            return {
                "success": False,
                "query": query,
                "message": "No results found",
                "results": []
            }
        
        results = []
        for hit in hits:
            source = hit["_source"]
            result = {
                "title": source.get("title", "No title"),
                "content": source.get("text", "No content"),
                "score": hit["_score"],
                "id": hit["_id"],
                "metadata": source.get("metadata", {})
            }
            results.append(result)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f} seconds, found {len(results)} results")
        
        return {
            "success": True,
            "query": query,
            "total_hits": total_hits,
            "elapsed_time": elapsed_time,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return {
            "success": False,
            "query": query,
            "message": f"Error: {str(e)}",
            "results": []
        }