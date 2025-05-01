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
    _loaded = False
    _env_vars = {}  # Cache environment variables in a dictionary

    @staticmethod
    def _find_env():
        load_dotenv()
        env_file_path = os.path.abspath(".env")
        logging.info(f"Absolute path of .env file: {env_file_path}")

        current_dir = os.getcwd()
        env_file_exists = os.path.exists(".env")
        env_file_status = "found" if env_file_exists else "not found"
        logging.info(f"Current directory: {current_dir}")
        logging.info(f".env file status: {env_file_status}")

    @staticmethod
    def load_env():
        if EnvLoader._loaded:
            # Return the cached dictionary
            return EnvLoader._env_vars

        EnvLoader._find_env() # Call the new method

        es_host = os.getenv("ES_HOST")
        es_port = os.getenv("ES_PORT")
        es_dump_index = os.getenv("ES_DUMP_INDEX")
        es_search_index = os.getenv("ES_SEARCH_INDEX", "wikipedia")  # Default to "wikipedia" if not set
        dump_file_path = os.getenv("DUMP_FILE_PATH") # Load dump file path

        logging.info(f"ES_HOST: {es_host}")
        logging.info(f"ES_PORT: {es_port}")
        logging.info(f"ES_DUMP_INDEX: {es_dump_index}")
        logging.info(f"ES_SEARCH_INDEX: {es_search_index}")
        logging.info(f"DUMP_FILE_PATH: {dump_file_path}") # Log dump file path

        if not es_host or not es_port or not es_dump_index:
            # Define these variables here only if needed for the error message
            current_dir = os.getcwd()
            env_file_exists = os.path.exists(".env")
            env_file_status = "found" if env_file_exists else "not found"
            raise EnvironmentError(f"Required environment variables ES_HOST, ES_PORT, or ES_DUMP_INDEX are not set.\\\\n"
                                   f"Current directory: {current_dir}\\\\n"
                                   f".env file status: {env_file_status}. Exiting.")

        # Cache the loaded values in the dictionary
        EnvLoader._env_vars = {
            "ES_HOST": es_host,
            "ES_PORT": es_port,
            "ES_DUMP_INDEX": es_dump_index,
            "ES_SEARCH_INDEX": es_search_index,
            "DUMP_FILE_PATH": dump_file_path
        }
        EnvLoader._loaded = True

        # Return the dictionary
        return EnvLoader._env_vars

    @classmethod
    def get_dump_file_path(cls):
        # Load env vars (returns cached dict if already loaded)
        env_vars = cls.load_env()
        
        dump_file_path = env_vars.get("DUMP_FILE_PATH")
        
        if not dump_file_path:
            # Raise error if DUMP_FILE_PATH wasn't set/found during load_env
            raise EnvironmentError("DUMP_FILE_PATH environment variable is not set or loaded.")
        
        # Return the value from the dictionary
        return dump_file_path


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
    logger = logging.getLogger("wiki_search") # Assuming logger is configured elsewhere or get it appropriately
    logger.info(f"Searching knowledge base for: '{query}' (max results: {max_results})")
    start_time = time.time()
    
    try:
        # Load environment variables as a dictionary
        env_vars = EnvLoader.load_env()
        
        # Get values from the dictionary
        es_host = env_vars.get("ES_HOST")
        es_port = env_vars.get("ES_PORT")
        es_search_index = env_vars.get("ES_SEARCH_INDEX")
        
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

# LLM Client Factory

def get_llm_base_url(llm_type: str) -> str:
    """
    Returns the base URL for the LLM API depending on llm_type.
    For 'local', returns VLLM_URL from env. For 'openai', returns None.
    """
    if llm_type == 'local':
        return os.getenv("VLLM_URL")
    return None

def get_llm_client(llm_type: str, model: str = None):
    """
    Factory function to create an LLM client based on llm_type and model name only.
    All other logic (API key, base_url, model_info) is handled internally.
    If llm_type is 'local', delegates to get_local_llm_client.
    """
    if llm_type == 'openai':
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        api_key = os.getenv("OPENAI_API_KEY")
        return OpenAIChatCompletionClient(model=model, api_key=api_key)
    elif llm_type == 'local':
        return get_local_llm_client(model)
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}")

def get_local_llm_client(model: str = None):
    """
    Create a local LLM client (e.g., vLLM) using environment variables for base_url and model_info.
    If model is None, query available models and pick the first one.
    """
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelInfo
    from openai import OpenAI
    import os
    vllm_url = os.getenv("VLLM_URL")
    api_key = os.getenv("VLLM_API_KEY", "NotRequired")
    if vllm_url is None:
        raise ValueError("VLLM_URL environment variable is not set.")
    #Set model to None to force query available models
    model = None
    # If model is None, query the available models from the vLLM server
    if model is None:
        client = OpenAI(api_key="EMPTY", base_url=vllm_url)
        models = client.models.list()
        if not models.data:
            raise RuntimeError("No models available from local LLM server.")
        model = models.data[0].id
    # Use a generic ModelInfo for vLLM
    model_info = ModelInfo(
        id=model,
        object="model",
        created=0,
        owned_by="vllm",
        root=model,
        parent=None,
        permission=[],
        max_tokens=30000,
        context_length=30000,
        prompt_token_cost=0,
        completion_token_cost=0,
        function_calling=True,
        json_output=True,
        structured_output=True,
        function_call_token_cost=0,
        family="vllm",
        vision=False,
    )
    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=vllm_url,
        model_info=model_info,
    )