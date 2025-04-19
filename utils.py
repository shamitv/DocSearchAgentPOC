# utils.py
import os
import logging
from dotenv import load_dotenv

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

        logging.info(f"ES_HOST: {es_host}")
        logging.info(f"ES_PORT: {es_port}")
        logging.info(f"ES_DUMP_INDEX: {es_dump_index}")

        if not es_host or not es_port or not es_dump_index:
            raise EnvironmentError(f"Required environment variables ES_HOST, ES_PORT, or ES_DUMP_INDEX are not set.\n"
                                   f"Current directory: {current_dir}\n"
                                   f".env file status: {env_file_status}. Exiting.")

        return es_host, es_port, es_dump_index

class LoggerConfig:
    @staticmethod
    def configure_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("download_and_index_dumps.log"),
                logging.StreamHandler()
            ]
        )