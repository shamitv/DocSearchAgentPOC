from openai import OpenAI
import os
from dotenv import load_dotenv
from utils import LoggerConfig

load_dotenv()
vllm_url = os.getenv("VLLM_URL")

client = OpenAI(api_key="EMPTY", base_url=vllm_url)

logger = LoggerConfig.configure_logging()

# Query the list of models
models = client.models.list()
logger.info("Available models:")
for model in models.data:
    logger.info(model.id)

# Send a chat completion request
logger.info("Sending request to VLLM")
response = client.chat.completions.create(model=models.data[0].id,
messages=[{"role": "user", "content": "What is the most common language in Switzerland?"}],)
logger.info("Chat completion response:")
logger.info(response.choices[0].message.content)