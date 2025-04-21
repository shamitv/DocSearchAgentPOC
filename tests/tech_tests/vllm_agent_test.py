from openai import OpenAI
import os
from dotenv import load_dotenv
from utils import LoggerConfig
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo

load_dotenv()
vllm_url = os.getenv("VLLM_URL")

client = OpenAI(api_key="EMPTY", base_url=vllm_url)

logger = LoggerConfig.configure_logging()

# Query the list of models
models = client.models.list()
model_id = models.data[0].id

#Ref : https://microsoft.github.io/autogen/stable//reference/python/autogen_core.models.html#autogen_core.models.ModelInfo
model_info = ModelInfo(
    id=model_id,
    object="model",
    created=0,
    owned_by="vllm",
    root=model_id,
    parent=None,
    permission=[],
    max_tokens=8192,         # Adjust as needed
    context_length=8192,     # Adjust as needed
    prompt_token_cost=0,
    completion_token_cost=0,
    function_calling=True,  # Enable function calling
    json_output=True,
    structured_output=True,
    function_call_token_cost=0,
    family="vllm",
    vision=False,            # Set to True if your model supports vision
)

vllm_llm = OpenAIChatCompletionClient(
    model=model_id,
    api_key="NotRequired",
    base_url=vllm_url,
    model_info=model_info,
)

logger.info("llm config used")

async def get_current_time() -> str:
    return "The current time is 12:00 PM."


async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model=model_id,
        api_key="NotRequired",
        base_url=vllm_url,
        model_info=model_info,
    )
    agent = AssistantAgent(name="assistant", model_client=model_client, tools=[get_current_time])

    await Console(
        agent.on_messages_stream(
            [TextMessage(content="What is the current time?", source="user")], CancellationToken()
        )
    )


asyncio.run(main())