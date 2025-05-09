from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import logging
import time
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import asyncio
from utils import EnvLoader, LoggerConfig, ElasticsearchClient

# Configure logging and obtain logger instance
logger = LoggerConfig.configure_logging()
logger.info("Initializing Knowledge Agent")

# Load environment variables
try:
    env_vars = EnvLoader.load_env()
    es_host = env_vars.get("ES_HOST")
    es_port = env_vars.get("ES_PORT")
    es_index = env_vars.get("ES_SEARCH_INDEX") # Assuming es_index corresponds to ES_SEARCH_INDEX
    if not all([es_host, es_port, es_index]):
        raise EnvironmentError("Required Elasticsearch environment variables not set.")
except EnvironmentError as e:
    logger.error(f"Failed to load environment variables: {e}")
    # Handle the error appropriately, maybe exit or raise
    raise

logger.info(f"Connecting to Elasticsearch at {es_host}:{es_port}, index: {es_index}")
try:
    es_client = ElasticsearchClient.get_client(es_host, es_port)
except Exception as e:
    logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
    raise

# Define search function to query Elasticsearch
async def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the Elasticsearch knowledge base for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        String containing search results
    """
    logger.info(f"Searching knowledge base for: '{query}' (max results: {max_results})")
    start_time = time.time()
    
    try:
        response = es_client.search(
            index=es_index,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "title"],
                        "type": "best_fields"
                    }
                },
                "size": max_results
            }
        )
        
        results = []
        hits = response.get("hits", {}).get("hits", [])
        
        if not hits:
            logger.warning(f"No results found for query: '{query}'")
            return f"No results found for query: '{query}'"
        
        logger.info(f"Found {len(hits)} results for query: '{query}'")
            
        for i, hit in enumerate(hits):
            source = hit["_source"]
            title = source.get("title", "No title")
            text = source.get("text", "No content")
            score = hit["_score"]
            
            # Log the title of each analyzed article
            logger.info(f"Analyzed article title: {title}")
            
            results.append(f"Result {i+1} (Score: {score}):\nTitle: {title}\nContent: {text[:500]}...\n")
            
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f} seconds")
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return f"Error searching knowledge base: {str(e)}"

# Define a model client
logger.info("Initializing OpenAI client")
try:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini", # Using a more capable model for complex reasoning
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.info("OpenAI client initialized")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    raise

# Define the knowledge agent
logger.info("Creating knowledge agent")
knowledge_agent = AssistantAgent(
    name="knowledge_agent",
    model_client=model_client,
    tools=[search_knowledge_base],
    system_message="""
You are an intelligent research assistant that helps answer questions using a knowledge base of Wikipedia articles.
You follow a systematic approach:

1. First, create a set of diverse search queries that might help answer the user's question
2. For each query, search the knowledge base and analyze the results
3. If you find an answer, provide it along with supporting evidence from the search results
4. If you don't find an answer, create new search queries based on previous results
5. Continue refining your queries until you either find an answer or reach 5 attempts

When providing answers:
- Cite specific information from the search results
- Distinguish between facts from the knowledge base and your own reasoning
- Be honest when you don't know or can't find information
- Structure your response clearly to show your search strategy and findings
""",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client
)
logger.info("Knowledge agent created successfully")

# Run the agent and stream the messages to the console
async def main() -> None:
    # You can replace the task with any question you want to ask
    task = "How much Tariff was put on India by the US in April 2025?"
    logger.info(f"Starting agent with task: {task}")
    try:
        await Console(knowledge_agent.run_stream(task=task))
        logger.info("Agent task completed successfully")
    except Exception as e:
        logger.error(f"Error during agent execution: {str(e)}")
    finally:
        # Close the connection to the model client
        logger.info("Closing model client connection")
        await model_client.close()
        logger.info("Model client connection closed")

if __name__ == "__main__":
    logger.info("Starting main function")
    asyncio.run(main())
    logger.info("Main function completed")