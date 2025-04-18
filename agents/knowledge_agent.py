from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import asyncio

# Load environment variables from .env file
load_dotenv()

# Initialize Elasticsearch client
es_host = os.getenv("ES_HOST", "localhost")
es_port = os.getenv("ES_PORT", "9200")
es_index = os.getenv("ES_INDEX", "wikipedia")

es_client = Elasticsearch([f"http://{es_host}:{es_port}"])

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
            return f"No results found for query: '{query}'"
            
        for i, hit in enumerate(hits):
            source = hit["_source"]
            title = source.get("title", "No title")
            text = source.get("text", "No content")
            score = hit["_score"]
            
            results.append(f"Result {i+1} (Score: {score}):\nTitle: {title}\nContent: {text[:500]}...\n")
            
        return "\n".join(results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# Define a model client
model_client = OpenAIChatCompletionClient(
    model="gpt-4.1-mini", # Using a more capable model for complex reasoning
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Define the knowledge agent
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

# Run the agent and stream the messages to the console
async def main() -> None:
    # You can replace the task with any question you want to ask
    task = "Who was the first person to walk on the moon and when did it happen?"
    await Console(knowledge_agent.run_stream(task=task))
    # Close the connection to the model client
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())