from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_knowledge_agent")
logger.info("Initializing Advanced Knowledge Agent")

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

# Initialize Elasticsearch client
es_host = os.getenv("ES_HOST", "localhost")
es_port = os.getenv("ES_PORT", "9200")
es_index = os.getenv("ES_INDEX", "wikipedia")

logger.info(f"Connecting to Elasticsearch at {es_host}:{es_port}, index: {es_index}")
try:
    es_client = Elasticsearch([f"http://{es_host}:{es_port}"])
    if es_client.ping():
        logger.info("Successfully connected to Elasticsearch")
    else:
        logger.error("Could not connect to Elasticsearch")
except Exception as e:
    logger.error(f"Error connecting to Elasticsearch: {str(e)}")
    raise

# Define search function that returns structured data for better analysis
async def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Search the Elasticsearch knowledge base for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        JSON string containing search results
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
                        "fields": ["text", "title^2"],  # Title gets higher weight
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
            return json.dumps({"success": False, "query": query, "message": f"No results found for query: '{query}'"})
            
        logger.info(f"Found {len(hits)} results for query: '{query}'")
        
        for i, hit in enumerate(hits):
            source = hit["_source"]
            title = source.get("title", "No title")
            text = source.get("text", "No content")
            score = hit["_score"]
            
            results.append({
                "rank": i+1,
                "score": score,
                "title": title,
                "content": text[:1000] + ("..." if len(text) > 1000 else "")
            })
            
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f} seconds")
        
        return json.dumps({
            "success": True,
            "query": query,
            "total_hits": len(hits),
            "results": results
        })
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return json.dumps({"success": False, "query": query, "message": f"Error searching knowledge base: {str(e)}"})

# Function to generate multiple search queries
async def generate_search_queries(question: str, previous_queries: List[str] = None, 
                                 search_results: List[Dict] = None, num_queries: int = 3) -> str:
    """
    Generate multiple search queries based on a question and optionally previous search results.
    
    Args:
        question: The original question to answer
        previous_queries: List of previously tried queries (optional)
        search_results: List of previous search results (optional)
        num_queries: Number of queries to generate
        
    Returns:
        JSON string containing new search queries
    """
    logger.info(f"Generating search queries for question: '{question}'")
    logger.info(f"Previous queries: {previous_queries}")
    start_time = time.time()
    
    prompt = f"""
    Based on the original question: "{question}"
    
    Generate {num_queries} different search queries that could help find information to answer this question.
    """
    
    if previous_queries:
        prompt += f"""
        
        These queries have been tried already:
        {json.dumps(previous_queries, indent=2)}
        """
    
    if search_results:
        prompt += f"""
        
        Here are the previous search results:
        {json.dumps(search_results, indent=2)}
        
        Based on these results, generate new queries that might find more relevant information.
        """
        
    prompt += """
    
    Return your response as a JSON array of strings, with each string being a search query.
    """
    
    # In a real implementation, you'd call your language model here
    # For simplicity, we'll return example queries
    if not previous_queries:
        # First set of queries if no previous queries
        example_queries = [
            f"{question}",
            f"wikipedia {question}",
            f"facts about {question}"
        ]
    else:
        # Refined queries (this would normally be generated by the language model)
        example_queries = [
            f"definition of {question}",
            f"details {question} when year date",
            f"explanation {question} history context"
        ]
    
    elapsed_time = time.time() - start_time
    logger.info(f"Generated {len(example_queries)} queries in {elapsed_time:.2f} seconds")
    logger.info(f"Generated queries: {example_queries}")
    
    return json.dumps(example_queries)

# Function to analyze search results and determine if the answer was found
async def analyze_search_results(question: str, search_results: List[Dict], max_tokens: int = 1500) -> str:
    """
    Analyze search results to determine if they answer the original question.
    
    Args:
        question: The original question
        search_results: List of search results to analyze
        max_tokens: Maximum tokens to include in prompt
        
    Returns:
        JSON string with analysis results including if answer was found and the answer itself
    """
    logger.info(f"Analyzing search results for question: '{question}'")
    logger.info(f"Number of search results to analyze: {len(search_results)}")
    start_time = time.time()
    
    prompt = f"""
    Original question: "{question}"
    
    Based on the following search results, determine if the question can be answered.
    If it can be answered, provide the answer with citations to specific search results.
    If it cannot be answered completely, identify what information is missing.
    
    Search results:
    {json.dumps(search_results, indent=2)}
    
    Return your response as a JSON object with these fields:
    - answer_found (boolean): whether the question can be answered with these results
    - answer (string): the answer to the question if found, otherwise null
    - missing_information (string): description of what information is missing, if answer_found is false
    - confidence (number): confidence score between 0-1
    - supporting_evidence (array): list of evidence from the search results that support the answer
    """
    
    # In a real implementation, you'd call your language model here
    # For demonstration, we'll return a placeholder
    example_analysis = {
        "answer_found": False,
        "answer": None,
        "missing_information": "This would be determined by the language model based on search results",
        "confidence": 0.0,
        "supporting_evidence": []
    }
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Analysis result - answer found: {example_analysis['answer_found']}")
    
    return json.dumps(example_analysis)

# Define a model client
logger.info("Initializing OpenAI client")
try:
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",  # Using a more capable model for complex reasoning
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.info("OpenAI client initialized")
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {str(e)}")
    raise

# Define the advanced knowledge agent
logger.info("Creating advanced knowledge agent")
advanced_knowledge_agent = AssistantAgent(
    name="advanced_knowledge_agent",
    model_client=model_client,
    tools=[search_knowledge_base, generate_search_queries, analyze_search_results],
    system_message="""
You are an advanced research assistant that answers questions using a Wikipedia knowledge base through Elasticsearch.
Follow this systematic approach for every question:

1. GENERATE INITIAL QUERIES:
   - Start by generating 3-5 different search queries for the user's question
   - Queries should be diverse to cover different aspects and phrasings

2. EXECUTE SEARCHES & ANALYZE RESULTS:
   - Search the knowledge base with each query
   - Analyze search results to determine if they contain the answer
   - Extract relevant information from search results

3. ITERATIVE REFINEMENT:
   - If the answer isn't found, generate new queries based on what you've learned
   - Consider synonyms, different phrasings, or more specific terms
   - Use information from previous searches to guide your new queries

4. TERMINATION:
   - Continue until you find a complete answer OR
   - Reach a maximum of 5 search iterations OR
   - Determine that the knowledge base likely doesn't contain the answer

5. ANSWER FORMULATION:
   - When answering, cite specific sources from the search results
   - Clearly distinguish between facts from the knowledge base and your reasoning
   - Structure your final answer with the most relevant information first

Make your thinking process explicit - explain what queries you're trying and why.
Remember that each tool call counts as one API call, so be strategic about your searches.
""",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client
)
logger.info("Advanced knowledge agent created successfully")

# Now let's create a runner function that orchestrates the whole process
async def answer_from_knowledge_base(question: str, max_iterations: int = 5) -> Dict[str, Any]:
    """
    Answer a question using the knowledge base with iterative search.
    
    Args:
        question: The question to answer
        max_iterations: Maximum number of search iterations
        
    Returns:
        Dictionary with the final answer and search history
    """
    logger.info(f"Starting search process for question: '{question}'")
    logger.info(f"Maximum iterations allowed: {max_iterations}")
    start_time = time.time()
    
    all_queries = []
    all_results = []
    iterations = 0
    answer_found = False
    final_answer = None
    
    while iterations < max_iterations and not answer_found:
        iterations += 1
        logger.info(f"Starting iteration {iterations}/{max_iterations}")
        
        # Generate search queries
        if iterations == 1:
            # Initial queries
            logger.info("Generating initial queries")
            queries_json = await generate_search_queries(question)
        else:
            # Refined queries based on previous results
            logger.info("Generating refined queries based on previous results")
            queries_json = await generate_search_queries(question, all_queries, all_results)
            
        queries = json.loads(queries_json)
        all_queries.extend(queries)
        
        # Execute searches
        logger.info(f"Executing {len(queries)} searches")
        iteration_results = []
        for query in queries:
            logger.info(f"Searching with query: '{query}'")
            results_json = await search_knowledge_base(query)
            results = json.loads(results_json)
            if results.get("success", False):
                iteration_results.extend(results.get("results", []))
                all_results.append({
                    "query": query,
                    "results": results.get("results", [])
                })
        
        # Analyze results to see if we found an answer
        if iteration_results:
            logger.info(f"Analyzing {len(iteration_results)} search results")
            analysis_json = await analyze_search_results(question, iteration_results)
            analysis = json.loads(analysis_json)
            
            if analysis.get("answer_found", False):
                logger.info("Answer found!")
                answer_found = True
                final_answer = analysis
                break
            else:
                logger.info("Answer not found in this iteration, continuing search")
        else:
            logger.warning("No results found in this iteration")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Search process completed in {elapsed_time:.2f} seconds after {iterations} iterations")
    logger.info(f"Answer found: {answer_found}")
    
    return {
        "question": question,
        "answer_found": answer_found,
        "final_answer": final_answer,
        "iterations": iterations,
        "search_history": all_results,
        "total_queries": len(all_queries),
        "processing_time": elapsed_time
    }

# Run the agent and stream the messages to the console
async def main() -> None:
    # You can replace the task with any question you want to ask
    task = "Who was the first person to walk on the moon and when did it happen?"
    logger.info(f"Starting agent with task: {task}")
    try:
        # First use the answer_from_knowledge_base function to get search results
        logger.info("Starting knowledge base search process")
        search_results = await answer_from_knowledge_base(task)
        
        # Then pass the results to the agent
        logger.info("Passing search results to agent for final response")
        
        # Create a new prompt with the search results
        enhanced_task = f"""
Question: {task}

Here are the search results from the knowledge base:
{json.dumps(search_results, indent=2)}

Please analyze these search results and provide a comprehensive answer to the question.
"""
        # Run the agent with the enhanced task containing search results
        await Console(advanced_knowledge_agent.run_stream(task=enhanced_task))
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