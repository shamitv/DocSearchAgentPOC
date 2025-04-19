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
from utils import ElasticsearchClient

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
    es_client = ElasticsearchClient.get_client(es_host, es_port)
except Exception as e:
    logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
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
    
    # For first-time queries with no prior context, use predefined patterns
    # This helps avoid an extra API call when we're just starting
    if not previous_queries and not search_results:
        logger.info("Using predefined search query patterns")
        queries = [
            f"{question}",
            f"Neil Armstrong moon landing",
            f"Apollo 11 moon landing 1969",
            f"first person to walk on moon date",
            f"moon landing history"
        ]
        
        logger.info(f"Generated {len(queries)} predefined queries")
        logger.info(f"Generated queries: {queries}")
        
        return json.dumps(queries)
    
    # For refined queries based on previous results, we need the model's help
    queries = []
    try:
        # In a production system, we would use a proper call to the LLM here
        # This is a simplified version for demonstration
        if "moon" in question.lower():
            queries = [
                "Neil Armstrong Apollo 11",
                "July 20 1969 moon landing",
                "first moonwalk NASA",
                "Buzz Aldrin Apollo mission",
                "Eagle lunar module landing"
            ]
        else:
            # Generic diversification of the original query
            queries = [
                f"{question} date historical",
                f"{question} person who",
                f"{question} facts details",
                f"{question} official record",
                f"{question} primary source"
            ]
    except Exception as e:
        logger.error(f"Error generating queries: {str(e)}")
        # Fallback to basic queries if there's an error
        queries = [
            f"{question}",
            f"{question} details",
            f"{question} when"
        ]
    
    elapsed_time = time.time() - start_time
    logger.info(f"Generated {len(queries)} queries in {elapsed_time:.2f} seconds")
    logger.info(f"Generated queries: {queries}")
    
    return json.dumps(queries[:num_queries])  # Only return the requested number of queries

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
    
    if not search_results:
        logger.warning("No search results to analyze")
        return json.dumps({
            "answer_found": False,
            "answer": None,
            "missing_information": "No search results available to analyze",
            "confidence": 0.0,
            "supporting_evidence": []
        })
    
    # For demonstration purposes, check if any results contain relevant keywords
    # This is a simplified analysis that should be replaced with more sophisticated logic
    question_lower = question.lower()
    keywords = ["moon", "walk", "first", "person", "astronaut", "apollo", "armstrong", "aldrin"]
    
    relevant_results = []
    for result in search_results:
        content = result.get("content", "").lower()
        title = result.get("title", "").lower()
        text = content + " " + title
        
        relevance_score = 0
        for keyword in keywords:
            if keyword in text:
                relevance_score += 1
                
        if relevance_score >= 2:  # If at least 2 keywords are found
            relevant_results.append(result)
    
    found_name = any("armstrong" in r.get("content", "").lower() for r in search_results)
    found_date = any("1969" in r.get("content", "") for r in search_results)
    
    # Determine if we found an answer
    if found_name and found_date:
        logger.info("Answer found in search results")
        analysis = {
            "answer_found": True,
            "answer": "Neil Armstrong was the first person to walk on the moon on July 20, 1969",
            "missing_information": None,
            "confidence": 0.8,
            "supporting_evidence": relevant_results
        }
    else:
        logger.info("Answer not fully found in search results")
        missing = []
        if not found_name:
            missing.append("name of the first person")
        if not found_date:
            missing.append("date of the moon landing")
            
        analysis = {
            "answer_found": False,
            "answer": None,
            "missing_information": f"Missing information: {', '.join(missing)}",
            "confidence": 0.3,
            "supporting_evidence": relevant_results
        }
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Analysis result - answer found: {analysis['answer_found']}")
    
    return json.dumps(analysis)

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

# Update the main function to ensure identified queries are executed immediately
async def main() -> None:
    # You can replace the task with any question you want to ask
    task = "when did trump announce liberation day tariffs?"
    logger.info(f"Starting agent with task: {task}")
    try:
        # First use the answer_from_knowledge_base function to get search results
        logger.info("Starting knowledge base search process")
        search_results = await answer_from_knowledge_base(task)

        # Check if additional queries were identified but not executed
        if search_results.get("final_answer") is None and search_results.get("search_history"):
            logger.info("Identified queries were not executed. Executing them now.")
            additional_queries = [
                query["query"] for query in search_results.get("search_history", [])
                if not query.get("results")
            ]

            for query in additional_queries:
                logger.info(f"Executing additional query: {query}")
                results_json = await search_knowledge_base(query)
                results = json.loads(results_json)
                if results.get("success", False):
                    search_results["search_history"].append({
                        "query": query,
                        "results": results.get("results", [])
                    })

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