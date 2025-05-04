import traceback

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage
import os
import asyncio
import json
import time
import re
from typing import List, Dict, Any, Optional
import sqlite3
from datetime import datetime, timezone

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import EnvLoader, LoggerConfig, ElasticsearchClient, search_knowledge_base as utils_search_knowledge_base, generate_run_id, get_llm_client, get_llm_base_url
from agents.metrics_logger import log_llm_metrics, init_metrics_db
from agents.db_init import init_main_db
from agents.db_logging import (
    log_run_start, log_query_generation, log_search_query, log_search_result, log_analysis_prompt, log_analysis_result
)

# Setup logging and obtain logger instance
logger = LoggerConfig.configure_logging()
logger.info("Initializing Advanced Knowledge Agent")

# Load environment variables
env_vars = EnvLoader.load_env()
es_host = env_vars.get("ES_HOST")
es_port = env_vars.get("ES_PORT")
es_dump_index = env_vars.get("ES_DUMP_INDEX")
es_index = env_vars.get("ES_SEARCH_INDEX") # Assuming es_index corresponds to ES_SEARCH_INDEX

logger.info("Environment variables loaded")

# Initialize Elasticsearch client using ElasticsearchClient
logger.info(f"Connecting to Elasticsearch at {es_host}:{es_port}, index: {es_index}")
try:
    es_client = ElasticsearchClient.get_client(es_host, es_port)
except Exception as e:
    logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
    raise

# SQLite DB initialization
DB_PATH = os.getenv('INTERMEDIATE_DB_PATH', 'intermediate_results.db')
conn, cursor = init_main_db(DB_PATH)

# Define search function that returns structured data for better analysis
async def search_knowledge_base(query: str, max_results: int = 5) -> str:
    """
    Wrapper for utils.search_knowledge_base returning JSON string.
    """
    results = utils_search_knowledge_base(query, max_results)
    return json.dumps(results)

# Function to generate multiple search queries
async def generate_search_queries(question: str, previous_queries: List[str] = None, 
                                 search_results: List[Dict] = None, num_queries: int = 5,
                                 run_id: str = None, iteration: int = None) -> str:
    """
    Generate multiple search queries based on a question and optionally previous search results
    using an LLM to think of useful queries.
    
    Args:
        question: The original question to answer
        previous_queries: List of previously tried queries (optional)
        search_results: List of previous search results (optional)
        num_queries: Number of queries to generate
        run_id: Current run ID for database logging
        iteration: Current iteration for database logging
        
    Returns:
        JSON string containing new search queries
    """
    logger.info(f"Generating search queries for question: '{question}'")
    if previous_queries:
        logger.info(f"Previous queries: {previous_queries}")
    start_time = time.time()
    
    # If we're in a refinement stage (after initial queries)
    refinement_mode = previous_queries is not None and search_results is not None
    
    try:
        # Construct a prompt for the LLM based on what stage we're in
        if refinement_mode:
            # Prompt for refinement stage
            prompt = f"""Given the original question and previous search attempts, generate {num_queries} new, more specific search queries.

Original Question: {question}

Previous Queries Tried:
{json.dumps(previous_queries, indent=2)}

Summary of Previous Results:
{json.dumps([res.get('title', 'No Title') for res in search_results], indent=2)}

Generate {num_queries} new queries focusing on aspects potentially missed or needing clarification. Output ONLY a JSON list of strings.
"""
        else:
            # Prompt for initial query generation
            prompt = f"""Generate {num_queries} diverse search queries for the following question. Output ONLY a JSON list of strings.

Question: {question}

Example Output: ["query 1", "query 2", "query 3"]
"""
        
        # Call the LLM to generate queries
        logger.info("Calling LLM to generate search queries")
        llm_call_start_time = time.time()
        try:
            # Use the dedicated query generation client
            response = await query_model_client.chat_completion(
                messages=[UserMessage(content=prompt)],
                temperature=0.5, # Encourage some creativity but stay focused
                max_tokens=500, # Ample space for queries
                response_format={"type": "json_object"} # Request JSON output
            )
            log_llm_metrics("generate_search_queries", query_model_client.model, time.time() - llm_call_start_time, response.usage)
            logger.info(f"LLM response for query generation received: {response.content}")
            
        except Exception as e:
            logger.error(f"LLM call failed during query generation: {str(e)}")
            # Fallback or re-raise depending on desired robustness
            raise
        
        # Process the response to extract queries
        generated_content = response.content
        queries = []
        if isinstance(generated_content, str):
            try:
                # Attempt to parse the string as JSON
                parsed_json = json.loads(generated_content)
                # Check if it's a list of strings
                if isinstance(parsed_json, list) and all(isinstance(item, str) for item in parsed_json):
                    queries = parsed_json
                # Handle cases where the LLM might wrap the list in a dict, e.g., {"queries": [...]} 
                elif isinstance(parsed_json, dict):
                    # Look for a key that likely contains the list (e.g., 'queries', 'results')
                    for key, value in parsed_json.items():
                        if isinstance(value, list) and all(isinstance(item, str) for item in value):
                            queries = value
                            logger.warning(f"Extracted queries from key '{key}' in LLM JSON response.")
                            break
                    if not queries:
                         logger.error(f"LLM returned JSON object but no list of strings found: {generated_content}")
                else:
                    logger.error(f"LLM JSON response is not a list of strings or expected dict: {generated_content}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON for queries: {generated_content}")
                # Attempt to extract list-like structures if JSON parsing fails (less reliable)
                # Example: Find content within square brackets
                import re
                match = re.search(r'\[\s*("[^"\\]*(?:\\.[^"\\]*)*"\s*,?\s*)+\]', generated_content)
                if match:
                    try:
                        queries = json.loads(match.group(0))
                        logger.warning("Recovered queries using regex fallback from non-JSON string.")
                    except json.JSONDecodeError:
                        logger.error("Regex fallback failed to parse extracted list.")
        else:
             logger.error(f"Unexpected LLM response type for queries: {type(generated_content)}")

        # Ensure we don't exceed num_queries requested
        queries = queries[:num_queries]
        
    except Exception as e:
        logger.error(f"Error generating queries with LLM: {str(e)}")
        # Return empty list on error to avoid breaking the flow
        queries = [] 
    
    elapsed_time = time.time() - start_time
    logger.info(f"Generated {len(queries)} queries in {elapsed_time:.2f} seconds")
    logger.info(f"Generated queries: {queries}")
    
    # Log the generated queries to the database
    log_query_generation(cursor, conn, run_id, iteration, question, json.dumps(queries))
    
    return json.dumps(queries)

async def analyze_search_results(question: str, search_results: List[Dict], max_tokens: int = 30000, 
                              run_id: str = None, iteration: int = None) -> str:
    """
    Analyze search results to determine if they contain an answer to the question.
    Only results marked as useful (is_useful=True) are considered for aggregation.
    
    Args:
        question: The question to answer
        search_results: List of search results to analyze
        max_tokens: Maximum tokens to use for analysis
        run_id: Current run ID for database logging
        iteration: Current iteration for database logging
        
    Returns:
        JSON string containing analysis results
    """
    if not search_results:
        return json.dumps({
            "answer_found": False,
            "answer": None,
            "missing_information": "No search results available to analyze",
            "confidence": 0.0,
            "supporting_evidence": [],
            "reasoning": "",
            "individual_results": []
        })

    # De-duplicate search results before analysis
    search_results = dedupe_search_results(search_results)

    individual_analyses = []
    answer_found = False
    answer_confidence = 0.0
    response = None  # Ensure response is always defined
    # Analyze each result separately
    for idx, result in enumerate(search_results, start=1):
        if answer_found and answer_confidence >= 0.8:
            break  # Skip analysis for the rest if answer already found with high confidence
        title = result.get("title", "No title")
        content = result.get("content", "No content")
        logger.info(f"Analyzing result {idx}/{len(search_results)}: {title}")
        prompt = f"""You are a search result analysis assistant. Given a question and a single search result, determine if this result is useful and if it answers the question.

QUESTION: {question}

RESULT:
TITLE: {title}
CONTENT: {content}

For each result, respond in JSON with these keys:
- is_useful (boolean): True if the result contains any information that could help answer the question, otherwise False. If False, the result will be discarded.
- answer_found (boolean): True if the result directly answers the question, otherwise False.
- answer (string or null): The answer if found, otherwise null.
- confidence (float between 0.0 and 1.0): Your confidence that the answer is correct.
- missing_information (string): What is missing from this result to fully answer the question, or empty if nothing is missing.
- summary (string): A concise summary of the relevant information in this result.
- relevant_text (string): The exact text span from the result that is most relevant to the question.
- reasoning (string): Explain your reasoning for the above fields.
"""
        start_time = time.time()
        try:
            # Use the dedicated analysis client
            response = await analysis_model_client.chat_completion(
                messages=[UserMessage(content=prompt)],
                temperature=0.2, # Lower temperature for factual analysis
                max_tokens=1000, # Max tokens for analyzing one result
                response_format={"type": "json_object"} # Request JSON output
            )
            log_llm_metrics("analyze_search_results_individual", analysis_model_client.model, time.time() - start_time, response.usage)
            analysis = json.loads(response.content)
            individual_analyses.append({"result_index": idx, "analysis": analysis})
            logger.info(f"Analysis for result {idx}: {analysis}")
            
            # Check if this result provides a high-confidence answer
            if analysis.get("answer_found") and analysis.get("confidence", 0.0) >= 0.8:
                 answer_found = True
                 answer_confidence = analysis.get("confidence", 0.0)
                 logger.info(f"High-confidence answer found in result {idx}. Confidence: {answer_confidence}")
                 # Optional: break early if high confidence answer found
                 # break 

        except Exception as e:
            logger.error(f"LLM call failed during analysis of result {idx}: {str(e)}")
            # Append a failure record
            individual_analyses.append({
                "result_index": idx,
                "analysis": {"error": f"LLM analysis failed: {str(e)}"}
            })
    # Aggregate individual analyses: pick highest-confidence positive
    positive = [a for a in individual_analyses if a["analysis"].get("answer_found")]
    if positive:
        best = max(positive, key=lambda a: a["analysis"]["confidence"])
        overall = {
            "answer_found": True,
            "answer": best["analysis"].get("answer"),
            "confidence": best["analysis"].get("confidence"),
            "supporting_evidence": [search_results[best["result_index"]-1]],
            "reasoning": best["analysis"].get("reasoning")
        }
    else:
        overall = {
            "answer_found": False,
            "answer": None,
            "confidence": 0.0,
            "supporting_evidence": [],
            "reasoning": "No individual result contained an answer."
        }
    overall["individual_results"] = individual_analyses
    # Log the final aggregated analysis
    log_analysis_result(cursor, conn, run_id, iteration, question, json.dumps(overall))
    return json.dumps(overall)

# Define separate model clients for different tasks
logger.info("Initializing LLM clients for different tasks")
try:
    llm_type = os.getenv("LLM_TYPE", "openai")
    # Model for query generation
    query_model_client = get_llm_client(llm_type, "gpt-4o-mini")
    logger.info("Query generation model client initialized")
    # Model for analyzing search results
    analysis_model_client = get_llm_client(llm_type, "gpt-4o-mini")
    logger.info("Analysis model client initialized")
    # Primary model for the agent's conversations
    agent_model_client = get_llm_client(llm_type, "gpt-4o-mini")
    logger.info("Agent model client initialized")
except Exception as e:
    logger.error(f"Error initializing LLM clients: {str(e)}")
    raise

# Define the advanced knowledge agent
logger.info("Creating advanced knowledge agent")
advanced_knowledge_agent = AssistantAgent(
    name="advanced_knowledge_agent",
    model_client=agent_model_client,
    tools=[search_knowledge_base, generate_search_queries, analyze_search_results],
    system_message=f"""
You are an advanced research assistant that answers questions using a Wikipedia knowledge base through Elasticsearch.

Today's date is {datetime.now().strftime('%-d %B %Y')}. Use this as a reference for any temporal or time-sensitive questions.

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
    model_client_stream=False,  # Disable streaming to allow tool use
)
logger.info("Advanced knowledge agent created successfully")

# Now let's create a runner function that orchestrates the whole process
async def answer_from_knowledge_base(question: str, max_iterations: int = 5) -> Dict[str, Any]:
    """
    Answer a question using the knowledge base with iterative search.
    Returns a dict with the final answer, search history, and the run_id used for logging.
    """
    logger.info(f"Starting search process for question: '{question}'")
    logger.info(f"Maximum iterations allowed: {max_iterations}")
    start_time = time.time()
    
    all_queries = []
    all_results = []
    iterations = 0
    answer_found = False
    final_answer = None
    
    run_id = log_run_start(cursor, conn, generate_run_id, question)
    
    while iterations < max_iterations and not answer_found:
        iterations += 1
        logger.info(f"Starting iteration {iterations}/{max_iterations}")
        
        # Generate search queries
        if iterations == 1:
            # Initial queries
            logger.info("Generating initial queries")
            queries_json = await generate_search_queries(question, run_id=run_id, iteration=iterations)
        else:
            # Refined queries based on previous results
            logger.info("Generating refined queries based on previous results")
            queries_json = await generate_search_queries(question, all_queries, all_results, run_id=run_id, iteration=iterations)
            
        queries = json.loads(queries_json)
        all_queries.extend(queries)
        
        log_query_generation(cursor, conn, run_id, iterations, question, queries_json)
        
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
                
                log_search_query(cursor, conn, run_id, iterations, query)
                log_search_result(cursor, conn, run_id, iterations, query, results_json)
        
        # Analyze results to see if we found an answer
        # Flatten all results across all iterations for deduplication and analysis
        all_flat_results = [item for iteration in all_results for item in iteration.get("results", [])]
        if all_flat_results:
            logger.info(f"Analyzing {len(all_flat_results)} search results (deduped across all iterations)")
            analysis_json = await analyze_search_results(question, all_flat_results, run_id=run_id, iteration=iterations)
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
        "processing_time": elapsed_time,
        "run_id": run_id
    }

async def run_agent_with_search_results(task: str, max_iterations: int = 1):
    """
    Run the agent with search results and return the agent's response.
    max_iterations: maximum number of search iterations to use in answer_from_knowledge_base
    Returns a tuple: (search_results dict, agent_response)
    """
    logger.info(f"Starting agent with task: {task}")
    try:
        logger.info("Starting knowledge base search process")
        search_results = await answer_from_knowledge_base(task, max_iterations)
        run_id = search_results.get("run_id")

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

        # Use new attributes from analyze_search_results for the agent's prompt
        final_answer = search_results.get("final_answer")
        useful_results = []
        if final_answer and "individual_results" in final_answer:
            for res in final_answer["individual_results"]:
                analysis = res.get("analysis", {})
                if analysis.get("is_useful", False):
                    useful_results.append({
                        "title": res["result"].get("title"),
                        "summary": analysis.get("summary"),
                        "relevant_text": analysis.get("relevant_text"),
                        "confidence": analysis.get("confidence"),
                        "reasoning": analysis.get("reasoning"),
                        "missing_information": analysis.get("missing_information"),
                        "answer_found": analysis.get("answer_found"),
                        "answer": analysis.get("answer")
                    })
        logger.info(f"Number of useful results considered for final analysis: {len(useful_results)}")
        # Create a new prompt with the filtered, useful results and their attributes
        enhanced_task = f"""
Question: {task}

Here are the useful search results from the knowledge base (filtered by is_useful=True):
{json.dumps(useful_results, indent=2)}

Each result includes:
- title: The result's title
- summary: A concise summary of relevant information
- relevant_text: The most relevant text span
- confidence: Confidence score for the answer
- reasoning: Explanation for the analysis
- missing_information: What is missing, if anything
- answer_found: Whether an answer was found in this result
- answer: The answer if found

Please analyze these search results and provide a comprehensive answer to the question, citing the most relevant results and explaining your reasoning.
"""
        agent_response = await advanced_knowledge_agent.run(task=enhanced_task)
        log_llm_metrics(agent_response, time.time(), model_name=getattr(agent_model_client, 'model_name', getattr(agent_model_client, 'model', 'Unknown')), is_query=False, run_id=run_id, iteration=None, raw_prompt=enhanced_task)
        logger.info("Agent task completed successfully")
        return search_results, agent_response
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error during agent execution: {str(e)}")
        return None, None
    finally:
        logger.info("Model client connection closed (no explicit close performed)")

def dedupe_search_results(results, key_fields=None):
    """
    Remove duplicate search results based on key fields (e.g., title and content).
    Accepts a flat list of results (across all iterations).
    Returns a list of unique results, preserving order.
    """
    if key_fields is None:
        key_fields = ["title", "content"]
    seen = set()
    unique_results = []
    for result in results:
        key = tuple(result.get(field, "") for field in key_fields)
        if key not in seen:
            seen.add(key)
            unique_results.append(result)
    return unique_results

async def main() -> None:
    # You can replace the task with any question you want to ask
    #task = "How much reciprocal tariffs were put on China by Trump?"
    task = "Which schools are present in Trsat"
    search_results, agent_response = await run_agent_with_search_results(task)
    print("\n===== FINAL SEARCH RESULTS =====\n")
    print(json.dumps(search_results, indent=2))
    print("\n===== AGENT RESPONSE =====\n")
    print(agent_response)

if __name__ == "__main__":
    logger.info("Starting main function")
    asyncio.run(main())
    logger.info("Main function completed")