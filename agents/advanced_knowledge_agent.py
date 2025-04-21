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

# Setup logging and obtain logger instance
logger = LoggerConfig.configure_logging()
logger.info("Initializing Advanced Knowledge Agent")

# Load environment variables using EnvLoader
es_host, es_port,es_dump_index ,es_index = EnvLoader.load_env()
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
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create tables if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT,
    start_time TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS query_generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    timestamp TEXT,
    prompt TEXT,
    response TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS search_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    query TEXT,
    timestamp TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS search_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    query TEXT,
    results TEXT,
    timestamp TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS analysis_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    timestamp TEXT,
    prompt TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    result_index INTEGER,
    timestamp TEXT,
    response TEXT
);
''')

# Add tables for LLM metrics
cursor.execute('''
CREATE TABLE IF NOT EXISTS query_llm_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    timestamp TEXT,
    model_name TEXT,
    execution_time_seconds REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    raw_prompt TEXT,
    raw_content TEXT,
    error_message TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS analysis_llm_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    iteration INTEGER,
    result_index INTEGER,
    timestamp TEXT,
    model_name TEXT,
    execution_time_seconds REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    raw_prompt TEXT,
    raw_content TEXT,
    error_message TEXT
);
''')

conn.commit()

# Logging helpers

def log_run_start(question: str) -> str:
    # Generate a random string ID for this run
    run_id = generate_run_id()
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO runs (id, question, start_time) VALUES (?, ?, ?)',
        (run_id, question, ts)
    )
    conn.commit()
    return run_id

def log_llm_metrics(response, start_time, model_name="Unknown", is_query=True, run_id=None, iteration=None, result_index=None, raw_prompt=None):
    """
    Log metrics from an LLM response including tokens and execution time.
    Also stores the metrics in the database.
    
    Args:
        response: The LLM response object
        start_time: The start time of the LLM call
        model_name: Name of the model used
        is_query: Whether this is a query generation (True) or analysis (False)
        run_id: Current run ID for database logging
        iteration: Current iteration number for database logging
        result_index: Result index for analysis operations (None for query operations)
        raw_prompt: The raw text of the prompt that was sent to the LLM
    """
    elapsed_time = time.time() - start_time
    operation = "Query generation" if is_query else "Result analysis"
    
    # Log execution time
    logger.info(f"{operation} execution time: {elapsed_time:.2f} seconds")
    
    # Initialize metrics with defaults
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None
    error_message = None
    raw_content = None
    
    # Try to extract token information if available in the response
    try:
        # Extract raw content for logging
        if hasattr(response, 'content'):
            raw_content = str(response.content)[:1000]  # Limit size for storage
        
        if hasattr(response, 'usage'):
            prompt_tokens = getattr(response.usage, 'prompt_tokens', None)
            completion_tokens = getattr(response.usage, 'completion_tokens', None)
            # Calculate total_tokens if not present
            if hasattr(response.usage, 'total_tokens') and response.usage.total_tokens is not None:
                total_tokens = response.usage.total_tokens
            elif prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            logger.info(f"{operation} token usage - Input: {prompt_tokens}, Output: {completion_tokens}, Total: {total_tokens}")
        else:
            logger.info(f"{operation} completed, but token usage information not available")
            error_message = "Token usage information not available"
    except Exception as e:
        error_message = str(e)
        logger.warning(f"Could not extract token usage information: {str(e)}")
    
    # Store in database if run_id is provided
    if run_id is not None:
        ts = datetime.now(timezone.utc).isoformat()
        
        try:
            if is_query:
                # Store query metrics
                cursor.execute('''
                INSERT INTO query_llm_metrics 
                (run_id, iteration, timestamp, model_name, execution_time_seconds, 
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', 
                (run_id, iteration, ts, model_name, elapsed_time, 
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message))
            else:
                # Store analysis metrics
                cursor.execute('''
                INSERT INTO analysis_llm_metrics 
                (run_id, iteration, result_index, timestamp, model_name, execution_time_seconds, 
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (run_id, iteration, result_index, ts, model_name, elapsed_time,
                prompt_tokens, completion_tokens, total_tokens, raw_prompt, raw_content, error_message))
            
            conn.commit()
            logger.info(f"Logged {operation} metrics to database")
        except Exception as e:
            logger.error(f"Failed to log {operation} metrics to database: {str(e)}")

def log_query_generation(run_id: int, iteration: int, prompt: str, response: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO query_generations (run_id, iteration, timestamp, prompt, response) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, ts, prompt, response)
    )
    conn.commit()


def log_search_query(run_id: int, iteration: int, query: str):
    ts = datetime.now(timezone.utc).isoformat()
    # Insert run_id, iteration, query, timestamp into search_queries (4 placeholders)
    cursor.execute(
        'INSERT INTO search_queries (run_id, iteration, query, timestamp) VALUES (?, ?, ?, ?)',
        (run_id, iteration, query, ts)
    )
    conn.commit()


def log_search_result(run_id: int, iteration: int, query: str, results: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO search_results (run_id, iteration, query, results, timestamp) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, query, results, ts)
    )
    conn.commit()


def log_analysis_prompt(run_id: int, iteration: int, result_index: int, prompt: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO analysis_prompts (run_id, iteration, result_index, timestamp, prompt) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, result_index, ts, prompt)
    )
    conn.commit()


def log_analysis_result(run_id: int, iteration: int, result_index: int, response: str):
    ts = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        'INSERT INTO analysis_results (run_id, iteration, result_index, timestamp, response) VALUES (?, ?, ?, ?, ?)',
        (run_id, iteration, result_index, ts, response)
    )
    conn.commit()

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
            # Format previous search results for the prompt
            formatted_results = ""
            for i, result in enumerate(search_results):
                if i >= 3:  # Limit to first 3 results to keep prompt size reasonable
                    break
                query = result.get("query", "Unknown query")
                results_list = result.get("results", [])
                formatted_results += f"Query: {query}\n"
                for j, res in enumerate(results_list[:2]):  # Only show first 2 results per query
                    title = res.get("title", "No title")
                    content_preview = res.get("content", "")[:100] + "..." if len(res.get("content", "")) > 100 else res.get("content", "")
                    formatted_results += f"  Result {j+1}: {title}\n  Preview: {content_preview}\n"
                formatted_results += "\n"
            
            # Create a prompt for refining queries based on previous results
            prompt = f"""You are a search query generator for a research system. Your goal is to generate refined search queries 
based on initial search results to better answer a user's question.

Original question: "{question}"

Previous queries tried:
{', '.join(previous_queries)}

Results from previous searches:
{formatted_results}

Based on these results, generate {num_queries} new search queries that:
1. Target specific information gaps in the initial results
2. Use different terminology or phrasing that might yield better results
3. Focus on aspects of the question not well covered in initial results
4. Are diverse in approach (entity-focused, date-focused, event-focused, etc.)
5. Would help complete the answer to the original question

Return ONLY a numbered list of search queries, one per line, with no explanations or additional text.
"""
        else:
            # Initial query generation prompt
            prompt = f"""You are a search query generator for a research system. Your goal is to generate effective search queries
to answer a user's question by searching a knowledge base.

Question: "{question}"

Generate {num_queries} different search queries that:
1. Cover different aspects and interpretations of the question
2. Use diverse phrasings and terminology
3. Include specific entities, dates, or events mentioned in the question
4. Vary in specificity (some broad, some narrow)
5. Would collectively help build a comprehensive answer

Return ONLY a numbered list of search queries, one per line, with no explanations or additional text.
"""
        
        # Call the LLM to generate queries
        logger.info("Calling LLM to generate search queries")
        llm_call_start_time = time.time()
        try:
            response = await query_model_client.create(
                messages=[
                    SystemMessage(content="You are a search query generation assistant. Generate concise, effective search queries."),
                    UserMessage(content=prompt, source="user")
                ]
            )
            
            # Log details about the LLM response including token counts
            log_llm_metrics(response, llm_call_start_time, 
                           model_name=getattr(query_model_client, 'model_name', getattr(query_model_client, 'model', 'Unknown')), 
                           is_query=True, run_id=run_id, iteration=iteration, raw_prompt=prompt)
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
            raise
        
        # Process the response to extract queries
        generated_text = response.content
        if isinstance(generated_text, list):
            # If it's a list of function calls, we can't use it for queries
            logger.warning("LLM returned function calls instead of text, using fallbacks")
            queries = [
                question,
                f"{question} facts",
                f"{question} details",
                f"{question} when",
                f"{question} who"
            ]
        else:
            # Use the string content
            logger.info(f"Raw LLM response: {generated_text}")
            
            # Extract queries from the numbered list format
            query_lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
            queries = []
            
            for line in query_lines:
                # Remove numbering (like "1.", "2.", etc.)
                if '. ' in line and line[0].isdigit():
                    query = line.split('. ', 1)[1].strip()
                else:
                    query = line.strip()
                    
                if query:
                    # Remove quotes if present
                    query = query.strip('"\'')
                    queries.append(query)
            
            # Ensure we have the requested number of queries
            if not queries:
                # Fallback if parsing failed
                logger.warning("Failed to parse queries from LLM response, using fallbacks")
                queries = [
                    question,
                    f"{question} facts",
                    f"{question} history",
                    f"{question} date",
                    f"{question} details"
                ]
            
            # Limit to requested number
            queries = queries[:num_queries]
        
    except Exception as e:
        logger.error(f"Error generating queries with LLM: {str(e)}")
        # Fallback to basic queries if there's an error
        queries = [
            question,
            f"{question} facts",
            f"{question} history",
            f"{question} date",
            f"{question} details"
        ]
    
    elapsed_time = time.time() - start_time
    logger.info(f"Generated {len(queries)} queries in {elapsed_time:.2f} seconds")
    logger.info(f"Generated queries: {queries}")
    
    return json.dumps(queries)

async def analyze_search_results(question: str, search_results: List[Dict], max_tokens: int = 30000, 
                              run_id: str = None, iteration: int = None) -> str:
    """
    Analyze search results to determine if they contain an answer to the question.
    
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
        prompt = f"""You are a search result analysis assistant. Given a question and a single search result, determine if this result answers the question.
QUESTION: {question}

RESULT:
TITLE: {title}
CONTENT: {content}

Respond in JSON with keys:
- answer_found (boolean)
- answer (string or null)
- confidence (float between 0.0 and 1.0)
- reasoning (string)
"""
        start_time = time.time()
        try:
            response = await analysis_model_client.create(
                messages=[
                    SystemMessage(content="You are a search result analysis assistant that provides concise JSON outputs."),
                    UserMessage(content=prompt, source="user")
                ]
            )
            raw_content = response.content
            if isinstance(raw_content, str):
                raw_content = raw_content.strip()
                match = re.search(r"```json(.*?)```", raw_content, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                else:
                    match = re.search(r"({[\s\S]*})", raw_content)
                    if match:
                        json_str = match.group(1)
                    else:
                        json_str = raw_content
                analysis = json.loads(json_str)
            else:
                analysis = json.loads(response.content)
        except Exception:
            traceback.print_exc()
            analysis = {"answer_found": False, "answer": None, "confidence": 0.0, "reasoning": "Failed to analyze via LLM."}
            if isinstance(content, str) and "answer is" in content.lower():
                part = content.lower().split("answer is", 1)[1].strip()
                answer_text = part.rstrip(".?!")
                analysis = {
                    "answer_found": True,
                    "answer": answer_text,
                    "confidence": 0.5,
                    "reasoning": "Heuristic fallback match extracted answer."
                }
        log_llm_metrics(response, start_time, model_name="gpt-4o-mini", is_query=False, run_id=run_id, iteration=iteration, raw_prompt=prompt)
        individual_analyses.append({
            "result_index": idx,
            "result": {"title": title, "content": content, "score": result.get("score")},
            "analysis": analysis
        })
        if analysis.get("answer_found") and analysis.get("confidence", 0.0) >= 0.8:
            answer_found = True
            answer_confidence = analysis.get("confidence", 0.0)

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
    model_client_stream=True,  # Enable streaming tokens from the model client
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
    
    run_id = log_run_start(question)
    
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
        
        log_query_generation(run_id, iterations, question, queries_json)
        
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
                
                log_search_query(run_id, iterations, query)
                log_search_result(run_id, iterations, query, results_json)
        
        # Analyze results to see if we found an answer
        if iteration_results:
            logger.info(f"Analyzing {len(iteration_results)} search results")
            analysis_json = await analyze_search_results(question, iteration_results, run_id=run_id, iteration=iterations)
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

        # Then pass the results to the agent
        logger.info("Passing search results to agent for final response")

        # Create a new prompt with the search results
        enhanced_task = f"""
Question: {task}

Here are the search results from the knowledge base:
{json.dumps(search_results, indent=2)}

Please analyze these search results and provide a comprehensive answer to the question.
"""
        # Capture the agent's output instead of just streaming to console
        agent_response = await advanced_knowledge_agent.run(task=enhanced_task)
        # Log LLM metrics for the agent's final response
        log_llm_metrics(agent_response, time.time(), model_name=getattr(agent_model_client, 'model_name', getattr(agent_model_client, 'model', 'Unknown')), is_query=False, run_id=run_id, iteration=None, raw_prompt=enhanced_task)
        logger.info("Agent task completed successfully")
        return search_results, agent_response
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error during agent execution: {str(e)}")
        return None, None
    finally:
        logger.info("Model client connection closed (no explicit close performed)")

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