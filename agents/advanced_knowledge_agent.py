from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import SystemMessage, UserMessage
import os
import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional
from utils import EnvLoader, LoggerConfig, ElasticsearchClient

# Setup logging using LoggerConfig
LoggerConfig.configure_logging()
logger = logging.getLogger("advanced_knowledge_agent")
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
                        "fields": ["text", "title^5"],  # Title gets higher weight
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
                                 search_results: List[Dict] = None, num_queries: int = 5) -> str:
    """
    Generate multiple search queries based on a question and optionally previous search results
    using an LLM to think of useful queries.
    
    Args:
        question: The original question to answer
        previous_queries: List of previously tried queries (optional)
        search_results: List of previous search results (optional)
        num_queries: Number of queries to generate
        
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
        try:
            response = await query_model_client.create(
                messages=[
                    SystemMessage(content="You are a search query generation assistant. Generate concise, effective search queries."),
                    UserMessage(content=prompt, source="user")
                ]
            )
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
                    f"{question} details",
                    f"{question} when",
                    f"{question} who"
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
    
    # Format search results for the prompt
    formatted_results = ""
    truncated_results = search_results[:10]  # Limit to 10 results to keep prompt size manageable
    
    for i, result in enumerate(truncated_results):
        title = result.get("title", "No title")
        content = result.get("content", "No content")
        score = result.get("score", 0)
        
        # Truncate content if very long to avoid excessive token usage
        content_preview = content
        if len(content) > max_tokens // len(truncated_results):
            content_preview = content[:max_tokens // len(truncated_results)] + "..."
            
        formatted_results += f"RESULT {i+1} (score: {score}):\n"
        formatted_results += f"TITLE: {title}\n"
        formatted_results += f"CONTENT: {content_preview}\n\n"
    
    # Construct a prompt for the LLM to analyze search results
    prompt = f"""You are an expert research analyst. Given a question and search results, your task is to:
1. Analyze if the search results contain an answer to the question
2. Extract and synthesize the answer if found
3. Identify what information is missing if the answer isn't complete
4. Rate your confidence in the answer (0.0-1.0)

QUESTION: {question}

SEARCH RESULTS:
{formatted_results}

Provide your analysis in the following structured format:
1. Answer found (yes/no): Based on whether the search results contain sufficient information to answer the question
2. Answer (if found): A comprehensive answer synthesized from the search results
3. Missing information (if any): What relevant information is missing from the search results
4. Confidence score (0.0-1.0): How confident you are in the answer based on the search results
5. Supporting evidence: List the specific results (by their number) that support your answer
6. Reasoning: Brief explanation of your analysis

FORMAT YOUR RESPONSE AS A JSON OBJECT with the following keys: 
"answer_found" (boolean),
"answer" (string or null),
"missing_information" (string or null), 
"confidence" (float between 0.0 and 1.0),
"supporting_evidence" (array of result numbers),
"reasoning" (string)
"""
    
    try:
        # Call the LLM to analyze the search results
        logger.info("Calling LLM to analyze search results")
        
        # Use the analysis_model_client to call the LLM
        response = await analysis_model_client.create(
            messages=[
                SystemMessage(content="You are a search result analysis assistant that provides accurate, factual JSON responses."),
                UserMessage(content=prompt, source="user")
            ]
        )
        
        # Extract the LLM's response
        analysis_text = response.content
        logger.info("Analysis response received from LLM")
        
        try:
            # Try to parse the LLM's response as JSON
            # The response might already be in JSON format
            if isinstance(analysis_text, str):
                # If it's a string, try to parse it
                # First, try to extract JSON if it's wrapped in markdown code blocks
                if "```json" in analysis_text:
                    json_part = analysis_text.split("```json")[1].split("```")[0].strip()
                    analysis = json.loads(json_part)
                elif "```" in analysis_text:
                    json_part = analysis_text.split("```")[1].split("```")[0].strip()
                    analysis = json.loads(json_part)
                else:
                    # Try to parse the whole text as JSON
                    analysis = json.loads(analysis_text)
            else:
                # If it's not a string (e.g., it might be a dict already)
                analysis = analysis_text
                
            # Ensure all required fields are present
            required_fields = ["answer_found", "answer", "missing_information", "confidence", 
                             "supporting_evidence", "reasoning"]
            for field in required_fields:
                if field not in analysis:
                    # Add missing field with default value
                    if field == "answer_found":
                        analysis[field] = False
                    elif field == "confidence":
                        analysis[field] = 0.0
                    elif field == "supporting_evidence":
                        analysis[field] = []
                    else:
                        analysis[field] = None
            
            # Ensure supporting_evidence is a list of integers
            if not isinstance(analysis["supporting_evidence"], list):
                analysis["supporting_evidence"] = []
            else:
                # Convert any string numbers to integers
                analysis["supporting_evidence"] = [
                    int(ev) if isinstance(ev, str) and ev.isdigit() else ev 
                    for ev in analysis["supporting_evidence"]
                ]
                
            # Ensure confidence is a float between 0 and 1
            try:
                analysis["confidence"] = float(analysis["confidence"])
                if analysis["confidence"] < 0 or analysis["confidence"] > 1:
                    analysis["confidence"] = max(0, min(1, analysis["confidence"]))
            except (ValueError, TypeError):
                analysis["confidence"] = 0.0
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.error(f"Raw LLM response: {analysis_text}")
            
            # Create a fallback analysis based on what might be in the text
            answer_found = "yes" in analysis_text.lower() and "answer found: yes" in analysis_text.lower()
            
            # Try to extract an answer if one was found
            answer = None
            if answer_found:
                # Look for patterns that might indicate the start of an answer
                patterns = ["answer:", "answer is:", "the answer is:"]
                for pattern in patterns:
                    if pattern in analysis_text.lower():
                        answer_part = analysis_text.lower().split(pattern)[1].split("\n")[0].strip()
                        if answer_part:
                            answer = answer_part
                            break
            
            analysis = {
                "answer_found": answer_found,
                "answer": answer,
                "missing_information": "Unable to extract missing information from LLM response",
                "confidence": 0.5 if answer_found else 0.0,
                "supporting_evidence": [],
                "reasoning": "Error parsing LLM response. This is a fallback analysis."
            }
    
    except Exception as e:
        logger.error(f"Error analyzing search results with LLM: {str(e)}")
        # Create a fallback analysis if the LLM call fails
        analysis = {
            "answer_found": False,
            "answer": None,
            "missing_information": f"Error analyzing search results: {str(e)}",
            "confidence": 0.0,
            "supporting_evidence": [],
            "reasoning": "LLM analysis failed due to an error."
        }
    
    # Add reference to actual supporting evidence objects
    if analysis.get("supporting_evidence"):
        # Convert supporting evidence numbers to actual result objects
        evidence_objects = []
        for evidence_num in analysis["supporting_evidence"]:
            try:
                # Evidence numbers are 1-indexed in the prompt
                idx = int(evidence_num) - 1
                if 0 <= idx < len(search_results):
                    evidence_objects.append(search_results[idx])
            except (ValueError, TypeError):
                continue
                
        # Replace the original list of numbers with the actual evidence objects
        analysis["supporting_evidence"] = evidence_objects
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
    logger.info(f"Analysis result - answer found: {analysis.get('answer_found', False)}")
    
    return json.dumps(analysis)

# Define separate model clients for different tasks
logger.info("Initializing OpenAI clients for different tasks")
try:
    # Model for query generation - using a lighter model for this task
    query_model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",  # Lighter model for generating search queries
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.info("Query generation model client initialized")
    
    # Model for analyzing search results - using a more capable model
    analysis_model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",  # More capable model for complex analysis
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.info("Analysis model client initialized")
    
    # Primary model for the agent's conversations
    agent_model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",  # Using a capable model for the assistant agent
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.info("Agent model client initialized")
except Exception as e:
    logger.error(f"Error initializing OpenAI clients: {str(e)}")
    raise

# Define the advanced knowledge agent
logger.info("Creating advanced knowledge agent")
advanced_knowledge_agent = AssistantAgent(
    name="advanced_knowledge_agent",
    model_client=agent_model_client,
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
    task = "How much reciprocal tariffs were put on China by Trump?"
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
        await agent_model_client.close()
        await query_model_client.close()
        await analysis_model_client.close()
        logger.info("Model client connection closed")

if __name__ == "__main__":
    logger.info("Starting main function")
    asyncio.run(main())
    logger.info("Main function completed")