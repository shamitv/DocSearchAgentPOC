import os
import json
import asyncio
import logging
import chainlit as cl
from typing import Dict, Any, List
from dotenv import load_dotenv
from agents.advanced_knowledge_agent import (
    search_knowledge_base,
    generate_search_queries,
    analyze_search_results,
    answer_from_knowledge_base,
    advanced_knowledge_agent,
    model_client
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chainlit_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chainlit_app")

# Load environment variables
load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session."""
    logger.info("Starting new chat session")
    
    # Set the initial message
    await cl.Message(
        content="Welcome to the Advanced Knowledge Agent! I can answer questions using a Wikipedia knowledge base. What would you like to know?",
        author="Advanced Knowledge Agent"
    ).send()
    
    # Store the model client in user session
    cl.user_session.set("model_client", model_client)

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages from the user."""
    logger.info(f"Received message: {message.content}")
    
    # Get user question
    question = message.content
    
    # Create a thinking message that will be updated with progress
    thinking_msg = cl.Message(content="Thinking...", author="System")
    await thinking_msg.send()
    
    try:
        # Start search process
        await thinking_msg.update(content="Generating initial search queries...")
        
        # Generate initial queries
        queries_json = await generate_search_queries(question)
        queries = json.loads(queries_json)
        
        # Update thinking message with the queries
        query_list = "\n".join([f"- {q}" for q in queries])
        await thinking_msg.update(content=f"Generated search queries:\n{query_list}\n\nSearching knowledge base...")
        
        # Track all results for display
        all_results = []
        search_elements = []
        
        # Execute searches for each query
        for i, query in enumerate(queries):
            # Create a search element to display results
            search_element = cl.Text(
                name=f"search-{i}",
                content=f"Searching for: {query}",
                display="inline"
            )
            await search_element.send()
            search_elements.append(search_element)
            
            # Execute search
            results_json = await search_knowledge_base(query)
            results = json.loads(results_json)
            
            # Update search element with results
            if results.get("success", False):
                result_items = results.get("results", [])
                all_results.extend(result_items)
                
                # Format results for display
                result_text = f"Results for '{query}':\n"
                for r in result_items:
                    result_text += f"- {r.get('title')}: {r.get('content')[:100]}...\n"
                
                await search_element.update(content=result_text)
            else:
                await search_element.update(content=f"No results found for '{query}'")
        
        # Analyze results
        await thinking_msg.update(content="Analyzing search results...")
        if all_results:
            analysis_json = await analyze_search_results(question, all_results)
            analysis = json.loads(analysis_json)
            
            # Check if answer was found
            if analysis.get("answer_found", False):
                answer = analysis.get("answer", "")
                confidence = analysis.get("confidence", 0)
                evidence = analysis.get("supporting_evidence", [])
                
                # Format evidence
                evidence_text = ""
                for e in evidence:
                    evidence_text += f"- {e.get('title')}: {e.get('content')[:150]}...\n"
                
                # Create final answer message
                final_answer = f"### Answer\n{answer}\n\n**Confidence**: {confidence*100:.0f}%\n\n### Supporting Evidence\n{evidence_text}"
                await cl.Message(content=final_answer, author="Knowledge Agent").send()
            else:
                # If no direct answer, use the agent to generate a comprehensive response
                missing = analysis.get("missing_information", "")
                
                # Create a prompt with search results for the agent
                enhanced_task = f"""
Question: {question}

Here are the search results from the knowledge base:
{json.dumps(all_results, indent=2)}

Missing information: {missing}

Please analyze these search results and provide the best possible answer to the question.
"""
                # Remove the thinking message since we're now using the agent
                await thinking_msg.update(content="Processing final answer...")
                
                # Use the advanced agent to generate a response
                final_response = ""
                async for chunk in advanced_knowledge_agent.run_stream(task=enhanced_task):
                    if isinstance(chunk, str):
                        final_response += chunk
                        await thinking_msg.update(content=f"Generating answer...\n\n{final_response}")
                
                # Send the final answer
                await cl.Message(content=final_response, author="Knowledge Agent").send()
        else:
            await cl.Message(
                content="I couldn't find any relevant information in the knowledge base. Could you try rephrasing your question?",
                author="Knowledge Agent"
            ).send()
            
        # Remove the thinking message
        await thinking_msg.remove()
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await thinking_msg.update(content=f"An error occurred: {str(e)}")
        await cl.Message(
            content="I encountered an error while processing your request. Please try again.",
            author="System"
        ).send()

@cl.on_chat_end
async def on_chat_end():
    """Clean up resources when the chat session ends."""
    logger.info("Ending chat session")
    
    # Get the model client from user session
    model_client = cl.user_session.get("model_client")
    
    # Close the model client connection if it exists
    if model_client:
        logger.info("Closing model client connection")
        await model_client.close()
        logger.info("Model client connection closed")

if __name__ == "__main__":
    # Note: This won't be called when running with `chainlit run`
    # It's here for documentation purposes
    logger.info("Chainlit app initialized")
