import os
import json
import traceback
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio

from utils import search_knowledge_base
from agents.advanced_knowledge_agent import (
    generate_search_queries,
    run_agent_with_search_results
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    question: str
    max_iterations: Optional[int] = 5

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

@app.post("/api/agent")
async def run_agent(request: AgentRequest):
    try:
        search_results, agent_response = await run_agent_with_search_results(
            request.question, request.max_iterations
        )
        def extract_agent_response(resp):
            if hasattr(resp, 'messages') and isinstance(resp.messages, list):
                for msg in reversed(resp.messages):
                    if hasattr(msg, 'content') and msg.content:
                        return msg.content
                if resp.messages:
                    return str(resp.messages[0])
            if hasattr(resp, 'content'):
                return resp.content
            if isinstance(resp, list):
                for msg in reversed(resp):
                    if hasattr(msg, 'content') and msg.content:
                        return msg.content
            try:
                return json.loads(resp)
            except Exception:
                return str(resp)
        agent_response_clean = extract_agent_response(agent_response)
        return {
            "search_results": search_results,
            "agent_response": agent_response_clean
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.post("/api/search")
async def run_search_query(request: SearchRequest):
    try:
        results = search_knowledge_base(request.query, max_results=request.max_results)
        return {
            "query": request.query,
            "max_results": request.max_results,
            "results": results.get("results", [])
        }
    except Exception as e:
        return {
            "error": f"Error running search: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.get("/api/queries")
async def generate_queries_endpoint(question: str = Query(...)):
    try:
        queries_json = await generate_search_queries(question)
        queries = json.loads(queries_json)
        return {"queries": queries}
    except Exception as e:
        return {"error": str(e)}

# To run: uvicorn agent_api:app --reload --port 8000
