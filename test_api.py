#!/usr/bin/env python3
# test_api.py

import os
import json
import unittest
import sys
import importlib.util
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from io import StringIO
from contextlib import redirect_stdout
import importlib
import sqlite3

# Import the search_knowledge_base function directly
from utils import search_knowledge_base
import asyncio  # Added to run async generate_search_queries
from agents.advanced_knowledge_agent import generate_search_queries, answer_from_knowledge_base, run_agent_with_search_results  # Import query generator and main agent function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize SQLite connection to intermediate results DB
DB_PATH = os.getenv('INTERMEDIATE_DB_PATH', 'intermediate_results.db')
db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db_conn.row_factory = sqlite3.Row

# Directories where test files are located
TEST_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'unit'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests', 'integration')
]

def get_test_modules():
    """Find all test modules in the test directories."""
    test_modules = []
    
    for test_dir in TEST_DIRS:
        if not os.path.exists(test_dir):
            logger.warning(f"Test directory not found: {test_dir}")
            continue
            
        for file in os.listdir(test_dir):
            if file.startswith('test_') and file.endswith('.py'):
                module_path = os.path.join(test_dir, file)
                module_name = file[:-3]  # Remove .py extension
                
                # Extract test class and methods
                try:
                    # Load the module
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find test classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, unittest.TestCase) and attr != unittest.TestCase:
                            test_class = {
                                'id': attr_name.lower(),
                                'name': attr_name,
                                'description': attr.__doc__ or f"Tests in {attr_name}",
                                'filepath': module_path,
                                'tests': []
                            }
                            
                            # Find test methods in the class
                            for method_name in dir(attr):
                                if method_name.startswith('test_'):
                                    method = getattr(attr, method_name)
                                    if callable(method):
                                        test_class['tests'].append({
                                            'id': method_name,
                                            'name': method_name,
                                            'description': method.__doc__ or f"Test {method_name}"
                                        })
                            
                            test_modules.append(test_class)
                except Exception as e:
                    logger.error(f"Error loading module {module_name}: {str(e)}")
                    continue
    
    return test_modules

def run_test(test_class_name, test_method=None):
    """Run a specific test or all tests in a class and return results."""
    results = {}
    start_time = None
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Capture stdout to get test output
    captured_output = StringIO()
    
    try:
        # Find the module that contains the test class
        test_class = None
        module_path = None
        
        for test_dir in TEST_DIRS:
            if not os.path.exists(test_dir):
                continue
                
            for file in os.listdir(test_dir):
                if file.startswith('test_') and file.endswith('.py'):
                    module_path = os.path.join(test_dir, file)
                    module_name = file[:-3]
                    
                    # Load the module
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for the test class
                    if hasattr(module, test_class_name):
                        test_class = getattr(module, test_class_name)
                        break
            
            if test_class:
                break
        
        if not test_class:
            return {"error": f"Test class {test_class_name} not found"}
        
        # Set up the test suite
        if test_method:
            # Run a specific test method
            suite = loader.loadTestsFromName(f"{test_method}", test_class)
        else:
            # Run all tests in the class
            suite = loader.loadTestsFromTestCase(test_class)
        
        # Run the tests with captured output
        runner = unittest.TextTestRunner(stream=captured_output, verbosity=2)
        
        with redirect_stdout(captured_output):
            test_result = runner.run(suite)
        
        # Get test output
        output = captured_output.getvalue()
        
        # Process the results
        success = test_result.wasSuccessful()
        failures = [(str(test), str(err)) for test, err in test_result.failures]
        errors = [(str(test), str(err)) for test, err in test_result.errors]
        
        results = {
            "success": success,
            "tests_run": test_result.testsRun,
            "failures": failures,
            "errors": errors,
            "output": output
        }
        
    except Exception as e:
        logger.error(f"Error running test {test_class_name}.{test_method}: {str(e)}")
        results = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "output": captured_output.getvalue()
        }
    
    return results


@app.route('/api/tests', methods=['GET'])
def list_tests():
    """API endpoint to list all available tests."""
    try:
        tests = get_test_modules()
        return jsonify(tests)
    except Exception as e:
        logger.error(f"Error listing tests: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/tests/run', methods=['POST'])
def execute_test():
    """API endpoint to run a specific test."""
    try:
        data = request.json
        test_class = data.get('suite')
        test_method = data.get('test')
        
        if not test_class:
            return jsonify({"error": "Test class name is required"}), 400
            
        result = run_test(test_class, test_method)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error executing test: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['POST'])
def run_search_query():
    """API endpoint to run a search query."""
    try:
        data = request.json
        query = data.get('query')
        max_results = data.get('max_results', 5)
        
        if not query:
            return jsonify({"error": "Search query is required"}), 400
            
        # Use the directly imported search_knowledge_base function
        results = search_knowledge_base(query, max_results=max_results)
        
        return jsonify({
            "query": query,
            "max_results": max_results,
            "results": results.get("results", [])
        })
            
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        return jsonify({
            "error": f"Error running search: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

# New endpoint to generate search queries for a sample question
@app.route('/api/queries', methods=['GET'])
def generate_queries_endpoint():
    """API endpoint to generate search queries from a sample question."""
    question = request.args.get('question')
    if not question:
        return jsonify({"error": "Question parameter is required"}), 400
    try:
        # Run the async generate_search_queries function
        queries_json = asyncio.run(generate_search_queries(question))
        queries = json.loads(queries_json)
        return jsonify({"queries": queries})
    except Exception as e:
        logger.error(f"Error generating queries: {str(e)}")
        return jsonify({"error": str(e)}), 500

# New endpoint to run the advanced knowledge agent
@app.route('/api/agent', methods=['POST'])
def run_agent():
    data = request.json
    question = data.get('question')
    max_iterations = data.get('max_iterations', 5)
    if not question:
        return jsonify({"error": "Question parameter is required"}), 400
    try:
        search_results, agent_response = asyncio.run(run_agent_with_search_results(question, max_iterations))
        # Make agent_response JSON serializable and more helpful
        def extract_agent_response(resp):
            # If it's a TaskResult/messages structure, extract the most relevant content
            if hasattr(resp, 'messages') and isinstance(resp.messages, list):
                # Find the last assistant message with content
                for msg in reversed(resp.messages):
                    if hasattr(msg, 'content') and msg.content:
                        return msg.content
                # Fallback: return string of first message
                if resp.messages:
                    return str(resp.messages[0])
            # If it has 'content' directly
            if hasattr(resp, 'content'):
                return resp.content
            # If it's a list of messages
            if isinstance(resp, list):
                for msg in reversed(resp):
                    if hasattr(msg, 'content') and msg.content:
                        return msg.content
            # Fallback: try to serialize or str
            try:
                return json.loads(resp)
            except Exception:
                return str(resp)
        agent_response_clean = extract_agent_response(agent_response)
        return jsonify({
            "search_results": search_results,
            "agent_response": agent_response_clean
        })
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# Endpoint to list all runs
@app.route('/api/runs', methods=['GET'])
def list_runs():
    """API endpoint to list all runs from intermediate_results.db"""
    try:
        cursor = db_conn.cursor()
        cursor.execute('SELECT id, question, start_time FROM runs ORDER BY start_time DESC')
        rows = cursor.fetchall()
        runs = [dict(row) for row in rows]
        return jsonify(runs)
    except Exception as e:
        logger.error(f"Error listing runs: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Endpoint to get detailed logs for a specific run
@app.route('/api/runs/<run_id>', methods=['GET'])
def get_run_details(run_id):
    """API endpoint to fetch all intermediate logs for a given run"""
    try:
        cursor = db_conn.cursor()
        data = {}
        tables = [
            'query_generations', 'search_queries', 'search_results',
            'analysis_prompts', 'analysis_results',
            'query_llm_metrics', 'analysis_llm_metrics'
        ]
        for table in tables:
            cursor.execute(f'SELECT * FROM {table} WHERE run_id = ? ORDER BY timestamp', (run_id,))
            rows = cursor.fetchall()
            data[table] = [dict(row) for row in rows]
        return jsonify({"run_id": run_id, "details": data})
    except Exception as e:
        logger.error(f"Error fetching run details for {run_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Simple UI to view runs and steps
@app.route('/', methods=['GET'])
def index():
    """Serve a simple HTML UI to browse run history"""
    return '''
<!DOCTYPE html>
<html>
<head>
  <title>Run History Viewer</title>
  <style>body{font-family:sans-serif;}#runs{width:30%;float:left;padding:10px;}#details{width:65%;float:right;padding:10px;}li{cursor:pointer;margin-bottom:5px;}pre{background:#f4f4f4;padding:10px;}</style>
</head>
<body>
  <h1>Run History Viewer</h1>
  <div id="runs"><h2>Runs</h2><ul id="runList"></ul></div>
  <div id="details"><h2>Details</h2><div id="runDetails"></div></div>
  <script>
    async function fetchRuns(){
      const res = await fetch('/api/runs');
      const runs = await res.json();
      const list = document.getElementById('runList');
      list.innerHTML = '';
      runs.forEach(r=>{
        const li = document.createElement('li');
        li.textContent = `${r.start_time} - ${r.question}`;
        li.onclick = ()=>showDetails(r.id);
        list.appendChild(li);
      });
    }
    async function showDetails(id){
      const res = await fetch(`/api/runs/${id}`);
      const data = await res.json();
      const container = document.getElementById('runDetails');
      container.innerHTML = '<h3>Details for run ' + id + '</h3>';
      for(const [table, entries] of Object.entries(data.details)){
        container.innerHTML += `<h4>${table}</h4><pre>${JSON.stringify(entries, null, 2)}</pre>`;
      }
    }
    fetchRuns();
  </script>
</body>
</html>
'''

# Only for development/testing
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Changed default from 5000 to 5001
    app.run(host='0.0.0.0', port=port, debug=True)