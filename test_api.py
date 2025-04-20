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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
            
        # Try to import the search function
        try:
            # First try to import from the main module
            try:
                import utils
                if hasattr(utils, 'search_knowledge_base'):
                    search_fn = utils.search_knowledge_base
                else:
                    raise ImportError("search_knowledge_base not found in utils")
            except ImportError:
                # If not found, try to find it in any module
                search_fn = None
                for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
                    if file.endswith('.py') and file != 'test_api.py':
                        module_name = file[:-3]
                        try:
                            spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(os.path.abspath(__file__)), file))
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            if hasattr(module, 'search_knowledge_base'):
                                search_fn = module.search_knowledge_base
                                break
                        except Exception:
                            continue
                
                if not search_fn:
                    raise ImportError("Could not find search_knowledge_base function in any module")
                    
            # Run the search
            results = search_fn(query, max_results=max_results)
            
            return jsonify({
                "query": query,
                "max_results": max_results,
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Error importing or running search function: {str(e)}")
            return jsonify({
                "error": f"Error running search: {str(e)}",
                "traceback": traceback.format_exc()
            }), 500
            
    except Exception as e:
        logger.error(f"Error processing search request: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Only for development/testing
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Changed default from 5000 to 5001
    app.run(host='0.0.0.0', port=port, debug=True)