#!/usr/bin/env python3
# tests/integration/test_search_knowledge_base_integration.py

import unittest
import json
import asyncio
import os
import sys
import logging
from datetime import datetime

# Import necessary modules from main project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agents.advanced_knowledge_agent import search_knowledge_base
from utils import ElasticsearchClient, EnvLoader


class TestSearchKnowledgeBaseIntegration(unittest.TestCase):
    """Integration tests for the search_knowledge_base function with a real Elasticsearch instance."""

    @classmethod
    def setUpClass(cls):
        """Set up the test class by initializing Elasticsearch connection and creating test data."""
        # Configure logging for tests
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        cls.logger = logging.getLogger(__name__)
        cls.logger.info("Setting up integration tests")
        
        # Load environment variables
        try:
            cls.es_host, cls.es_port, cls.es_index = EnvLoader.load_env()
            cls.logger.info(f"Using ES host: {cls.es_host}, port: {cls.es_port}, index: {cls.es_index}")
        except Exception as e:
            cls.logger.error(f"Failed to load environment variables: {str(e)}")
            raise
        
        # Create Elasticsearch client
        try:
            cls.es_client = ElasticsearchClient.get_client(cls.es_host, cls.es_port)
            cls.logger.info("Connected to Elasticsearch")
        except Exception as e:
            cls.logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise
        
        # Create a test index with a unique name for isolation
        cls.test_index = f"test_index_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        cls.logger.info(f"Creating test index: {cls.test_index}")
        
        # Define mapping for the test index
        mapping = {
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "text": {"type": "text"}
                }
            }
        }
        
        # Create the test index with mapping
        try:
            cls.es_client.indices.create(index=cls.test_index, body=mapping)
            cls.logger.info("Test index created successfully")
        except Exception as e:
            cls.logger.error(f"Failed to create test index: {str(e)}")
            raise
        
        # Insert test data
        test_docs = [
            {
                "title": "Python Programming",
                "text": "Python is a high-level, interpreted programming language known for its readability and versatility."
            },
            {
                "title": "Elasticsearch Guide",
                "text": "Elasticsearch is a distributed, RESTful search and analytics engine capable of solving a growing number of use cases."
            },
            {
                "title": "Machine Learning",
                "text": "Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data."
            },
            {
                "title": "Trump Speech",
                "text": "In his speech on Liberation Day, Trump mentioned the importance of freedom and democracy. He highlighted the sacrifices made by veterans."
            }
        ]
        
        # Bulk insert the test documents
        bulk_data = []
        for i, doc in enumerate(test_docs):
            bulk_data.append({"index": {"_index": cls.test_index, "_id": f"{i+1}"}})
            bulk_data.append(doc)
        
        try:
            cls.es_client.bulk(index=cls.test_index, body=bulk_data, refresh=True)
            cls.logger.info(f"Inserted {len(test_docs)} test documents")
        except Exception as e:
            cls.logger.error(f"Failed to insert test documents: {str(e)}")
            raise
            
        # Replace the global es_index with our test index temporarily
        cls.original_es_index = os.environ.get("ES_DUMP_INDEX")
        os.environ["ES_DUMP_INDEX"] = cls.test_index
        cls.logger.info(f"Set ES_DUMP_INDEX to {cls.test_index}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests by deleting the test index."""
        cls.logger.info("Cleaning up after tests")
        
        # Delete the test index
        try:
            cls.es_client.indices.delete(index=cls.test_index)
            cls.logger.info(f"Test index {cls.test_index} deleted")
        except Exception as e:
            cls.logger.error(f"Failed to delete test index: {str(e)}")
        
        # Restore original es_index if it existed
        if cls.original_es_index:
            os.environ["ES_DUMP_INDEX"] = cls.original_es_index
            cls.logger.info(f"Restored ES_DUMP_INDEX to {cls.original_es_index}")

    async def run_search(self, query, max_results=5):
        """Helper method to run search and return parsed results."""
        result_json = await search_knowledge_base(query, max_results)
        return json.loads(result_json)

    def test_basic_search(self):
        """Test basic search functionality."""
        # Search for Python
        results = asyncio.run(self.run_search("Python programming"))
        
        self.assertTrue(results["success"])
        self.assertEqual(results["query"], "Python programming")
        
        # Should match the Python document
        found_python = False
        for result in results["results"]:
            if "Python" in result["title"]:
                found_python = True
                break
                
        self.assertTrue(found_python, "Should find the Python document")

    def test_multi_term_search(self):
        """Test search with multiple terms that should match different documents."""
        results = asyncio.run(self.run_search("machine learning elasticsearch"))
        
        self.assertTrue(results["success"])
        
        # Should match both Elasticsearch and Machine Learning documents
        titles = [result["title"] for result in results["results"]]
        self.assertTrue(any("Elasticsearch" in title for title in titles), 
                        "Should find the Elasticsearch document")
        self.assertTrue(any("Machine Learning" in title for title in titles), 
                        "Should find the Machine Learning document")

    def test_max_results_limit(self):
        """Test that max_results parameter correctly limits results."""
        # Request only 2 results
        results = asyncio.run(self.run_search("programming", max_results=2))
        
        self.assertTrue(results["success"])
        self.assertLessEqual(len(results["results"]), 2, 
                          "Should return no more than max_results documents")

    def test_no_matches(self):
        """Test search with no matches."""
        results = asyncio.run(self.run_search("xyzabcnonexistent"))
        
        self.assertFalse(results["success"])
        self.assertIn("No results found", results["message"])

    def test_specific_content_search(self):
        """Test search for specific content that should match a document."""
        results = asyncio.run(self.run_search("Trump Liberation Day"))
        
        self.assertTrue(results["success"])
        
        # Should find the Trump document
        found_trump = False
        for result in results["results"]:
            if "Trump" in result["content"]:
                found_trump = True
                break
                
        self.assertTrue(found_trump, "Should find the Trump document")


if __name__ == "__main__":
    unittest.main()