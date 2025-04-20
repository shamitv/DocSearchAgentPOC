#!/usr/bin/env python3
# tests/unit/test_search_knowledge_base.py

import unittest
import json
import asyncio
from unittest.mock import patch, MagicMock

# Import the function under test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agents.advanced_knowledge_agent import search_knowledge_base


class TestSearchKnowledgeBase(unittest.TestCase):
    """Unit tests for the search_knowledge_base function."""

    @patch('agents.advanced_knowledge_agent.es_client')
    def test_successful_search(self, mock_es_client):
        """Test search with successful results."""
        # Set up the mock response from Elasticsearch
        mock_es_client.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_score": 0.9,
                        "_source": {
                            "title": "Test Title 1",
                            "text": "This is test content 1"
                        }
                    },
                    {
                        "_score": 0.8,
                        "_source": {
                            "title": "Test Title 2",
                            "text": "This is test content 2"
                        }
                    }
                ]
            }
        }

        # Call the function and get the result
        result = asyncio.run(search_knowledge_base("test query", max_results=5))
        result_dict = json.loads(result)

        # Assertions
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["query"], "test query")
        self.assertEqual(result_dict["total_hits"], 2)
        self.assertEqual(len(result_dict["results"]), 2)
        self.assertEqual(result_dict["results"][0]["title"], "Test Title 1")
        self.assertEqual(result_dict["results"][1]["title"], "Test Title 2")

        # Verify search was called with correct parameters
        mock_es_client.search.assert_called_once()
        args, kwargs = mock_es_client.search.call_args
        self.assertEqual(kwargs["body"]["query"]["multi_match"]["query"], "test query")
        self.assertEqual(kwargs["size"], 5)

    @patch('agents.advanced_knowledge_agent.es_client')
    def test_no_results_found(self, mock_es_client):
        """Test search with no results."""
        # Set up the mock to return no hits
        mock_es_client.search.return_value = {
            "hits": {
                "total": {"value": 0},
                "hits": []
            }
        }

        # Call the function and get the result
        result = asyncio.run(search_knowledge_base("empty query"))
        result_dict = json.loads(result)

        # Assertions
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["query"], "empty query")
        self.assertIn("No results found", result_dict["message"])

    @patch('agents.advanced_knowledge_agent.es_client')
    def test_elasticsearch_exception(self, mock_es_client):
        """Test handling of Elasticsearch exception."""
        # Configure the mock to raise an exception
        mock_es_client.search.side_effect = Exception("Test error")

        # Call the function and get the result
        result = asyncio.run(search_knowledge_base("error query"))
        result_dict = json.loads(result)

        # Assertions
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["query"], "error query")
        self.assertIn("Error searching knowledge base", result_dict["message"])
        self.assertIn("Test error", result_dict["message"])

    @patch('agents.advanced_knowledge_agent.es_client')
    def test_max_results_parameter(self, mock_es_client):
        """Test that max_results parameter is properly used."""
        # Set up a basic mock response
        mock_es_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_score": 0.9,
                        "_source": {
                            "title": "Test Title",
                            "text": "Test content"
                        }
                    }
                ]
            }
        }

        # Call with different max_results values
        asyncio.run(search_knowledge_base("test query", max_results=10))
        
        # Verify the size parameter
        args, kwargs = mock_es_client.search.call_args
        self.assertEqual(kwargs["size"], 10)

    @patch('agents.advanced_knowledge_agent.es_client')
    def test_content_truncation(self, mock_es_client):
        """Test that long content is properly truncated in results."""
        # Create a very long text
        long_text = "a" * 2000
        
        # Set up the mock response with long text
        mock_es_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_score": 0.9,
                        "_source": {
                            "title": "Long Content",
                            "text": long_text
                        }
                    }
                ]
            }
        }

        # Call the function
        result = asyncio.run(search_knowledge_base("long content"))
        result_dict = json.loads(result)

        # Assertions
        self.assertTrue(result_dict["success"])
        self.assertEqual(len(result_dict["results"][0]["content"]), 1003)  # 1000 chars + "..." suffix
        self.assertTrue(result_dict["results"][0]["content"].endswith("..."))


if __name__ == '__main__':
    unittest.main()