import streamlit as st
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Elasticsearch configuration
ES_HOST = os.getenv("ES_HOST")
ES_PORT = int(os.getenv("ES_PORT"))  
ES_INDEX = os.getenv("ES_INDEX")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Log Elasticsearch parameters
logger.info(f"Elasticsearch Host: {ES_HOST}")
logger.info(f"Elasticsearch Port: {ES_PORT}")
logger.info(f"Elasticsearch Index: {ES_INDEX}")

# Initialize Elasticsearch client
es = Elasticsearch([{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}])

# Streamlit app
st.title("Elasticsearch Search App")

# Input for search query
query = st.text_input("Enter your search query:")

# Add a slider to filter by year
year_filter = st.sidebar.slider("Select Year", min_value=2000, max_value=2025, value=2025)

# Trigger search on pressing Enter
if query:
    # Elasticsearch search query with year filter
    response = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title", "text"]
                            }
                        }
                    ],
                    "filter": [
                        {
                            "range": {
                                "metadata.timestamp": {
                                    "gte": f"{year_filter}-01-01T00:00:00",
                                    "lte": f"{year_filter}-12-31T23:59:59"
                                }
                            }
                        }
                    ]
                }
            }
        }
    )
    # Enhanced formatting for displaying results
    hits = response.get("hits", {}).get("hits", [])
    if hits:
        st.write(f"Found {len(hits)} results:")
        for hit in hits:
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})
            st.markdown(
                f"""<div style='border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 8px; background-color: #ffffff;'>
                <h2 style='color: #2c3e50; font-family: Arial, sans-serif;'>{source.get('title', 'No Title')}</h2>
                <p style='font-size: 16px; color: #34495e; line-height: 1.5;'><strong>Text:</strong> {source.get('text', 'No Text')}</p>
                <p style='font-size: 14px; color: #7f8c8d;'><strong>Timestamp:</strong> {metadata.get('timestamp', 'No Timestamp')}</p>
                <p style='font-size: 14px; color: #7f8c8d;'><strong>Indexed On:</strong> {source.get('indexed_on', 'No Indexed Date')}</p>
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.write("No results found.")
else:
    st.write("Please enter a search query.")
