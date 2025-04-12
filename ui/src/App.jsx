import React from "react";
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import {
  SearchProvider,
  SearchBox,
  Results,
  Paging
} from "@elastic/react-search-ui";
import "@elastic/react-search-ui-views/lib/styles/styles.css";

const onSearch = async (searchTerm) => {
  const isDevelopment = import.meta.env.MODE === "development" || process.env.NODE_ENV === "development";

  if (isDevelopment) {
    console.log("Search term received:", searchTerm); // Debugging log
  }

  // Extract the search term if it's nested in an object
  const query = typeof searchTerm === "string" ? searchTerm : searchTerm.searchTerm || "";

  // Update the API request to avoid appending /search
  const response = await fetch(`/api?q=${encodeURIComponent(query)}`);
  const results = await response.json();

  if (isDevelopment) {
    console.log("API response:", results); // Log the API response for debugging
  }

  // Map the API response to the format expected by react-search-ui
  return {
    results: results.hits.hits.map((hit) => ({
      id: hit._id, // Use the unique _id field from the API response
      title: { raw: hit._source.title || "No Title" }, // Fallback for missing title
      description: { raw: hit._source.text || "No Description" }, // Fallback for missing description
    })),
  };
};

const config = {
  onSearch,
};

const App = () => {
  const [count, setCount] = useState(0)

  var searchApiUrl = import.meta.env.VITE_SEARCH_API_URL || "http://i3tiny1.local:7020/wikipedia/_search";
  if (import.meta.env.MODE === "development") {
    console.log("Development mode detected");
    console.log("Search API URL:", searchApiUrl);
  }
  return (
    <SearchProvider config={config}>
      <div style={{ backgroundColor: "#F1F5F9", color: "#0F172A", padding: "20px" }}>
        <h1 style={{ color: "#0EA5E9" }}>Search UI</h1>
        <SearchBox
          inputView={({ getInputProps }) => (
            <input
              {...getInputProps({
                placeholder: "Search...",
                style: {
                  padding: "10px",
                  border: "1px solid #0EA5E9",
                  borderRadius: "5px",
                  width: "100%",
                },
              })}
            />
          )}
        />
        <Results
          resultView={({ result }) => (
            <div
              key={result.id?.raw || `result-${Math.random()}`} // Ensure a unique fallback key
              style={{
                border: "1px solid #0EA5E9",
                borderRadius: "5px",
                padding: "10px",
                margin: "10px 0",
              }}
            >
              <h2>{result.title.raw}</h2>
              <p>{result.description.raw}</p>
            </div>
          )}
        />
        <Paging />
      </div>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.jsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </SearchProvider>
  )
}

export default App
