import React, { useState, useEffect } from "react";
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import TestExplorer from './components/TestExplorer';
import {
  SearchProvider,
  SearchBox,
  Results,
  Paging
} from "@elastic/react-search-ui";
import "@elastic/react-search-ui-views/lib/styles/styles.css";
import { marked } from "marked";

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
  const [count, setCount] = useState(0);
  const [sampleQuestion, setSampleQuestion] = useState("");  // State for sample input
  const [generatedQueries, setGeneratedQueries] = useState([]);  // State for generated queries
  const [selectedQuery, setSelectedQuery] = useState(null);  // State for clicked query
  const [selectedResults, setSelectedResults] = useState([]);  // State for results of clicked query
  // State for advanced agent interaction
  const [agentQuestion, setAgentQuestion] = useState("");
  const [maxIterations, setMaxIterations] = useState(5);
  const [agentResponse, setAgentResponse] = useState(null);
  const [agentLoading, setAgentLoading] = useState(false);

  const [activeTab, setActiveTab] = useState('search');  // Default to Search UI tab

  var searchApiUrl = import.meta.env.VITE_SEARCH_API_URL || "http://i3tiny1.local:7020/wikipedia/_search";
  const agentApiUrl = import.meta.env.VITE_AGENT_API_URL || "http://localhost:8000";  // URL for agent API (FastAPI)
  if (import.meta.env.MODE === "development") {
    console.log("Development mode detected");
    console.log("Search API URL:", searchApiUrl);
    console.log("Agent API URL:", agentApiUrl);
  }

  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [runDetails, setRunDetails] = useState({});

  useEffect(() => {
    if (activeTab === 'history') {
      fetch('/api/runs')
        .then(res => res.json())
        .then(data => setRuns(Array.isArray(data) ? data : []))
        .catch(err => {
          console.error('Error fetching runs:', err);
          setRuns([]);
        });
    }
  }, [activeTab]);

  useEffect(() => {
    if (selectedRunId) {
      fetch(`/api/runs/${selectedRunId}`)
        .then(res => res.json())
        .then(data => setRunDetails(data.details))
        .catch(err => console.error('Error fetching run details:', err));
    }
  }, [selectedRunId]);

  return (
    <div className="app-container">
      <div className="app-header">
        <h1>DocSearch Agent POC</h1>
        <div className="app-tabs">
          <button 
            className={`tab-button ${activeTab === 'testExplorer' ? 'active' : ''}`}
            onClick={() => setActiveTab('testExplorer')}
          >
            Test Explorer
          </button>
          <button 
            className={`tab-button ${activeTab === 'search' ? 'active' : ''}`}
            onClick={() => setActiveTab('search')}
          >
            Search UI
          </button>
          <button 
            className={`tab-button ${activeTab === 'agent' ? 'active' : ''}`}
            onClick={() => setActiveTab('agent')}
          >
            Agent
          </button>
          <button className={`tab-button ${activeTab === 'history' ? 'active' : ''}`} onClick={() => setActiveTab('history')}>
            History
          </button>
        </div>
      </div>

      {activeTab === 'testExplorer' && (
        <TestExplorer />
      )}

      {activeTab === 'search' && (
        <SearchProvider config={config}>
          <div style={{ backgroundColor: "#F1F5F9", color: "#0F172A", padding: "20px" }}>
            {/* Section to generate search queries */}
            <div style={{ marginBottom: "20px" }}>
              <input
                type="text"
                placeholder="Enter sample question"
                value={sampleQuestion}
                onChange={(e) => setSampleQuestion(e.target.value)}
                style={{ padding: "8px", width: "70%", marginRight: "10px", borderRadius: "4px", border: "1px solid #ccc" }}
              />
              <button
                onClick={async () => {
                  if (!sampleQuestion) return;
                  try {
                    const response = await fetch(`${agentApiUrl}/api/queries?question=${encodeURIComponent(sampleQuestion)}`);
                    const data = await response.json();
                    setGeneratedQueries(data.queries || []);
                  } catch (err) {
                    console.error("Error generating queries:", err);
                  }
                }}
                style={{ padding: "8px 16px", backgroundColor: "#0EA5E9", color: "#fff", border: "none", borderRadius: "4px", cursor: "pointer" }}
              >
                Generate Queries
              </button>
            </div>
            {/* Display generated queries */}
            {generatedQueries.length > 0 && (
              <div style={{ marginBottom: "20px" }}>
                <h3 style={{ color: "#0EA5E9" }}>Generated Queries:</h3>
                <ul>
                  {generatedQueries.map((q, i) => (
                    <li key={i} style={{ marginBottom: "8px" }}>
                      <a
                        href="#"
                        onClick={async (e) => {
                          e.preventDefault();
                          setSelectedQuery(q);
                          try {
                            const resp = await fetch(
                              `${agentApiUrl}/api/search`,
                              {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ query: q, max_results: 5 })
                              }
                            );
                            const data = await resp.json();
                            setSelectedResults(data.results || []);
                          } catch (err) {
                            console.error('Error fetching search results:', err);
                          }
                        }}
                        style={{ color: '#0EA5E9', textDecoration: 'underline', cursor: 'pointer' }}
                      >
                        {q}
                      </a>
                    </li>
                  ))}
                </ul>

                {/* Display clicked query results */}
                {selectedResults.length > 0 && (
                  <div style={{ marginTop: '20px' }}>
                    <h3 style={{ color: '#0EA5E9' }}>Results for: {selectedQuery}</h3>
                    {selectedResults.map((res, idx) => (
                      <div
                        key={idx}
                        style={{
                          border: '1px solid #0EA5E9',
                          borderRadius: '5px',
                          padding: '10px',
                          margin: '10px 0'
                        }}
                      >
                        <h4 style={{ margin: 0 }}>{res.title}</h4>
                        <p style={{ margin: '5px 0' }}>{res.content}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
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
        </SearchProvider>
      )}
      {activeTab === 'agent' && (
        <div style={{ padding: '20px', backgroundColor: '#f9fafb', color: '#111827' }}>
          <h2>Advanced Knowledge Agent</h2>
          <div style={{ marginBottom: '10px' }}>
            <input
              type="text"
              placeholder="Enter your question"
              value={agentQuestion}
              onChange={(e) => setAgentQuestion(e.target.value)}
              style={{ padding: '8px', width: '60%', marginRight: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
            <input
              type="number"
              placeholder="Max iterations"
              value={maxIterations}
              onChange={(e) => setMaxIterations(Number(e.target.value))}
              style={{ padding: '8px', width: '100px', marginRight: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
            />
            <button
              onClick={async () => {
                if (!agentQuestion) return;
                setAgentLoading(true);
                try {
                  const resp = await fetch(
                    `${agentApiUrl}/api/agent`,
                    {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ question: agentQuestion, max_iterations: maxIterations })
                    }
                  );
                  const data = await resp.json();
                  // If the backend returns {search_results, agent_response}, merge them for display
                  if (data && data.search_results && data.agent_response) {
                    setAgentResponse({
                      ...data.search_results,
                      agent_response: data.agent_response
                    });
                  } else {
                    setAgentResponse(data);
                  }
                } catch (err) {
                  console.error('Error calling agent:', err);
                  setAgentResponse({ error: err.toString() });
                } finally {
                  setAgentLoading(false);
                }
              }}
              style={{ padding: '8px 16px', backgroundColor: '#10b981', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
            >
              {agentLoading ? 'Running...' : 'Run Agent'}
            </button>
          </div>
          {agentResponse && (
            <div style={{ marginTop: '20px', backgroundColor: '#fff', padding: '20px', borderRadius: '6px', border: '1px solid #e5e7eb' }}>
              <h3>Agent Response</h3>
              {agentResponse.error ? (
                <div style={{ color: 'red' }}>{agentResponse.error}</div>
              ) : (
                <>
                  <p><strong>Answer Found:</strong> {agentResponse.answer_found ? 'Yes' : 'No'}</p>
                  
                  <div style={{ marginTop: '20px' }}>
                          <h4>Summary</h4>
                          <div
                            style={{ background: '#f3f4f6', padding: '12px', borderRadius: '4px' }}
                            dangerouslySetInnerHTML={{
                              __html: marked.parse(
                                String(agentResponse.agent_response ?? "")
                              )
                            }}
                          />
                        </div>
                  <p><strong>Iterations:</strong> {agentResponse.iterations}</p>
                  <p><strong>Processing Time:</strong> {agentResponse.processing_time?.toFixed(2)} seconds</p>
                  {agentResponse.answer_found && agentResponse.final_answer && (
                    <div style={{ marginTop: '10px' }}>
                      <h4>Answer</h4>
                      <p>{agentResponse.final_answer.answer}</p>
                      <p><strong>Confidence:</strong> {agentResponse.final_answer.confidence}</p>
                      {agentResponse.final_answer.supporting_evidence && agentResponse.final_answer.supporting_evidence.length > 0 && (
                        <div>
                          <h5>Supporting Evidence</h5>
                          <ul>
                            {agentResponse.final_answer.supporting_evidence.map((ev, idx) => (
                              <li key={idx} style={{ marginBottom: '8px' }}>
                                <strong>{ev.title}</strong>: {ev.content.slice(0, 100)}...
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <h5>Reasoning</h5>
                      <p>{agentResponse.final_answer.reasoning}</p>
                      {/* Render markdown summary if present */}
                      {agentResponse.final_answer.answer && (
                        <div style={{ marginTop: '20px' }}>
                          <h4>Summary</h4>
                          <div
                            style={{ background: '#f3f4f6', padding: '12px', borderRadius: '4px' }}
                            dangerouslySetInnerHTML={{
                              __html: marked.parse(
                                Array.isArray(agentResponse.final_answer.answer)
                                  ? agentResponse.final_answer.answer.join('\n')
                                  : String(agentResponse.final_answer.answer ?? "")
                              )
                            }}
                          />
                        </div>
                      )}
                    </div>
                  )}
                  {/* Show the raw agent LLM response if available */}
                  {agentResponse.agent_response && (
                    <details style={{ marginTop: '20px' }}>
                      <summary>Show Raw Agent LLM Response</summary>
                      <pre style={{ whiteSpace: 'pre-wrap', marginTop: '10px', backgroundColor: '#f3f4f6', padding: '10px', borderRadius: '4px' }}>
                        {typeof agentResponse.agent_response === 'string' ? agentResponse.agent_response : JSON.stringify(agentResponse.agent_response, null, 2)}
                      </pre>
                    </details>
                  )}
                  {/* Collapsible raw JSON for advanced users */}
                  <details style={{ marginTop: '20px' }}>
                    <summary>Show Raw JSON</summary>
                    <pre style={{ whiteSpace: 'pre-wrap', marginTop: '10px', backgroundColor: '#f3f4f6', padding: '10px', borderRadius: '4px' }}>
                      {JSON.stringify(agentResponse, null, 2)}
                    </pre>
                  </details>
                </>
              )}
            </div>
          )}
        </div>
      )}
      {activeTab === 'history' && (
        <div style={{ display: 'flex', padding: '20px' }}>
          <div style={{ width: '30%', borderRight: '1px solid #ccc', paddingRight: '10px' }}>
            <h2>Runs</h2>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              {runs.map(r => (
                <li key={r.id} style={{ marginBottom: '10px', cursor: 'pointer', color: selectedRunId===r.id? '#0EA5E9':'#111' }} onClick={() => setSelectedRunId(r.id)}>
                  {new Date(r.start_time).toLocaleString()} - {r.question}
                </li>
              ))}
            </ul>
          </div>
          <div style={{ width: '65%', paddingLeft: '10px' }}>
            <h2>Details</h2>
            {selectedRunId ? (
              <>
                {Object.entries(runDetails).map(([table, entries]) => {
                  // Special rendering for LLM metrics tables
                  if (table === 'query_llm_metrics' || table === 'analysis_llm_metrics') {
                    return (
                      <div key={table}>
                        <h3>{table.replace(/_/g, ' ').toUpperCase()}</h3>
                        {entries.length === 0 ? <p>No entries</p> : entries.map((entry, idx) => (
                          <div key={idx} style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: '6px', marginBottom: '16px', padding: '10px' }}>
                            <div><strong>Timestamp:</strong> {entry.timestamp}</div>
                            <div><strong>Model:</strong> {entry.model_name}</div>
                            <div><strong>Execution Time:</strong> {entry.execution_time_seconds?.toFixed(2)}s</div>
                            <div><strong>Prompt Tokens:</strong> {entry.prompt_tokens}</div>
                            <div><strong>Completion Tokens:</strong> {entry.completion_tokens}</div>
                            <div><strong>Total Tokens:</strong> {entry.total_tokens}</div>
                            {entry.result_index !== undefined && <div><strong>Result Index:</strong> {entry.result_index}</div>}
                            <details style={{ marginTop: '8px' }}>
                              <summary>Prompt</summary>
                              <pre style={{ background: '#f3f4f6', padding: '8px', borderRadius: '4px', whiteSpace: 'pre-wrap' }}>{entry.raw_prompt}</pre>
                            </details>
                            <details style={{ marginTop: '8px' }}>
                              <summary>LLM Response</summary>
                              <pre style={{ background: '#f3f4f6', padding: '8px', borderRadius: '4px', whiteSpace: 'pre-wrap' }}>{entry.raw_content}</pre>
                            </details>
                            {entry.error_message && <div style={{ color: 'red' }}><strong>Error:</strong> {entry.error_message}</div>}
                          </div>
                        ))}
                      </div>
                    );
                  }
                  // Default rendering for other tables
                  return (
                    <div key={table}>
                      <h3>{table}</h3>
                      <pre style={{ background: '#f4f4f4', padding: '10px' }}>{JSON.stringify(entries, null, 2)}</pre>
                    </div>
                  );
                })}
              </>
            ) : (
              <p>Select a run to view details</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
