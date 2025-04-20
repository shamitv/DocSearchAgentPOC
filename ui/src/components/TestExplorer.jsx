import { useState, useEffect } from 'react';
import './TestExplorer.css';

const API_BASE_URL = 'http://localhost:5001/api';

const TestExplorer = () => {
  const [testSuites, setTestSuites] = useState([]);
  const [selectedSuite, setSelectedSuite] = useState(null);
  const [selectedTest, setSelectedTest] = useState(null);
  const [testResults, setTestResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTestSuites();
  }, []);

  const fetchTestSuites = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/tests`);
      if (!response.ok) {
        throw new Error(`Failed to fetch tests: ${response.statusText}`);
      }
      const data = await response.json();
      setTestSuites(data);
    } catch (err) {
      setError(`Failed to fetch test suites: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const runTest = async (suiteId, testId) => {
    setLoading(true);
    setError(null);
    try {
      // Find the actual suite name from the ID (since the API needs the real class name)
      const suite = testSuites.find(s => s.id === suiteId);
      if (!suite) {
        throw new Error(`Suite ${suiteId} not found`);
      }
      
      const response = await fetch(`${API_BASE_URL}/tests/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          suite: suite.name, // Use actual class name for API call
          test: testId
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to run test: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Parse the test result and format it for display
      const status = result.success ? 'passed' : 'failed';
      const duration = '0.000'; // The API doesn't return duration currently
      const output = result.output || 'No output available';
      
      setTestResults(prev => ({
        ...prev,
        [suiteId]: {
          ...(prev[suiteId] || {}),
          [testId]: { 
            status, 
            duration, 
            output 
          }
        }
      }));
      
    } catch (err) {
      setError(`Failed to run test: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };
  
  const runAllTestsInSuite = async (suiteId) => {
    setLoading(true);
    setError(null);
    try {
      const suite = testSuites.find(s => s.id === suiteId);
      if (!suite) return;
      
      // Run each test in sequence
      for (const test of suite.tests) {
        await runTest(suiteId, test.id);
      }
    } catch (err) {
      setError(`Failed to run all tests in suite: ${err.message}`);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getTestStatus = (suiteId, testId) => {
    if (!testResults[suiteId] || !testResults[suiteId][testId]) {
      return null;
    }
    return testResults[suiteId][testId].status;
  };

  const getSuiteStatus = (suiteId) => {
    if (!testResults[suiteId]) return null;
    
    const tests = Object.values(testResults[suiteId]);
    if (tests.length === 0) return null;
    
    const allPassed = tests.every(test => test.status === 'passed');
    const anyFailed = tests.some(test => test.status === 'failed');
    
    if (anyFailed) return 'failed';
    if (allPassed) return 'passed';
    return 'partial';
  };

  return (
    <div className="test-explorer">
      <h1>Test Explorer</h1>
      
      {error && <div className="error-message">{error}</div>}
      
      <div className="test-container">
        <div className="test-suites">
          <h2>Test Suites</h2>
          {testSuites.length === 0 && !loading && !error && (
            <p className="no-data-message">No test suites found. Make sure the test API is running.</p>
          )}
          {testSuites.map(suite => (
            <div 
              key={suite.id} 
              className={`test-suite ${selectedSuite === suite.id ? 'selected' : ''} ${getSuiteStatus(suite.id) ? `status-${getSuiteStatus(suite.id)}` : ''}`}
              onClick={() => setSelectedSuite(suite.id)}
            >
              <h3>{suite.name}</h3>
              <p>{suite.description}</p>
              <span className="test-count">{suite.tests.length} tests</span>
              <button 
                className="run-all-button" 
                onClick={(e) => {
                  e.stopPropagation();
                  runAllTestsInSuite(suite.id);
                }}
                disabled={loading}
              >
                Run All
              </button>
            </div>
          ))}
        </div>
        
        {selectedSuite && (
          <div className="test-details">
            <h2>Tests in {testSuites.find(s => s.id === selectedSuite)?.name}</h2>
            <div className="test-list">
              {testSuites.find(s => s.id === selectedSuite)?.tests.map(test => (
                <div 
                  key={test.id} 
                  className={`test-item ${selectedTest === test.id ? 'selected' : ''} ${getTestStatus(selectedSuite, test.id) ? `status-${getTestStatus(selectedSuite, test.id)}` : ''}`}
                  onClick={() => setSelectedTest(test.id)}
                >
                  <h4>{test.name}</h4>
                  <p>{test.description}</p>
                  {testResults[selectedSuite]?.[test.id] && (
                    <div className="test-result-summary">
                      <span className={`status status-${testResults[selectedSuite][test.id].status}`}>
                        {testResults[selectedSuite][test.id].status.toUpperCase()}
                      </span>
                      <span className="duration">{testResults[selectedSuite][test.id].duration}s</span>
                    </div>
                  )}
                  <button 
                    className="run-button" 
                    onClick={(e) => {
                      e.stopPropagation();
                      runTest(selectedSuite, test.id);
                    }}
                    disabled={loading}
                  >
                    Run
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {selectedSuite && selectedTest && testResults[selectedSuite]?.[selectedTest] && (
          <div className="test-result">
            <h2>Test Result</h2>
            <div className={`result-card status-${testResults[selectedSuite][selectedTest].status}`}>
              <h3>
                {testSuites.find(s => s.id === selectedSuite)?.tests.find(t => t.id === selectedTest)?.name}
              </h3>
              <div className="result-header">
                <span className={`status status-${testResults[selectedSuite][selectedTest].status}`}>
                  {testResults[selectedSuite][selectedTest].status.toUpperCase()}
                </span>
                <span className="duration">Duration: {testResults[selectedSuite][selectedTest].duration}s</span>
              </div>
              <pre className="output">{testResults[selectedSuite][selectedTest].output}</pre>
            </div>
          </div>
        )}
      </div>
      
      {loading && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>Running tests...</p>
        </div>
      )}
    </div>
  );
};

export default TestExplorer;