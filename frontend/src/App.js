import React, { useState, useRef } from "react";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [activeSession, setActiveSession] = useState(null); // { query, answer, sources }
  const [history, setHistory] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [activeFilter, setActiveFilter] = useState("all");
  const [availableDocs, setAvailableDocs] = useState([]);
  const fileInputRef = useRef(null);

  const handleAsk = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    const currentQuery = query;
    setIsLoading(true);
    setQuery("");

    // Convert history into a conversational array for the LLM context
    const chatContext = history.slice(0, 4).reverse().flatMap(h => [
      { role: "user", content: h.query },
      { role: "assistant", content: h.answer || "" }
    ]);
    
    // Optimistically set active session so we see the question immediately
    setActiveSession({
      query: currentQuery,
      answer: "",
      sources: []
    });

    try {
      // Stage 1: Hit retrieval engine
      const retrievePayload = { query: currentQuery };
      if (activeFilter !== "all") {
        retrievePayload.source_filter = activeFilter;
      }
      
      const retrieveResponse = await fetch(`${API_URL}/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(retrievePayload)
      });
      const retrieveData = await retrieveResponse.json();
      const chunks = retrieveData.sources || [];
      
      // Update UI with the found citations instantly
      setActiveSession(prev => ({ ...prev, sources: chunks }));
      
      // Stage 2: Stream the synthesized answer
      const streamPayload = {
        query: currentQuery, 
        chunks: chunks,
        history: chatContext
      };
      if (activeFilter !== "all") {
        streamPayload.source_filter = activeFilter;
      }
      
      const streamResponse = await fetch(`${API_URL}/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(streamPayload),
      });

      const reader = streamResponse.body.getReader();
      const decoder = new TextDecoder();
      let answerText = "";

      setIsLoading(false); // We are receiving tokens now, stop the thinking indicator

      while (true) {
        const { done, value } = await reader.read();
        
        if (value) {
          answerText += decoder.decode(value, { stream: true });
          setActiveSession(prev => ({ ...prev, answer: answerText }));
        }

        if (done) {
          // Ensure final text captures properly and add to React history block
          setHistory(prev => [
             { query: currentQuery, answer: answerText, sources: chunks }, 
             ...prev
          ]);
          break;
        }
      }

    } catch (error) {
      console.error("Error:", error);
      setActiveSession({
        query: currentQuery,
        answer: "Connection failed. Ensure the FastAPI backend is running.",
        sources: []
      });
      setIsLoading(false);
    }
  };

  const loadHistoryItem = (item) => {
    setActiveSession(item);
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setUploadStatus("Uploading...");
    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setUploadStatus(`Success: Ingested ${data.chunks_inserted} chunks!`);
        if (!availableDocs.includes(data.filename)) {
          setAvailableDocs(prev => [...prev, data.filename]);
        }
        setActiveFilter(data.filename); // Auto-focus on the new file
        setTimeout(() => setUploadStatus(""), 4000);
      } else {
        setUploadStatus(`Error: ${data.error}`);
      }
    } catch (err) {
      setUploadStatus("Failed to connect to server.");
    }
    // reset file input
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="app-layout">
      
      {/* Sidebar for History / Navigation */}
      <div className="sidebar glass-panel">
        <h1 className="brand-title">PaperPilot</h1>
        
        <div className="upload-section">
          <input 
            type="file" 
            accept=".pdf" 
            ref={fileInputRef} 
            onChange={handleUpload} 
            style={{ display: "none" }} 
            id="file-upload" 
          />
          <label htmlFor="file-upload" className="upload-btn">
            {uploadStatus || "Upload PDF 📄"}
          </label>
        </div>

        <div style={{color: "var(--text-muted)", fontSize: "0.8rem", marginBottom: "15px", letterSpacing: "1px"}}>
          RECENT RESEARCH
        </div>
        
        <div className="history-list">
          {history.length === 0 ? (
            <div style={{color: "rgba(255,255,255,0.2)", fontStyle: "italic", fontSize: "0.9rem"}}>
              No recent searches
            </div>
          ) : (
            history.map((item, idx) => (
              <div 
                key={idx} 
                className="history-item" 
                onClick={() => loadHistoryItem(item)}
              >
                {item.query}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Research Workspace */}
      <div className="main-workspace">
        
        {/* Top Dock: Document Filter */}
        <div className="top-dock glass-panel">
           <span style={{fontSize: "0.85rem", color: "var(--text-muted)", marginRight: "10px"}}>Query Focus:</span>
           <select 
             className="filter-dropdown" 
             value={activeFilter} 
             onChange={(e) => setActiveFilter(e.target.value)}
           >
             <option value="all">🌐 Search All Documents</option>
             {availableDocs.map((doc, idx) => (
               <option key={idx} value={doc}>📄 {doc}</option>
             ))}
           </select>
        </div>

        <div className="content-display">
          {!activeSession && !isLoading ? (
            <div className="hero-state">
              <h2 className="hero-title">Unlock Your Documents</h2>
              <p className="hero-subtitle">
                Ask profound questions, discover hidden insights, and instantly review source citations drawn directly from your knowledge base.
              </p>
            </div>
          ) : (
            <>
              {/* Question & Answer Area */}
              <div className="qa-container">
                <div className="user-query">{activeSession?.query}</div>
                
                {isLoading && !activeSession?.answer ? (
                  <div className="thinking-indicator">
                    <span>Synthesizing intelligence</span>
                    <div className="dot"></div>
                    <div className="dot"></div>
                    <div className="dot"></div>
                  </div>
                ) : (
                  <div className="ai-answer">
                    {activeSession?.answer}
                  </div>
                )}
              </div>

              {/* Source Materials Grid */}
              {activeSession?.sources && activeSession.sources.length > 0 && !isLoading && (
                <div className="sources-section">
                  <div className="sources-title">Source Materials Used</div>
                  <div className="sources-grid">
                    {activeSession.sources.map((src, index) => {
                      const pdfUrl = src.source ? `${API_URL}/pdfs/${encodeURIComponent(src.source)}` : "#";
                      return (
                        <div className="source-card" key={index}>
                          <div className="source-actions">
                            <a 
                              href={pdfUrl} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="source-badge"
                            >
                              <span role="img" aria-label="file">📄</span> {src.source || "Unknown Source"}
                            </a>
                            {src.source && (
                              <a 
                                href={pdfUrl} 
                                download={src.source}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="download-btn"
                                title="Download PDF"
                              >
                                📥 Download
                              </a>
                            )}
                          </div>
                          <div className="source-text">{src.text || src}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Persistent Input Bar */}
        <div className="input-dock">
          <form className="search-box" onSubmit={handleAsk}>
            <input
              type="text"
              className="search-input"
              placeholder="Ask your query (e.g., 'What is RAG?')"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isLoading}
            />
            <button type="submit" className="ask-btn" disabled={isLoading || !query.trim()}>
              {isLoading ? "Searching..." : "Analyze"}
            </button>
          </form>
        </div>
      </div>
      
    </div>
  );
}

export default App;