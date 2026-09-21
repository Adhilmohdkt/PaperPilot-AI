import React, { useRef, useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";

const HISTORY_TURNS = 2;
const HISTORY_ANSWER_CHARS = 1000;

export function sourceHref(source) {
  const candidates = [source.url, source.pdf_url, source.paper_url, source.source];
  return candidates.find((value) => typeof value === "string" && /^https?:\/\//.test(value)) || "";
}

export function sourceLabel(source) {
  return source.title || source.source || "Untitled source";
}

export function sourceSnippet(source) {
  return source.text || source.abstract || "";
}

// ─── Backend helpers ─────────────────────────────────────────────────────────

async function apiCreateConversation() {
  const res = await fetch(`${API_URL}/conversations`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to create conversation");
  return (await res.json()).conversation_id;
}

async function apiFetchConversations() {
  const res = await fetch(`${API_URL}/conversations`);
  if (!res.ok) throw new Error("Failed to fetch conversations");
  return (await res.json()).conversations || [];
}

async function apiFetchConversation(convId) {
  const res = await fetch(`${API_URL}/conversations/${convId}`);
  if (!res.ok) throw new Error("Conversation not found");
  return (await res.json()).messages || [];
}

// ─── App ──────────────────────────────────────────────────────────────────────

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  // history is chronological: every turn remains rendered in the active chat.
  const [history, setHistory] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [activeFilter, setActiveFilter] = useState("all");
  const [availableDocs, setAvailableDocs] = useState([]);
  const fileInputRef = useRef(null);
  const conversationEndRef = useRef(null);

  // conversationId = null means "no active conversation yet"
  const [conversationId, setConversationId] = useState(null);
  // conversationList = summaries from GET /conversations (for sidebar)
  const [conversationList, setConversationList] = useState([]);

  // ── Fetch sidebar list on mount ──────────────────────────────────────────
  useEffect(() => {
    apiFetchConversations()
      .then(setConversationList)
      .catch((e) => console.error("Bootstrap error:", e));
  }, []);

  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [history, isLoading]);

  const refreshConversations = async () => {
    try {
      setConversationList(await apiFetchConversations());
    } catch (e) {
      console.error("Could not refresh conversations:", e);
    }
  };

  // ── Load an existing conversation from the sidebar ───────────────────────
  const loadConversation = async (convId) => {
    try {
      const messages = await apiFetchConversation(convId);
      setConversationId(convId);
      // Convert flat [user, assistant, …] into UI { query, answer } pairs
      const msgs = [];
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (msg.role === "user") {
          const next = messages[i + 1];
          msgs.push({
            query: msg.content,
            answer: next && next.role === "assistant" ? next.content : "",
            sources: [],
          });
        }
      }
      setHistory(msgs);
    } catch (e) {
      console.error("Failed to load conversation:", e);
    }
  };

  // ── "New Chat" ───────────────────────────────────────────────────────────
  const startNewChat = async () => {
    try {
      const newId = await apiCreateConversation();
      setConversationId(newId);
      setHistory([]);
      await refreshConversations();
    } catch (e) {
      console.error("Failed to start new chat:", e);
    }
  };

  // ── Send a message ───────────────────────────────────────────────────────
  const handleAsk = async (event) => {
    event?.preventDefault();
    if (!query.trim()) return;

    const currentQuery = query;
    setIsLoading(true);
    setQuery("");
    const turnId = `${Date.now()}-${Math.random()}`;
    const priorHistory = history;
    setHistory((prev) => [...prev, { id: turnId, query: currentQuery, answer: "", sources: [] }]);

    try {
      // Lazily create a conversation the first time a message is sent
      let activeConvId = conversationId;
      if (!activeConvId) {
        activeConvId = await apiCreateConversation();
        setConversationId(activeConvId);
        await refreshConversations();
      }

      const chatContext = priorHistory
        .slice(-HISTORY_TURNS)
        .flatMap((item) => [
          { role: "user", content: item.query },
          { role: "assistant", content: (item.answer || "").slice(0, HISTORY_ANSWER_CHARS) },
        ]);

      const payload = {
        conversation_id: activeConvId,   // ← always the SAME id per chat
        query: currentQuery,
        history: chatContext,
      };
      if (activeFilter !== "all") payload.source_filter = activeFilter;

      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok || !response.body) {
        let errorDetail = "";
        try {
          const errJson = await response.json();
          errorDetail = errJson.detail
            ? typeof errJson.detail === "string"
              ? errJson.detail
              : JSON.stringify(errJson.detail)
            : errJson.message || "";
        } catch {
          try { errorDetail = await response.text(); } catch {}
        }
        throw new Error(
          errorDetail
            ? `Research service error (${response.status}): ${errorDetail}`
            : `Research service is unavailable (HTTP ${response.status}).`
        );
      }

      // ── SSE streaming ──────────────────────────────────────────────────
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let answer = "";
      let sources = [];
      let buffer = "";
      let streamError = null;

      const processEvent = (message) => {
        let name = message.match(/^event: (.+)$/m)?.[1];
        const raw = message.match(/^data: (.+)$/m)?.[1];
        if (!raw) return;
        let data;
        try { data = JSON.parse(raw); } catch { return; }
        if (!name && data?.event) name = data.event;
        if (!name) return;
        if (name === "sources") {
          sources = data.sources || [];
          setHistory((prev) => prev.map((turn) => turn.id === turnId ? { ...turn, sources } : turn));
        }
        if (name === "token") {
          const text = typeof data.text === "string" ? data.text : (data.text?.text || "");
          answer += text;
          setIsLoading(false);
          setHistory((prev) => prev.map((turn) => turn.id === turnId ? { ...turn, answer } : turn));
        }
        if (name === "error") streamError = data.message || "Research workflow failed.";
      };

      while (true) {
        const { done, value } = await reader.read();
        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const msgs = buffer.split("\n\n");
          buffer = msgs.pop();
          msgs.forEach(processEvent);
        }
        if (done) break;
      }

      if (streamError) throw new Error(streamError);

      await refreshConversations();
    } catch (error) {
      console.error(error);
      setHistory((prev) => prev.map((turn) => turn.id === turnId ? {
        ...turn,
        answer: error.message || "Connection failed. Ensure the backend is running.",
      } : turn));
    } finally {
      setIsLoading(false);
    }
  };

  // ── PDF upload ───────────────────────────────────────────────────────────
  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setUploadStatus("Indexing PDF...");
    try {
      const response = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.error || "Upload failed.");
      setUploadStatus(`Indexed ${data.chunks_inserted} semantic chunks`);
      setAvailableDocs((prev) => prev.includes(data.filename) ? prev : [...prev, data.filename]);
      setActiveFilter(data.filename);
      setTimeout(() => setUploadStatus(""), 4000);
    } catch (error) {
      setUploadStatus(error.message || "Failed to upload PDF.");
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Remove a document ────────────────────────────────────────────────────
  const removeActiveDocument = async () => {
    if (activeFilter === "all") return;
    const documentName = activeFilter;
    try {
      const response = await fetch(
        `${API_URL}/documents/${encodeURIComponent(documentName)}`,
        { method: "DELETE" }
      );
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not remove this paper.");
      setAvailableDocs((prev) => prev.filter((item) => item !== documentName));
      setActiveFilter("all");
      setUploadStatus(`Removed ${documentName} and its vectors`);
      setTimeout(() => setUploadStatus(""), 4000);
    } catch (error) {
      setUploadStatus(error.message || "Could not remove this paper.");
    }
  };

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="app-layout">
      <aside className="sidebar glass-panel">
        <h1 className="brand-title">PaperPilot</h1>

        {/* New Chat button */}
        <button
          className="new-chat-btn"
          onClick={startNewChat}
        >
          + New Chat
        </button>

        <div className="upload-section">
          <input
            id="file-upload"
            type="file"
            accept=".pdf"
            ref={fileInputRef}
            onChange={handleUpload}
            style={{ display: "none" }}
          />
          <label htmlFor="file-upload" className="upload-btn">
            {uploadStatus || "Upload PDF"}
          </label>
        </div>

        <div className="section-label">RECENT RESEARCH</div>
        <div className="history-list">
          {conversationList.length ? (
            conversationList.map((conv) => (
              <button
                key={conv.conversation_id}
                className={`history-item${conv.conversation_id === conversationId ? " active" : ""}`}
                onClick={() => loadConversation(conv.conversation_id)}
                title={conv.last_query || "New conversation"}
              >
                <span className="history-item-text">
                  {conv.last_query || "New conversation"}
                </span>
              </button>
            ))
          ) : (
            <div className="empty-history">No recent searches</div>
          )}
        </div>
      </aside>

      <main className="main-workspace">
        <div className="top-dock glass-panel">
          <span>Query focus:</span>
          <select
            className="filter-dropdown"
            value={activeFilter}
            onChange={(e) => setActiveFilter(e.target.value)}
          >
            <option value="all">Search all documents</option>
            {availableDocs.map((doc) => (
              <option key={doc} value={doc}>{doc}</option>
            ))}
          </select>
          {activeFilter !== "all" && (
            <button className="remove-document-btn" type="button" onClick={removeActiveDocument}>
              Remove paper
            </button>
          )}
        </div>

        <section className="content-display">
          {history.length === 0 && !isLoading ? (
            <div className="hero-state">
              <h2 className="hero-title">Unlock Your Documents</h2>
              <p className="hero-subtitle">
                Ask focused questions and get source-backed answers from your papers or live research.
              </p>
            </div>
          ) : (
            <>
              {history.map((turn, turnIndex) => (
                <React.Fragment key={turn.id || `${turn.query}-${turnIndex}`}>
                  <div className="qa-container">
                    <div className="user-query">{turn.query}</div>
                    {isLoading && turnIndex === history.length - 1 && !turn.answer ? (
                      <div className="thinking-indicator">Searching papers and synthesizing evidence...</div>
                    ) : (
                      <div className="ai-answer markdown-body">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeRaw]}
                          components={{
                            a: ({ node, children, ...props }) => (
                              <a {...props} target="_blank" rel="noopener noreferrer">
                                {children}
                              </a>
                            ),
                            table: ({ node, children, ...props }) => (
                              <div className="table-wrapper">
                                <table {...props}>{children}</table>
                              </div>
                            ),
                          }}
                        >
                          {turn.answer}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                  {turn.sources?.length > 0 && !(isLoading && turnIndex === history.length - 1) && (
                    <div className="sources-section">
                  <div className="sources-title">Source materials used</div>
                  <div className="sources-grid">
                    {turn.sources.map((source, index) => {
                      const href = sourceHref(source);
                      const label = sourceLabel(source);
                      const academic = source.kind === "academic" || Boolean(href);
                      return (
                        <div className="source-card" key={index}>
                          <div className="source-actions">
                            {href ? (
                              <a href={href} target="_blank" rel="noreferrer" className="source-badge">
                                {academic ? "Paper" : "Live"}: {label}
                              </a>
                            ) : (
                              <span className="source-badge">
                                PDF: {label}{source.page ? ` · page ${source.page}` : ""}
                              </span>
                            )}
                          </div>
                          <div className="source-text">{sourceSnippet(source)}</div>
                        </div>
                      );
                    })}
                  </div>
                    </div>
                  )}
                </React.Fragment>
              ))}
              <div ref={conversationEndRef} />
            </>
          )}
        </section>

        <div className="input-dock">
          <form className="search-box" onSubmit={handleAsk}>
            <input
              className="search-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={isLoading}
              placeholder="Ask a question about your research"
            />
            <button className="ask-btn" type="submit" disabled={isLoading || !query.trim()}>
              {isLoading ? "Researching..." : "Analyze"}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
