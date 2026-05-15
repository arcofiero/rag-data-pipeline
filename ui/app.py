"""
RAG Data Pipeline — Streamlit UI

Professional query interface. Calls the FastAPI /query and /health endpoints.
Run with: streamlit run ui/app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")
MAX_HISTORY  = 10

# ─── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Data Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Palette & tokens ──────────────────────────────────── */
:root {
    --primary:        #4F46E5;
    --primary-light:  #818CF8;
    --primary-muted:  #EEF2FF;
    --success:        #059669;
    --success-bg:     #D1FAE5;
    --danger:         #DC2626;
    --danger-bg:      #FEE2E2;
    --surface:        #FFFFFF;
    --surface-alt:    #F9FAFB;
    --border:         #E5E7EB;
    --text-primary:   #111827;
    --text-secondary: #6B7280;
    --text-muted:     #9CA3AF;

    --badge-pdf-bg:  #EFF6FF; --badge-pdf-fg:  #1D4ED8;
    --badge-web-bg:  #ECFDF5; --badge-web-fg:  #065F46;
    --badge-str-bg:  #FFF7ED; --badge-str-fg:  #C2410C;
    --badge-unk-bg:  #F3F4F6; --badge-unk-fg:  #374151;

    --radius:    8px;
    --radius-lg: 12px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.05);
    --shadow:    0 1px 3px rgba(0,0,0,.10), 0 1px 2px -1px rgba(0,0,0,.08);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,.08), 0 2px 4px -2px rgba(0,0,0,.06);
}

/* ── Hide Streamlit chrome ─────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton           { display: none; }

/* ── Global layout ─────────────────────────────────────── */
.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ── App header ────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.app-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.025em;
    margin: 0;
}
.app-subtitle {
    color: var(--text-secondary);
    font-size: 0.875rem;
    margin: 0.3rem 0 0;
}
.api-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.3rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}
.api-online  { background: var(--success-bg); color: var(--success); }
.api-offline { background: var(--danger-bg);  color: var(--danger);  }

/* ── Answer card ───────────────────────────────────────── */
.answer-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--primary);
    border-radius: var(--radius-lg);
    padding: 1.75rem 2rem;
    box-shadow: var(--shadow-md);
    margin-bottom: 1.25rem;
}
.answer-label {
    text-transform: uppercase;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: var(--primary);
    margin: 0 0 0.875rem;
}
.answer-text {
    font-size: 0.975rem;
    line-height: 1.8;
    color: var(--text-primary);
    margin: 0;
    white-space: pre-wrap;
}

/* ── Metadata pills ────────────────────────────────────── */
.meta-row {
    display: flex;
    gap: 0.625rem;
    flex-wrap: wrap;
    margin-bottom: 1.75rem;
}
.meta-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 9999px;
    padding: 0.275rem 0.75rem;
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Section header ────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin: 0;
}
.section-count {
    background: var(--border);
    color: var(--text-secondary);
    padding: 0.1rem 0.45rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
}

/* ── Source card ───────────────────────────────────────── */
.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem;
    box-shadow: var(--shadow-sm);
    height: 100%;
    transition: box-shadow 0.15s ease, border-color 0.15s ease;
    margin-bottom: 1rem;
}
.source-card:hover {
    box-shadow: var(--shadow-md);
    border-color: var(--primary-light);
}
.source-card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.75rem;
}
.source-badge {
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-pdf { background: var(--badge-pdf-bg); color: var(--badge-pdf-fg); }
.badge-web { background: var(--badge-web-bg); color: var(--badge-web-fg); }
.badge-structured { background: var(--badge-str-bg); color: var(--badge-str-fg); }
.badge-unknown    { background: var(--badge-unk-bg); color: var(--badge-unk-fg); }

.score-value {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ── Score bar ─────────────────────────────────────────── */
.score-bar-wrap { margin-bottom: 1rem; }
.score-bar-track {
    background: var(--border);
    border-radius: 9999px;
    height: 5px;
    overflow: hidden;
    margin-top: 0.25rem;
}
.score-bar-fill {
    height: 100%;
    border-radius: 9999px;
    background: linear-gradient(90deg, var(--primary), var(--primary-light));
}

/* ── Source metadata table ─────────────────────────────── */
.src-meta { margin-bottom: 0.875rem; }
.src-meta-row {
    display: flex;
    gap: 0.375rem;
    font-size: 0.775rem;
    margin-bottom: 0.3rem;
    align-items: flex-start;
}
.src-meta-key {
    font-weight: 600;
    color: var(--text-muted);
    min-width: 4.5rem;
    flex-shrink: 0;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding-top: 1px;
}
.src-meta-val {
    font-family: "SF Mono", "Menlo", "Consolas", monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
    word-break: break-all;
}

/* ── Content preview ───────────────────────────────────── */
.src-preview {
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem;
    font-size: 0.8rem;
    line-height: 1.65;
    color: var(--text-secondary);
    max-height: 110px;
    overflow-y: auto;
    font-style: italic;
}

/* ── Empty state ───────────────────────────────────────── */
.empty-wrap {
    text-align: center;
    padding: 5rem 2rem 3rem;
}
.empty-icon  { font-size: 3.5rem; opacity: 0.25; margin-bottom: 1rem; }
.empty-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.empty-body  {
    font-size: 0.875rem;
    color: var(--text-muted);
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Sidebar ───────────────────────────────────────────── */
.sb-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.015em;
    margin: 0 0 0.25rem;
}
.sb-tagline {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0 0 1rem;
}
.sb-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin: 0 0 0.6rem;
}
.health-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.83rem;
}
.health-row:last-child { border-bottom: none; }
.health-name { color: var(--text-primary); font-weight: 500; }
.health-ok   { color: var(--success); font-weight: 600; font-size: 0.8rem; }
.health-fail { color: var(--danger);  font-weight: 600; font-size: 0.8rem; }

.stack-item {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    font-size: 0.8rem;
    border-bottom: 1px solid var(--border);
}
.stack-item:last-child { border-bottom: none; }
.stack-key { color: var(--text-muted); font-size: 0.75rem; }
.stack-val { color: var(--text-secondary); font-weight: 500; font-size: 0.78rem; }

/* ── Streamlit widget overrides ────────────────────────── */
.stButton > button {
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.15s ease !important;
}
.stTextArea textarea {
    border-radius: var(--radius) !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    border: 1.5px solid var(--border) !important;
    padding: 0.875rem 1rem !important;
    transition: border-color 0.15s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-muted) !important;
}
div[data-testid="stForm"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem !important;
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.75rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "prefill_query" not in st.session_state:
    st.session_state.prefill_query = ""
if "elapsed" not in st.session_state:
    st.session_state.elapsed = 0.0


# ─── API helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=20, show_spinner=False)
def _health(api_url: str) -> dict:
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "offline", "pinecone_ready": False, "openai_key_set": False}


def _query(api_url: str, query: str, top_k: int, filter_source: Optional[str]) -> dict:
    payload: dict = {"query": query, "top_k": top_k}
    if filter_source and filter_source != "All":
        payload["filter_source"] = filter_source
    r = requests.post(f"{api_url}/query", json=payload, timeout=45)
    r.raise_for_status()
    return r.json()


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _badge_class(source_type: str) -> str:
    return {
        "PDF":        "badge-pdf",
        "WEB":        "badge-web",
        "STRUCTURED": "badge-structured",
    }.get(source_type.upper(), "badge-unknown")


# ─── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sb-brand">⚡ RAG Data Pipeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="sb-tagline">Semantic search · Grounded answers</p>', unsafe_allow_html=True)

    st.divider()

    # ── Pipeline health ──────────────────────────────
    st.markdown('<p class="sb-section-label">Pipeline Status</p>', unsafe_allow_html=True)
    health   = _health(API_BASE_URL)
    api_up   = health.get("status") not in ("offline", None)
    pc_ready = health.get("pinecone_ready", False)

    st.markdown(f"""
    <div class="health-row">
        <span class="health-name">API Server</span>
        <span class="{'health-ok' if api_up else 'health-fail'}">
            {'● Online' if api_up else '● Offline'}
        </span>
    </div>
    <div class="health-row">
        <span class="health-name">Pinecone</span>
        <span class="{'health-ok' if pc_ready else 'health-fail'}">
            {'● Connected' if pc_ready else '● Disconnected'}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Search settings ──────────────────────────────
    st.markdown('<p class="sb-section-label">Search Settings</p>', unsafe_allow_html=True)
    top_k = st.slider("Results to retrieve", min_value=1, max_value=20, value=5)
    filter_source = st.selectbox(
        "Filter by source type",
        ["All", "PDF", "WEB", "STRUCTURED"],
        index=0,
    )

    st.divider()

    # ── Query history ────────────────────────────────
    st.markdown('<p class="sb-section-label">Recent Queries</p>', unsafe_allow_html=True)
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history[-MAX_HISTORY:])):
            label = entry["query"]
            short = (label[:38] + "…") if len(label) > 38 else label
            if st.button(f"↩  {short}", key=f"hist_{i}", use_container_width=True):
                st.session_state.prefill_query = label
                st.session_state.result = None
                st.rerun()
        st.markdown("")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history.clear()
            st.rerun()
    else:
        st.caption("No queries yet — run a search to see history here.")

    st.divider()

    # ── Stack info ───────────────────────────────────
    st.markdown('<p class="sb-section-label">Stack</p>', unsafe_allow_html=True)
    chat_provider = os.getenv("CHAT_PROVIDER", "openai").capitalize()
    embed_provider = os.getenv("EMBEDDING_PROVIDER", "openai").capitalize()
    chat_model = os.getenv("GEMINI_CHAT_MODEL", os.getenv("OPENAI_CHAT_MODEL", "—"))

    st.markdown(f"""
    <div class="stack-item">
        <span class="stack-key">Embedding</span>
        <span class="stack-val">{embed_provider}</span>
    </div>
    <div class="stack-item">
        <span class="stack-key">LLM</span>
        <span class="stack-val">{chat_provider}</span>
    </div>
    <div class="stack-item">
        <span class="stack-key">Model</span>
        <span class="stack-val">{chat_model.replace("models/", "")}</span>
    </div>
    <div class="stack-item">
        <span class="stack-key">Vector store</span>
        <span class="stack-val">Pinecone</span>
    </div>
    <div class="stack-item">
        <span class="stack-key">Pipeline</span>
        <span class="stack-val">PySpark + Delta</span>
    </div>
    <div class="stack-item">
        <span class="stack-key">API</span>
        <span class="stack-val">{API_BASE_URL}</span>
    </div>
    """, unsafe_allow_html=True)


# ─── Main content ─────────────────────────────────────────────────────────────────

# ── App header with live status badge ────────────────────
api_badge_class = "api-online" if api_up else "api-offline"
api_badge_text  = "● API Online" if api_up else "● API Offline"

st.markdown(f"""
<div class="app-header">
    <div>
        <h1 class="app-title">⚡ RAG Data Pipeline</h1>
        <p class="app-subtitle">
            Semantic search across your ingested knowledge base —
            retrieves the most relevant chunks and generates a grounded answer.
        </p>
    </div>
    <span class="api-badge {api_badge_class}">{api_badge_text}</span>
</div>
""", unsafe_allow_html=True)

# ── Query form ────────────────────────────────────────────
with st.form("query_form", border=False):
    query_input = st.text_area(
        label="query",
        value=st.session_state.prefill_query,
        placeholder="Ask a question about your documents… e.g. 'What is the customer data retention policy?'",
        height=110,
        label_visibility="collapsed",
    )
    _, _, submit_col = st.columns([4, 1, 1])
    with submit_col:
        submitted = st.form_submit_button(
            "🔍  Search",
            use_container_width=True,
            type="primary",
        )

# ── Handle submission ─────────────────────────────────────
if submitted and query_input.strip():
    clean_query = query_input.strip()
    st.session_state.prefill_query = clean_query
    with st.spinner("Retrieving chunks and generating answer…"):
        t0 = time.perf_counter()
        try:
            result = _query(API_BASE_URL, clean_query, top_k, filter_source)
            st.session_state.elapsed = time.perf_counter() - t0
            st.session_state.result  = result
            if not st.session_state.history or st.session_state.history[-1]["query"] != clean_query:
                st.session_state.history.append({
                    "query":     clean_query,
                    "timestamp": datetime.now().strftime("%H:%M"),
                })
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach the API server at **{API_BASE_URL}**. "
                     f"Start it with: `uvicorn api.rag_endpoint:app --port 8000`")
            st.session_state.result = None
        except requests.exceptions.HTTPError as exc:
            st.error(f"API returned an error: **{exc}**")
            st.session_state.result = None
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.session_state.result = None

elif submitted and not query_input.strip():
    st.warning("Please enter a question before searching.")

# ─── Results ──────────────────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">Ask your knowledge base anything</div>
        <p class="empty-body">
            Type a question above and press <strong>Search</strong>. The pipeline
            will retrieve the most relevant document chunks from Pinecone and
            generate a grounded answer using your configured LLM provider.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    answer     = result.get("answer", "")
    sources    = result.get("sources", [])
    model      = result.get("model", "—").replace("models/", "")
    chunks_ret = result.get("chunks_retrieved", len(sources))
    elapsed    = st.session_state.elapsed

    # ── Answer card ───────────────────────────────────
    st.markdown(f"""
    <div class="answer-card">
        <p class="answer-label">Answer</p>
        <p class="answer-text">{_esc(answer)}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Meta row ──────────────────────────────────────
    st.markdown(f"""
    <div class="meta-row">
        <span class="meta-pill">🤖&nbsp; {_esc(model)}</span>
        <span class="meta-pill">📄&nbsp; {chunks_ret} source{'s' if chunks_ret != 1 else ''}</span>
        <span class="meta-pill">⏱&nbsp; {elapsed:.2f}s</span>
        {'<span class="meta-pill">🔎&nbsp; ' + _esc(filter_source) + '</span>' if filter_source and filter_source != 'All' else ''}
    </div>
    """, unsafe_allow_html=True)

    # ── Copy answer button ────────────────────────────
    copy_col, _ = st.columns([1, 5])
    with copy_col:
        if st.button("Copy answer", use_container_width=True):
            st.write(f"```\n{answer}\n```")

    # ── Source cards ──────────────────────────────────
    if sources:
        st.markdown(f"""
        <div class="section-header">
            <span class="section-title">Retrieved Sources</span>
            <span class="section-count">{len(sources)}</span>
        </div>
        """, unsafe_allow_html=True)

        cols_per_row = 3
        rows = [sources[i:i + cols_per_row] for i in range(0, len(sources), cols_per_row)]

        for row in rows:
            cols = st.columns(cols_per_row)
            for col, citation in zip(cols, row):
                with col:
                    src_type   = citation.get("source", "UNKNOWN").upper()
                    badge_cls  = _badge_class(src_type)
                    score      = float(citation.get("score", 0.0))
                    score_pct  = int(score * 100)
                    chunk_id   = citation.get("chunk_id",    "—")
                    doc_id     = citation.get("document_id", "—")
                    preview    = _esc(citation.get("content_preview") or "No preview available.")

                    def _trim(s: str, n: int = 30) -> str:
                        return (s[:n] + "…") if len(s) > n else s

                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-card-top">
                            <span class="source-badge {badge_cls}">{src_type}</span>
                            <span class="score-value">{score:.3f}</span>
                        </div>
                        <div class="score-bar-wrap">
                            <div class="score-bar-track">
                                <div class="score-bar-fill" style="width:{score_pct}%"></div>
                            </div>
                        </div>
                        <div class="src-meta">
                            <div class="src-meta-row">
                                <span class="src-meta-key">Document</span>
                                <span class="src-meta-val">{_trim(_esc(doc_id), 32)}</span>
                            </div>
                            <div class="src-meta-row">
                                <span class="src-meta-key">Chunk</span>
                                <span class="src-meta-val">{_trim(_esc(chunk_id), 32)}</span>
                            </div>
                        </div>
                        <div class="src-preview">{preview}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No source chunks were retrieved for this query. "
                "The answer is based on the model's general knowledge only.")
