# app.py
"""
Streamlit UI for your RAG (FAISS + Groq API) with:
✅ Ask question -> Top chunks -> Answer
✅ Confidence score
✅ Saves history to PROJECT_ROOT/rag_history/rag_history.jsonl
✅ Google Drive integration for FAISS files
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional

import numpy as np
import requests
import streamlit as st
import gdown

try:
    import faiss
except ImportError:
    st.error("faiss not installed. Run: pip install faiss-cpu")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("sentence-transformers not installed. Run: pip install sentence-transformers")
    raise


# =========================
# CONSTANTS
# =========================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL = "mixtral-8x7b-32768"  # Better model for RAG
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_S = 30

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Google Drive IDs
FAISS_FILE_ID = "1Zvt2fP0ih70dGFXoIvuDX27427wQUYym"
META_FILE_ID = "1bVrE_JFgdK0kdZaaHCBxPItfPda_xxvo"
CHUNKS_FILE_ID = "16eTgJEilBGdH6dmgkoH92bY5N87T7-Wm"

TOP_K = 15
FINAL_K = 5
PER_CHUNK_CHARS = 900

PORTRAIT_PATH = r"C:\Users\User\OneDrive\Desktop\output.jpg"
CHAT_HEIGHT_PX = 420


# =========================
# DOWNLOAD FROM GOOGLE DRIVE
# =========================
def download_from_drive(file_id: str, destination: Path):
    if destination.exists():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(destination), quiet=False)
        return destination.exists()
    except Exception as e:
        st.warning(f"Could not download {destination.name}: {str(e)}")
        return False


# =========================
# ROOT DETECT
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(40):
        if (cur / "embeddings" / "chunks.faiss").exists():
            return cur
        cur = cur.parent
    return start.resolve()


# =========================
# IO HELPERS
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except:
                continue


def write_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# CHUNK TEXT MAP
# =========================
def load_chunks_text_map(chunks_path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for r in iter_jsonl(chunks_path):
        cid = r.get("chunk_id")
        txt = r.get("text")
        if isinstance(cid, str) and cid and isinstance(txt, str):
            m[cid] = txt
    return m


# =========================
# CONFIDENCE
# =========================
def compute_confidence(top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not top_chunks:
        return {"confidence": 0, "reason": "no_chunks"}

    scores = np.array([float(c.get("score", 0.0)) for c in top_chunks], dtype=np.float32)
    s1 = float(scores[0])
    s2 = float(scores[1]) if len(scores) > 1 else (s1 - 0.15)

    level = np.clip((s1 - 0.20) / 0.55, 0.0, 1.0)
    margin = np.clip((s1 - s2) / 0.20, 0.0, 1.0)

    good = (scores >= max(0.30, s1 - 0.10)).sum()
    conc = np.clip(good / max(1, len(scores)), 0.0, 1.0)

    srcs = [str(c.get("scan_source") or "") for c in top_chunks]
    srcs = [s for s in srcs if s]
    agree = 0.0
    if srcs:
        most = max(srcs.count(x) for x in set(srcs))
        agree = np.clip(most / len(srcs), 0.0, 1.0)

    conf01 = 0.55 * level + 0.25 * margin + 0.10 * conc + 0.10 * agree
    conf = int(round(float(np.clip(conf01, 0.0, 1.0) * 100)))

    return {
        "confidence": conf,
        "top_score": s1,
        "second_score": s2,
        "level": float(level),
        "margin": float(margin),
        "concentration": float(conc),
        "source_agreement": float(agree),
    }


# =========================
# CONTEXT BUILD
# =========================
def truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "..."


def build_context(top_chunks: List[Dict[str, Any]], per_chunk_chars: int) -> str:
    """Build context with clear formatting"""
    if not top_chunks:
        return "No context available."
    
    parts = ["RELEVANT DOCUMENT EXCERPTS:"]
    for i, c in enumerate(top_chunks, start=1):
        txt = truncate_text(c.get("text", ""), per_chunk_chars)
        source_info = []
        if c.get('title'):
            source_info.append(f"Title: {c.get('title')}")
        if c.get('author'):
            source_info.append(f"Author: {c.get('author')}")
        if c.get('publish_year'):
            source_info.append(f"Year: {c.get('publish_year')}")
        
        source_str = " | ".join(source_info) if source_info else "Source information not available"
        
        parts.append(f"\n--- Excerpt {i} ---")
        parts.append(f"[{source_str}]")
        parts.append(f"{txt}")
    
    return "\n".join(parts)


# =========================
# GROQ GENERATION - WORKING VERSION
# =========================
def groq_generate(context: str, question: str) -> str:
    """Generate response using Groq API"""
    if not GROQ_API_KEY:
        return "Error: Groq API key missing"
    
    system_prompt = """You are Abraham Lincoln. Answer questions based ONLY on the provided context.
    
CRITICAL RULES:
1. Use ONLY information from the context below
2. If the answer is NOT in the context, say: "Not found in provided documents."
3. Keep answers concise (1-3 sentences)
4. Answer naturally as Abraham Lincoln
5. Do NOT mention that you're using context
6. Do NOT cite sources or use phrases like "according to the context"
"""
    
    user_prompt = f"""CONTEXT:
{context}

QUESTION: {question}

Answer as Abraham Lincoln using ONLY the context above. If unsure, say "Not found in provided documents."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 250,
        "top_p": 0.9,
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=GROQ_TIMEOUT_S)
        
        if response.status_code != 200:
            return f"API Error {response.status_code}"
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip()
            
            # Clean up common prefixes
            prefixes = [
                "Based on the context,",
                "According to the context,",
                "The context states that",
                "In the provided context,",
                "As Abraham Lincoln,",
                "As the context shows,"
            ]
            
            for prefix in prefixes:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
            
            return answer if answer else "Not found in provided documents."
        else:
            return "Not found in provided documents."
            
    except Exception as e:
        return f"Error: {str(e)[:100]}"


# =========================
# LOAD RESOURCES
# =========================
@st.cache_resource(show_spinner=True)
def load_rag_resources(project_root: Path):
    emb_dir = project_root / "embeddings"
    chunks_dir = project_root / "chunks"

    index_path = emb_dir / "chunks.faiss"
    meta_path = emb_dir / "chunks_meta.jsonl"
    chunks_path = chunks_dir / "all_chunks.jsonl"

    # Download files if needed
    download_from_drive(FAISS_FILE_ID, index_path)
    download_from_drive(META_FILE_ID, meta_path)
    download_from_drive(CHUNKS_FILE_ID, chunks_path)

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSONL not found: {meta_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSONL not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    meta_rows = list(iter_jsonl(meta_path))
    chunk_text = load_chunks_text_map(chunks_path)
    emb_model = SentenceTransformer(EMBED_MODEL_NAME)

    return {
        "index": index,
        "meta_rows": meta_rows,
        "chunk_text": chunk_text,
        "emb_model": emb_model,
    }


# =========================
# SEARCH
# =========================
def retrieve_chunks(resources: Dict[str, Any], question: str):
    index = resources["index"]
    meta_rows = resources["meta_rows"]
    chunk_text = resources["chunk_text"]
    emb_model = resources["emb_model"]

    q_emb = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, max(TOP_K, FINAL_K))

    candidates: List[Dict[str, Any]] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta_rows):
            continue
        m = meta_rows[idx]
        cid = m.get("chunk_id")
        txt = chunk_text.get(cid, "")
        if txt.strip():
            candidates.append({
                "score": float(score),
                "chunk_id": cid,
                "doc_id": m.get("doc_id"),
                "title": m.get("title"),
                "author": m.get("author"),
                "publish_year": m.get("publish_year"),
                "publisher": m.get("publisher"),
                "scan_source": m.get("scan_source"),
                "source_url": m.get("source_url"),
                "page_number": m.get("page_number"),
                "text": txt,
            })

    # Sort by score and take top FINAL_K
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:FINAL_K], candidates


# =========================
# UI (SAME AS BEFORE)
# =========================
def inject_css():
    st.markdown("""
    <style>
    .stApp { background: #2d2a25; }
    .lincoln-top {
        background: linear-gradient(180deg, #3b6b8a 0%, #2f556f 100%);
        border-radius: 10px; padding: 14px 18px; margin: 6px 0 12px 0;
        color: #f5e8cf; text-align:center; font-family: Georgia; 
        font-weight: 900; letter-spacing: 1px; text-transform: uppercase; font-size: 26px;
    }
    .leftTitle { font-family: Georgia; font-size: 22px; font-weight: 900; color: #f3e7cf; margin-top: 12px; }
    .leftSub { color: rgba(243,231,207,0.85); font-size: 13px; margin-top: 2px; margin-bottom: 6px; }
    .row { display:flex; gap:10px; align-items:flex-start; margin: 8px 0; }
    .avatar { width: 34px; height: 34px; border-radius: 50%; background: rgba(0,0,0,0.12); 
              border: 1px solid rgba(0,0,0,0.18); display:flex; align-items:center; 
              justify-content:center; font-family: Georgia; font-weight: 900; color: rgba(0,0,0,0.65); }
    .bubble { border-radius: 12px; padding: 10px 12px; border: 1px solid rgba(0,0,0,0.18);
              box-shadow: 0 6px 12px rgba(0,0,0,0.12); line-height: 1.35; max-width: 92%; word-wrap: break-word; }
    .assistant { background: rgba(255,255,255,0.65); }
    .user { background: rgba(220,235,245,0.70); margin-left:auto; }
    .metaLine { color: rgba(0,0,0,0.65) !important; font-size: 12px; margin-top: 4px; }
    .divider { border-top: 1px solid rgba(0,0,0,0.18); margin: 12px 0; }
    .stButton > button { background: #0f172a; color: #ffffff; border-radius: 10px; padding: 10px 16px; font-weight: 800; }
    .stButton > button:hover { background: #111c35; }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="The Abraham Lincoln Chatbot", layout="wide")
    inject_css()
    
    st.markdown('<div class="lincoln-top">THE ABRAHAM LINCOLN CHATBOT (GROQ)</div>', unsafe_allow_html=True)
    
    if not GROQ_API_KEY:
        st.error("❌ Groq API key missing! Add to secrets.toml")
        st.stop()
    
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    history_path = project_root / "rag_history" / "rag_history.jsonl"
    
    try:
        resources = load_rag_resources(project_root)
    except Exception as e:
        st.error(str(e))
        st.stop()
    
    # State
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "show_context" not in st.session_state:
        st.session_state.show_context = False
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False
    if "pending_user_q" not in st.session_state:
        st.session_state.pending_user_q = ""
    
    left, right = st.columns([1, 2.2], gap="large")
    
    with left:
        p = Path(PORTRAIT_PATH)
        if p.exists():
            st.image(str(p), use_container_width=True)
        else:
            st.warning("Portrait not found")
        st.markdown('<div class="leftTitle">Chat with Abraham Lincoln</div>', unsafe_allow_html=True)
        st.markdown('<div class="leftSub">Ask me anything about my life and times.</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("• Groq API")
        st.markdown("• FAISS Vector Search")
    
    with right:
        chat_area = st.container(height=CHAT_HEIGHT_PX, border=False)
        with chat_area:
            if not st.session_state.chat:
                st.markdown("""
                <div class="row">
                  <div class="avatar">AL</div>
                  <div class="bubble assistant">Hello there! I am Abraham Lincoln. How can I assist you today?</div>
                </div>
                """, unsafe_allow_html=True)
            
            for msg in st.session_state.chat[-200:]:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="row" style="justify-content:flex-end;">
                      <div class="bubble user"><b>You</b>&nbsp;&nbsp; {msg['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    conf_html = f'<div class="metaLine">Confidence: <b>{msg.get("confidence", 0)}/100</b></div>' if msg.get("confidence") is not None else ""
                    st.markdown(f"""
                    <div class="row">
                      <div class="avatar">AL</div>
                      <div>
                        <div class="bubble assistant">{msg['text']}</div>
                        {conf_html}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.session_state.is_typing:
                st.markdown("""
                <div class="row">
                  <div class="avatar">AL</div>
                  <div class="typing">
                    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
                  </div>
                </div>
                <style>
                .typing { display: inline-flex; gap: 6px; align-items: center; padding: 10px 12px;
                          border-radius: 12px; border: 1px solid rgba(0,0,0,0.18); background: rgba(255,255,255,0.65); }
                .dot { width: 7px; height: 7px; border-radius: 50%; background: rgba(0,0,0,0.45); animation: blink 1.2s infinite; }
                .dot:nth-child(2){ animation-delay: 0.2s; } .dot:nth-child(3){ animation-delay: 0.4s; }
                @keyframes blink { 0%,80%,100% { opacity: 0.25; } 40% { opacity: 1; } }
                </style>
                """, unsafe_allow_html=True)
            
            st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.session_state.show_context = st.checkbox("Show retrieved context", value=st.session_state.show_context)
        
        with st.form("chat_form", clear_on_submit=True):
            in_col, send_col = st.columns([5, 1])
            with in_col:
                question = st.text_input("Type your message...", label_visibility="collapsed", placeholder="Type your message...")
            with send_col:
                send = st.form_submit_button("Send", use_container_width=True)
        
        if send and question.strip():
            st.session_state.chat.append({"role": "user", "text": question.strip()})
            st.session_state.pending_user_q = question.strip()
            st.session_state.is_typing = True
            st.rerun()
    
    # Generation
    if st.session_state.is_typing:
        user_q = st.session_state.pending_user_q.strip()
        
        with st.spinner("Searching documents..."):
            top_chunks, _ = retrieve_chunks(resources, user_q)
        
        if not top_chunks:
            answer = "Not found in provided documents."
            st.session_state.chat.append({"role": "assistant", "text": answer, "confidence": 0, "sources": []})
            write_jsonl_line(history_path, {"ts": utc_now_iso(), "question": user_q, "answer": answer, "confidence": 0})
            st.session_state.is_typing = False
            st.session_state.pending_user_q = ""
            st.rerun()
        
        conf = compute_confidence(top_chunks)
        context = build_context(top_chunks, PER_CHUNK_CHARS)
        
        with st.spinner("Generating answer..."):
            answer = groq_generate(context, user_q)
        
        st.session_state.chat.append({
            "role": "assistant", 
            "text": answer, 
            "confidence": conf.get("confidence", 0), 
            "sources": top_chunks
        })
        
        write_jsonl_line(history_path, {
            "ts": utc_now_iso(),
            "question": user_q,
            "answer": answer,
            "confidence": conf.get("confidence", 0),
            "confidence_details": conf,
            "sources": [{
                "score": c.get("score"),
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "author": c.get("author"),
                "text": c.get("text")[:200] + "..." if len(c.get("text", "")) > 200 else c.get("text", "")
            } for c in top_chunks]
        })
        
        st.session_state.is_typing = False
        st.session_state.pending_user_q = ""
        st.rerun()
    
    # Sources
    last = next((m for m in reversed(st.session_state.chat) if m.get("role") == "assistant" and m.get("sources")), None)
    if last and last.get("sources"):
        st.markdown("### 📌 Sources")
        for i, c in enumerate(last["sources"], 1):
            with st.expander(f"{i}. Score: {c['score']:.3f} | {c.get('title', 'Unknown')}"):
                st.write(f"**Author:** {c.get('author', 'Unknown')}")
                st.write(f"**Year:** {c.get('publish_year', 'Unknown')}")
                if st.session_state.show_context:
                    st.divider()
                    st.write(c.get("text", ""))


if __name__ == "__main__":
    main()