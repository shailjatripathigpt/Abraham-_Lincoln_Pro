# app.py
"""
Streamlit UI for your RAG (FAISS + Groq API) with:
✅ Ask question -> Top chunks -> Answer
✅ Confidence score
✅ Saves history to PROJECT_ROOT/rag_history/rag_history.jsonl
✅ Google Drive integration for FAISS files

UI extras added (NO change to RAG functionality):
✅ Fixed-height scrollable chat box (TRUE fixed height using st.container(height=...))
✅ Auto-scroll to latest answer (scrolls inside the chat box, not page)
✅ Typing animation (visual only)
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
# CONSTANTS (NO UI SETTINGS)
# =========================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and accurate model
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

# ✅ chat fixed height (keep same feel as your picture)
CHAT_HEIGHT_PX = 420

SYSTEM_PROMPT = """You are a strict retrieval-grounded QA assistant.

RULES (must follow):
1) Use ONLY the provided CONTEXT. Do NOT use outside knowledge.
2) Answer in 1–3 sentences maximum.
3) Do NOT write "CITATIONS:", do NOT mention SOURCE numbers, do NOT mention chunk ids.
4) If the answer is not explicitly present in CONTEXT, reply exactly:
   Not found in provided documents.
"""


# =========================
# DOWNLOAD FROM GOOGLE DRIVE
# =========================
def download_from_drive(file_id: str, destination: Path):
    """Download file from Google Drive if it doesn't exist"""
    if destination.exists():
        return True
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            str(destination),
            quiet=False
        )
        return destination.exists()
    except Exception as e:
        st.warning(f"Could not download {destination.name} from Google Drive: {str(e)}")
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
        buf = ""
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
                buf = ""
                continue
            except Exception:
                pass

            buf += s
            try:
                obj = json.loads(buf)
                if isinstance(obj, dict):
                    yield obj
                    buf = ""
            except Exception:
                if len(buf) > 5_000_000:
                    buf = ""


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
    return s[:max_chars].rstrip() + " ...[TRUNCATED]"


def build_context(top_chunks: List[Dict[str, Any]], per_chunk_chars: int) -> str:
    parts = []
    for i, c in enumerate(top_chunks, start=1):
        txt = truncate_text(c.get("text", ""), per_chunk_chars)
        parts.append(
            f"SOURCE {i}:\n"
            f"title={c.get('title','Unknown')} | author={c.get('author','Unknown')} | year={c.get('publish_year')} | page={c.get('page_number')}\n"
            f"text={txt}\n"
        )
    return "\n".join(parts)


# =========================
# GROQ GENERATION - FIXED TO MATCH PHI3:MINI FORMAT
# =========================
def groq_generate(prompt: str) -> str:
    """Generate response using Groq API - using same prompt format as phi3:mini"""
    if not GROQ_API_KEY:
        raise ValueError("Groq API key is missing")
    
    # Use the EXACT same prompt format that worked with phi3:mini
    # The prompt already contains the system instructions at the beginning
    # So we just need to send it as a user message
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that answers questions based on the provided context."
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.0,  # EXACTLY same as phi3:mini (0.0 for deterministic)
        "max_tokens": 300,
        "top_p": 0.7,  # EXACTLY same as phi3:mini
        "stream": False
    }
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=GROQ_TIMEOUT_S
        )
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0]["message"]["content"].strip()
            
            # Remove any prefix if present
            if answer.lower().startswith("answer:"):
                answer = answer[7:].strip()
                
            return answer
        else:
            return "Error: No response from Groq API"
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Groq API connection error: {str(e)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid response from Groq API: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Groq API error: {str(e)}")


# =========================
# LOAD RESOURCES (CACHED)
# =========================
@st.cache_resource(show_spinner=True)
def load_rag_resources(project_root: Path):
    emb_dir = project_root / "embeddings"
    chunks_dir = project_root / "chunks"

    index_path = emb_dir / "chunks.faiss"
    meta_path = emb_dir / "chunks_meta.jsonl"
    chunks_path = chunks_dir / "all_chunks.jsonl"

    # Download files from Google Drive if needed
    with st.spinner("Checking for FAISS index..."):
        download_from_drive(FAISS_FILE_ID, index_path)
    
    with st.spinner("Checking for metadata..."):
        download_from_drive(META_FILE_ID, meta_path)
    
    with st.spinner("Checking for chunks data..."):
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
        "paths": {"index": index_path, "meta": meta_path, "chunks": chunks_path},
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
        candidates.append(
            {
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
                "text": chunk_text.get(cid, ""),
            }
        )

    filtered = [c for c in candidates if isinstance(c.get("text"), str) and c["text"].strip()]
    return filtered[:FINAL_K], candidates


# =========================
# UI (EXACTLY THE SAME AS YOUR ORIGINAL)
# =========================
def inject_css():
    st.markdown(
        """
        <style>
          .stApp { background: #2d2a25; }
          header[data-testid="stHeader"] { background: transparent; }
          footer { visibility: hidden; }

          .lincoln-top {
            background: linear-gradient(180deg, #3b6b8a 0%, #2f556f 100%);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 14px 18px;
            margin: 6px 0 12px 0;
            box-shadow: 0 10px 22px rgba(0,0,0,0.25);
            color: #f5e8cf;
            text-align:center;
            font-family: Georgia, 'Times New Roman', serif;
            font-weight: 900;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-size: 26px;
          }

          /* Style ONLY the first main two-column block */
          div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stVerticalBlock"]{
            background: rgba(80,55,30,0.35);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.25);
          }
          div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stVerticalBlock"]{
            background: linear-gradient(180deg, rgba(246,235,212,0.96) 0%, rgba(235,217,185,0.96) 100%);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.25);
          }

          .leftTitle {
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 22px;
            font-weight: 900;
            color: #f3e7cf;
            margin-top: 12px;
          }
          .leftSub {
            color: rgba(243,231,207,0.85);
            font-size: 13px;
            margin-top: 2px;
            margin-bottom: 6px;
          }

          /* Chat bubbles (text black) */
          .row { display:flex; gap:10px; align-items:flex-start; margin: 8px 0; }
          .bubble { color:#000 !important; }

          .avatar {
            width: 34px; height: 34px; border-radius: 50%;
            background: rgba(0,0,0,0.12);
            border: 1px solid rgba(0,0,0,0.18);
            display:flex; align-items:center; justify-content:center;
            font-family: Georgia, serif; font-weight: 900; color: rgba(0,0,0,0.65);
            flex: 0 0 auto;
          }
          .bubble {
            border-radius: 12px;
            padding: 10px 12px;
            border: 1px solid rgba(0,0,0,0.18);
            box-shadow: 0 6px 12px rgba(0,0,0,0.12);
            line-height: 1.35;
            max-width: 92%;
            word-wrap: break-word;
          }
          .assistant { background: rgba(255,255,255,0.65); }
          .user { background: rgba(220,235,245,0.70); margin-left:auto; }

          .metaLine { color: rgba(0,0,0,0.65) !important; font-size: 12px; margin-top: 4px; }
          .divider { border-top: 1px solid rgba(0,0,0,0.18); margin: 12px 0; }

          /* Send button */
          .stButton > button {
            background: #0f172a;
            color: #ffffff;
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 10px 16px;
            font-weight: 800;
          }
          .stButton > button:hover { background: #111c35; color: #fff; }

          /* Remove extra boxes around input */
          div[data-testid="stForm"], div[data-testid="stForm"] > div {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
          }
          div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
          }

          /* Input style (single box) */
          div[data-testid="stTextInput"] input {
            width: 100% !important;
            background: rgba(255,255,255,0.65) !important;
            border: 1px solid rgba(0,0,0,0.25) !important;
            border-radius: 10px !important;
            padding: 14px 14px !important;
            color: rgba(0,0,0,0.75) !important;
            box-shadow: none !important;
          }
          div[data-testid="stTextInput"] input::placeholder { color: rgba(0,0,0,0.45) !important; }

          /* Typing animation dots */
          .typing {
            display: inline-flex;
            gap: 6px;
            align-items: center;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.18);
            box-shadow: 0 6px 12px rgba(0,0,0,0.12);
            background: rgba(255,255,255,0.65);
          }
          .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: rgba(0,0,0,0.45);
            animation: blink 1.2s infinite;
          }
          .dot:nth-child(2){ animation-delay: 0.2s; }
          .dot:nth-child(3){ animation-delay: 0.4s; }

          @keyframes blink {
            0%, 80%, 100% { opacity: 0.25; transform: translateY(0px); }
            40% { opacity: 1; transform: translateY(-2px); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar():
    st.markdown('<div class="lincoln-top">THE ABRAHAM LINCOLN CHATBOT (GROQ)</div>', unsafe_allow_html=True)


def show_portrait():
    p = Path(PORTRAIT_PATH)
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.warning("Portrait not found. Check PORTRAIT_PATH.")


def render_chat_bubble(role: str, text: str, confidence: Optional[int] = None):
    if role == "user":
        st.markdown(
            f'<div class="row" style="justify-content:flex-end;">'
            f'  <div class="bubble user"><b>You</b>&nbsp;&nbsp; {text}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        conf_html = ""
        if confidence is not None:
            conf_html = f'<div class="metaLine">Confidence: <b>{confidence}/100</b></div>'
        st.markdown(
            f'<div class="row">'
            f'  <div class="avatar">AL</div>'
            f'  <div>'
            f'    <div class="bubble assistant">{text}</div>'
            f'    {conf_html}'
            f'  </div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_typing_indicator():
    st.markdown(
        """
        <div class="row">
          <div class="avatar">AL</div>
          <div class="typing" aria-label="typing">
            <span class="dot"></span><span class="dot"></span><span class="dot"></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def autoscroll_inside_chat():
    # This scrolls INSIDE the fixed-height st.container() (not the whole page)
    st.markdown(
        """
        <script>
          (function() {
            const root = window.parent.document;
            const el = root.getElementById('chat-scroll-anchor');
            if (el) el.scrollIntoView({behavior:'smooth', block:'end'});
          })();
        </script>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="The Abraham Lincoln Chatbot", layout="wide")
    inject_css()
    render_topbar()

    # Check for Groq API key
    if not GROQ_API_KEY:
        st.error("❌ Groq API key is missing! Please add it to your Streamlit secrets.toml file.")
        st.info("Add this to your `.streamlit/secrets.toml` file:\n\nGROQ_API_KEY = \"your-groq-api-key-here\"")
        st.stop()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    history_path = project_root / "rag_history" / "rag_history.jsonl"

    try:
        resources = load_rag_resources(project_root)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # state
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
        show_portrait()
        st.markdown('<div class="leftTitle">Chat with Abraham Lincoln</div>', unsafe_allow_html=True)
        st.markdown('<div class="leftSub">Ask me anything about my life and times.</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("• Groq API (Llama 3.1)")
        st.markdown("• Google Drive Storage")
        st.markdown("• FAISS Vector Search")

    with right:
        # ✅ TRUE fixed-height scroll container (this prevents the box from growing)
        chat_area = st.container(height=CHAT_HEIGHT_PX, border=False)

        with chat_area:
            if not st.session_state.chat:
                render_chat_bubble("assistant", "Hello there! I am Abraham Lincoln. How can I assist you today?")

            for msg in st.session_state.chat[-200:]:
                if msg["role"] == "user":
                    render_chat_bubble("user", msg["text"])
                else:
                    render_chat_bubble("assistant", msg["text"], confidence=msg.get("confidence"))

            if st.session_state.is_typing:
                render_typing_indicator()

            # anchor must be INSIDE the scroll container
            st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)

        # auto-scroll after chat renders
        autoscroll_inside_chat()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.session_state.show_context = st.checkbox(
            "Show retrieved context (inside Sources)",
            value=st.session_state.show_context,
        )

        with st.form("chat_form", clear_on_submit=True):
            in_col, send_col = st.columns([5, 1])
            with in_col:
                question = st.text_input(
                    "Type your message...",
                    value="",
                    label_visibility="collapsed",
                    placeholder="Type your message...",
                )
            with send_col:
                send = st.form_submit_button("Send", use_container_width=True)

        if send:
            user_q = (question or "").strip()
            if not user_q:
                st.warning("Please type a message.")
                st.stop()

            st.session_state.chat.append({"role": "user", "text": user_q})
            st.session_state.pending_user_q = user_q
            st.session_state.is_typing = True
            st.rerun()

    # Generation step (using Groq API instead of Ollama)
    if st.session_state.is_typing:
        user_q = (st.session_state.pending_user_q or "").strip()
        if not user_q:
            st.session_state.is_typing = False
            st.stop()

        with st.spinner("Retrieving relevant chunks..."):
            top_chunks, _ = retrieve_chunks(resources, user_q)

        if not top_chunks:
            answer = "Not found in provided documents."
            st.session_state.chat.append({"role": "assistant", "text": answer, "confidence": 0, "sources": []})
            write_jsonl_line(
                history_path,
                {"ts": utc_now_iso(), "question": user_q, "answer": answer, "confidence": 0, "sources": []},
            )
            st.session_state.is_typing = False
            st.session_state.pending_user_q = ""
            st.rerun()

        conf = compute_confidence(top_chunks)
        context = build_context(top_chunks, per_chunk_chars=PER_CHUNK_CHARS)

        # EXACT SAME PROMPT AS PHI3:MINI VERSION
        prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
"""

        with st.spinner("Generating answer with Groq..."):
            try:
                answer = groq_generate(prompt)
            except Exception as e:
                answer = f"Error: {type(e).__name__}: {str(e)[:300]}"
                conf = {"confidence": 0}

        st.session_state.chat.append(
            {"role": "assistant", "text": answer, "confidence": conf.get("confidence", 0), "sources": top_chunks}
        )

        write_jsonl_line(
            history_path,
            {
                "ts": utc_now_iso(),
                "question": user_q,
                "answer": answer,
                "confidence": conf.get("confidence", 0),
                "confidence_details": conf,
                "sources": [
                    {
                        "score": c.get("score"),
                        "chunk_id": c.get("chunk_id"),
                        "doc_id": c.get("doc_id"),
                        "title": c.get("title"),
                        "author": c.get("author"),
                        "publish_year": c.get("publish_year"),
                        "publisher": c.get("publisher"),
                        "scan_source": c.get("scan_source"),
                        "source_url": c.get("source_url"),
                        "page_number": c.get("page_number"),
                    }
                    for c in top_chunks
                ],
            },
        )

        st.session_state.is_typing = False
        st.session_state.pending_user_q = ""
        st.rerun()

    # SOURCES (unchanged)
    last = None
    for m in reversed(st.session_state.chat):
        if m.get("role") == "assistant" and m.get("sources"):
            last = m
            break

    if last and last.get("sources"):
        st.markdown("### 📌 Sources (real)")
        for i, c in enumerate(last["sources"], start=1):
            with st.expander(
                f"{i}. score={c['score']:.4f} | {c.get('scan_source')} | page={c.get('page_number')}"
            ):
                st.write(f"**Title:** {c.get('title')}")
                st.write(f"**Author:** {c.get('author')}")
                st.write(f"**Year:** {c.get('publish_year')}")
                st.write(f"**Publisher:** {c.get('publisher')}")
                st.write(f"**Chunk ID:** `{c.get('chunk_id')}`")
                st.write(f"**Source URL:** {c.get('source_url')}")
                if st.session_state.show_context:
                    st.divider()
                    st.write(c.get("text", ""))


if __name__ == "__main__":
    main()