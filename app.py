# app.py
"""
Streamlit UI for your RAG (FAISS + Ollama phi3:mini) with:
✅ Ask question -> Top chunks -> Answer
✅ Confidence score
✅ Saves history to PROJECT_ROOT/rag_history/rag_history.jsonl

UI extras added (NO change to RAG functionality):
✅ Fixed-height scrollable chat box (TRUE fixed height using st.container(height=...))
✅ Auto-scroll to latest answer (scrolls inside the chat box, not page)
✅ Typing animation (visual only)

ADDED (without changing RAG retrieval / UI flow):
✅ Groq answering (better answers) + fallback to Ollama
✅ Google Drive service account integration (optional) to upload rag_history.jsonl
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional, Tuple

import numpy as np
import requests
import streamlit as st

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

# --- Optional (Google Drive upload) ---
# Install if you want Drive upload:
# pip install google-api-python-client google-auth
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    _GDRIVE_OK = True
except Exception:
    _GDRIVE_OK = False


# =========================
# CONSTANTS (NO UI SETTINGS)
# =========================
OLLAMA_MODEL = "phi3:mini"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 600
OLLAMA_NUM_CTX = 2048
OLLAMA_NUM_PREDICT = 160

# ✅ Groq settings (better answering)
# Put key in Streamlit secrets or env:
# - st.secrets["GROQ_API_KEY"] = "..."
# or environment: GROQ_API_KEY=...
GROQ_MODEL = "llama-3.1-70b-versatile"  # you can change anytime
GROQ_TIMEOUT_S = 60

TOP_K = 15
FINAL_K = 5
PER_CHUNK_CHARS = 900

PORTRAIT_PATH = r"C:\Users\User\OneDrive\Desktop\output.jpg"

# ✅ chat fixed height (keep same feel as your picture)
CHAT_HEIGHT_PX = 420

# ✅ Stronger grounding + better phrasing (still strict rules)
SYSTEM_PROMPT = """You are a strict retrieval-grounded QA assistant.

HARD RULES (must follow):
1) Use ONLY the provided CONTEXT. Do NOT use outside knowledge.
2) Answer in 1–3 sentences maximum.
3) Do NOT write "CITATIONS:", do NOT mention SOURCE numbers, do NOT mention chunk ids, do NOT mention "context says".
4) If the answer is not explicitly present in CONTEXT, reply exactly:
   Not found in provided documents.

ANSWER QUALITY:
- Be direct and specific.
- Prefer exact names/dates/places stated in CONTEXT.
- If multiple facts are present, combine them into 1–3 clean sentences.
"""

# ✅ Google Drive (optional): auto-upload history file after each answer
# Put service account json in st.secrets (recommended) or env.
# REQUIRED secrets (suggested):
# st.secrets["GDRIVE_SERVICE_ACCOUNT_JSON"] = { ... full json ... }
# st.secrets["GDRIVE_FOLDER_ID"] = "your_drive_folder_id"
UPLOAD_HISTORY_TO_DRIVE = False  # set True if you want auto-upload
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


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
# OLLAMA
# =========================
def ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.7,
            "num_ctx": int(OLLAMA_NUM_CTX),
            "num_predict": int(OLLAMA_NUM_PREDICT),
        },
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_S)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


# =========================
# GROQ (OpenAI-compatible endpoint)
# =========================
def _get_groq_key() -> str:
    k = ""
    try:
        k = (st.secrets.get("GROQ_API_KEY") or "").strip()
    except Exception:
        k = ""
    if not k:
        k = (os.getenv("GROQ_API_KEY") or "").strip()
    return k


def groq_answer(system_prompt: str, context: str, question: str) -> str:
    """
    Uses Groq chat completions (OpenAI-compatible).
    Improves answer quality while enforcing your strict rules.
    """
    api_key = _get_groq_key()
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    user_content = f"""
You will answer the QUESTION using ONLY the CONTEXT.

CONTEXT:
{context}

QUESTION:
{question}

RESPONSE REQUIREMENTS:
- Output ONLY the final answer (no preface, no bullet labels).
- 1–3 sentences maximum.
- Do not mention sources, citations, chunk ids, or the word "context".
- If the answer is not explicitly stated in CONTEXT, output exactly:
Not found in provided documents.
""".strip()

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "top_p": 0.7,
        "max_tokens": 220,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=GROQ_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    txt = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

    # Light output guardrails (generator-only; retrieval unchanged)
    bad_markers = ["CITATIONS", "SOURCE", "Chunk", "chunk_id", "context", "Context"]
    for bm in bad_markers:
        if bm in txt:
            txt = txt.replace(bm, "").strip()

    if not txt or len(txt) < 3:
        return "Not found in provided documents."

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
    if len(sents) > 3:
        txt = " ".join(sents[:3]).strip()

    txt = txt.replace("ANSWER:", "").replace("Final answer:", "").strip()

    if any(k in txt.lower() for k in ["source 1", "source 2", "chunk id", "chunk_id", "citations"]):
        return "Not found in provided documents."

    return txt


def generate_answer(system_prompt: str, context: str, question: str) -> Tuple[str, str]:
    """
    Returns (answer, engine_used)
    Priority: Groq -> fallback to Ollama
    """
    try:
        ans = groq_answer(system_prompt, context, question)
        return ans, "groq"
    except Exception:
        prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        ans = ollama_generate(prompt)
        return ans, "ollama"


# =========================
# GOOGLE DRIVE (optional upload)
# =========================
def _get_drive_folder_id() -> str:
    fid = ""
    try:
        fid = (st.secrets.get("GDRIVE_FOLDER_ID") or "").strip()
    except Exception:
        fid = ""
    if not fid:
        fid = (os.getenv("GDRIVE_FOLDER_ID") or "").strip()
    return fid


def _get_drive_service():
    """
    Loads service account credentials from:
    - st.secrets["GDRIVE_SERVICE_ACCOUNT_JSON"] (dict or json string), OR
    - env var GDRIVE_SERVICE_ACCOUNT_JSON (json string), OR
    - env var GOOGLE_APPLICATION_CREDENTIALS (path to json file)
    """
    if not _GDRIVE_OK:
        raise RuntimeError("Google Drive libs not installed. Install google-api-python-client google-auth")

    sa_info = None

    # 1) Streamlit secrets (best)
    try:
        sa_info = st.secrets.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    except Exception:
        sa_info = None

    # 2) Env json
    if sa_info is None:
        env_json = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
        if env_json:
            try:
                sa_info = json.loads(env_json)
            except Exception:
                sa_info = None

    # 3) GOOGLE_APPLICATION_CREDENTIALS path
    if sa_info is None:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and Path(cred_path).exists():
            sa_info = json.loads(Path(cred_path).read_text(encoding="utf-8"))

    if sa_info is None:
        raise RuntimeError("Missing Google Drive service account JSON in secrets/env")

    if isinstance(sa_info, str):
        sa_info = json.loads(sa_info)

    creds = Credentials.from_service_account_info(sa_info, scopes=GDRIVE_SCOPES)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    return service


def upload_history_to_drive(history_file: Path) -> None:
    """
    Uploads (or overwrites) rag_history.jsonl into a Drive folder.
    Best-effort; does not affect RAG/UI flow.
    """
    folder_id = _get_drive_folder_id()
    if not folder_id:
        raise RuntimeError("Missing GDRIVE_FOLDER_ID")

    service = _get_drive_service()

    filename = history_file.name
    q = f"'{folder_id}' in parents and name='{filename}' and trashed=false"
    res = service.files().list(q=q, fields="files(id,name)", pageSize=10).execute()
    files = res.get("files") or []

    media = MediaFileUpload(str(history_file), mimetype="application/json", resumable=True)

    if files:
        file_id = files[0]["id"]
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        meta = {"name": filename, "parents": [folder_id]}
        service.files().create(body=meta, media_body=media, fields="id").execute()


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
# UI
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
    st.markdown('<div class="lincoln-top">THE ABRAHAM LINCOLN CHATBOT</div>', unsafe_allow_html=True)


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

        groq_present = bool(_get_groq_key())
        st.caption(f"Answering engine: {'Groq (primary)' if groq_present else 'Ollama (fallback)'}")

    with right:
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

            st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)

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

    # Generation step (same RAG functionality)
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

        with st.spinner("Generating answer (Groq preferred, Ollama fallback)..."):
            try:
                answer, engine_used = generate_answer(SYSTEM_PROMPT, context, user_q)
            except Exception as e:
                answer = f"Generation error: {type(e).__name__}: {str(e)[:300]}"
                engine_used = "error"
                conf = {"confidence": 0}

        st.session_state.chat.append(
            {
                "role": "assistant",
                "text": answer,
                "confidence": conf.get("confidence", 0),
                "sources": top_chunks,
                "engine": engine_used,
            }
        )

        write_jsonl_line(
            history_path,
            {
                "ts": utc_now_iso(),
                "question": user_q,
                "answer": answer,
                "engine": engine_used,
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

        # Optional: upload history to Drive (best-effort; no UI/flow change)
        if UPLOAD_HISTORY_TO_DRIVE:
            try:
                upload_history_to_drive(history_path)
            except Exception:
                pass

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
            with st.expander(f"{i}. score={c['score']:.4f} | {c.get('scan_source')} | page={c.get('page_number')}"):
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
