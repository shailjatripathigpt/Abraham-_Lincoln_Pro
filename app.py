# app.py
"""
Streamlit UI for your RAG (FAISS + Ollama phi3:mini OR Groq) with:
✅ Ask question -> Top chunks -> Answer
✅ Confidence score
✅ Saves history to PROJECT_ROOT/rag_history/rag_history.jsonl

UI extras added (NO change to RAG functionality):
✅ Fixed-height scrollable chat box (TRUE fixed height using st.container(height=...))
✅ Auto-scroll to latest answer (scrolls inside the chat box, not page)
✅ Typing animation (visual only)

ADDED (without changing your UI / flow):
✅ Google Drive download (gdown) for chunks + meta + faiss (Streamlit Cloud)
✅ Groq as generator (reads GROQ_API_KEY from Streamlit Secrets)
✅ Phi3:mini kept as LOCAL fallback when Groq not available
✅ Better retrieval: de-dup + diversity selection (MMR-lite) to improve answer quality
✅ Better chunk truncation: more factual sentence-first

FIXED:
✅ If answer == "Not found in provided documents." then confidence forced to 0
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

# =========================
# OPTIONAL: gdown for Google Drive download
# =========================
try:
    import gdown
except Exception:
    gdown = None  # if not installed locally, app still works if files exist

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
# GOOGLE DRIVE IDs (YOUR FILES)
# =========================
FAISS_FILE_ID = "1Zvt2fP0ih70dGFXoIvuDX27427wQUYym"
META_FILE_ID = "1bVrE_JFgdK0kdZaaHCBxPItfPda_xxvo"
CHUNKS_FILE_ID = "16eTgJEilBGdH6dmgkoH92bY5N87T7-Wm"


# =========================
# CONSTANTS (NO UI SETTINGS)
# =========================
OLLAMA_MODEL = "phi3:mini"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 600
OLLAMA_NUM_CTX = 2048
OLLAMA_NUM_PREDICT = 180

# Groq
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_TIMEOUT_S = 60

TOP_K = 22
FINAL_K = 6
PER_CHUNK_CHARS = 750

PORTRAIT_PATH = r"C:\Users\User\OneDrive\Desktop\output.jpg"
CHAT_HEIGHT_PX = 420

SYSTEM_PROMPT = """You are a strict retrieval-grounded QA assistant.

RULES (must follow):
1) Use ONLY the provided CONTEXT. Do NOT use outside knowledge.
2) Answer in 1–3 sentences maximum.
3) Do NOT write "CITATIONS:", do NOT mention SOURCE numbers, do NOT mention chunk ids.
4) If the answer is not explicitly present in CONTEXT, reply exactly:
   Not found in provided documents.

STYLE:
- Be direct and specific.
- Prefer exact names/dates/places as written in the context.
- Do not add extra commentary.
"""


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


def is_not_found_answer(ans: str) -> bool:
    return (ans or "").strip() == "Not found in provided documents."


# =========================
# GOOGLE DRIVE DOWNLOAD
# =========================
def download_from_drive(file_id: str, destination: Path) -> bool:
    if destination.exists():
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)

    if gdown is None:
        st.warning("gdown not installed; cannot download from Google Drive. "
                   "Install: pip install gdown (or ensure files exist in repo).")
        return False

    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(destination), quiet=False)
        return destination.exists()
    except Exception as e:
        st.warning(f"Could not download {destination.name}: {str(e)}")
        return False


def ensure_rag_files_present(project_root: Path) -> None:
    emb_dir = project_root / "embeddings"
    chunks_dir = project_root / "chunks"

    index_path = emb_dir / "chunks.faiss"
    meta_path = emb_dir / "chunks_meta.jsonl"
    chunks_path = chunks_dir / "all_chunks.jsonl"

    if index_path.exists() and meta_path.exists() and chunks_path.exists():
        return

    with st.spinner("Downloading RAG files from Google Drive..."):
        ok1 = download_from_drive(FAISS_FILE_ID, index_path)
        ok2 = download_from_drive(META_FILE_ID, meta_path)
        ok3 = download_from_drive(CHUNKS_FILE_ID, chunks_path)

    if not (ok1 and ok2 and ok3):
        missing = [str(p) for p in [index_path, meta_path, chunks_path] if not p.exists()]
        raise FileNotFoundError("Missing RAG files:\n" + "\n".join(missing))


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
# RETRIEVAL IMPROVEMENTS (De-dup + diversity)
# =========================
def _dedup_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best = {}
    for c in cands:
        doc = str(c.get("doc_id") or "")
        page = str(c.get("page_number") or "")
        txt = (c.get("text") or "").strip()
        key = (doc, page, txt[:140])
        if key not in best or float(c.get("score", 0.0)) > float(best[key].get("score", 0.0)):
            best[key] = c
    return list(best.values())


def _mmr_select(candidates: List[Dict[str, Any]], k: int, lambda_mult: float = 0.78) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    scores = [float(c.get("score", 0.0)) for c in candidates]
    lo, hi = min(scores), max(scores)

    def norm(s: float) -> float:
        if hi <= lo:
            return 0.0
        return (s - lo) / (hi - lo)

    selected = []
    used_doc = set()
    used_page = set()
    used_title = set()

    for _ in range(min(k, len(candidates))):
        best = None
        best_val = -1e9

        for c in candidates:
            if c.get("_picked"):
                continue

            rel = norm(float(c.get("score", 0.0)))

            doc = str(c.get("doc_id") or "")
            page = str(c.get("page_number") or "")
            title = str(c.get("title") or "")

            penalty = 0.0
            if doc and doc in used_doc:
                penalty += 0.45
            if (doc, page) in used_page:
                penalty += 0.35
            if title and title in used_title:
                penalty += 0.20

            mmr = lambda_mult * rel - (1.0 - lambda_mult) * penalty
            if mmr > best_val:
                best_val = mmr
                best = c

        if best is None:
            break

        best["_picked"] = True
        selected.append(best)

        doc = str(best.get("doc_id") or "")
        page = str(best.get("page_number") or "")
        title = str(best.get("title") or "")

        if doc:
            used_doc.add(doc)
        used_page.add((doc, page))
        if title:
            used_title.add(title)

    for c in candidates:
        if "_picked" in c:
            del c["_picked"]

    return selected


# =========================
# CONTEXT BUILD (better snippets)
# =========================
def truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if not s:
        return ""

    # Prefer first 3–4 sentences (usually the factual part)
    sents = re.split(r"(?<=[.!?])\s+", s)
    head = " ".join([x for x in sents[:4] if x]).strip()
    if len(head) >= 120:
        s = head

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
# OLLAMA (phi3:mini)
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
# GROQ (preferred when available)
# =========================
def _get_groq_key() -> str:
    try:
        k = st.secrets["GROQ_API_KEY"]
        if isinstance(k, str) and k.strip():
            return k.strip()
    except Exception:
        pass
    return (os.getenv("GROQ_API_KEY") or "").strip()


def _get_groq_model() -> str:
    try:
        m = st.secrets.get("GROQ_MODEL")
        if isinstance(m, str) and m.strip():
            return m.strip()
    except Exception:
        pass
    return (os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL


def groq_generate(system_prompt: str, context: str, question: str) -> str:
    api_key = _get_groq_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in Streamlit Secrets.")

    model = _get_groq_model()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    user_content = f"""
Answer the QUESTION using ONLY the CONTEXT.

STRICT RULES:
- If the exact answer is not explicitly stated in CONTEXT, reply exactly:
Not found in provided documents.
- Do not guess, do not use outside knowledge.
- 1–3 sentences maximum.
- Do not mention sources, citations, chunk ids, or the word "context".
- Use exact names/dates/places as written.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "top_p": 0.7,
        "max_tokens": 240,
        "stream": False,
    }

    r = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=GROQ_TIMEOUT_S)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"raw": r.text}
        raise RuntimeError(f"Groq error {r.status_code}: {err}")

    data = r.json()
    txt = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

    txt = txt.replace("ANSWER:", "").replace("Final answer:", "").strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", txt) if s.strip()]
    if len(sents) > 3:
        txt = " ".join(sents[:3]).strip()
    if not txt:
        return "Not found in provided documents."
    return txt


def generate_answer(system_prompt: str, context: str, question: str) -> Tuple[str, str]:
    # Prefer Groq when available (Streamlit Cloud). Else use phi3:mini locally.
    if _get_groq_key():
        return groq_generate(system_prompt, context, question), "groq"

    prompt = f"""{system_prompt}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return ollama_generate(prompt), "ollama"


# =========================
# LOAD RESOURCES (CACHED)
# =========================
@st.cache_resource(show_spinner=True)
def load_rag_resources(project_root: Path):
    ensure_rag_files_present(project_root)

    emb_dir = project_root / "embeddings"
    chunks_dir = project_root / "chunks"

    index_path = emb_dir / "chunks.faiss"
    meta_path = emb_dir / "chunks_meta.jsonl"
    chunks_path = chunks_dir / "all_chunks.jsonl"

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
# SEARCH (improved)
# =========================
def retrieve_chunks(resources: Dict[str, Any], question: str):
    index = resources["index"]
    meta_rows = resources["meta_rows"]
    chunk_text = resources["chunk_text"]
    emb_model = resources["emb_model"]

    q_emb = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    fetch_k = max(TOP_K, FINAL_K * 4)
    D, I = index.search(q_emb, fetch_k)

    candidates: List[Dict[str, Any]] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta_rows):
            continue
        m = meta_rows[idx]
        cid = m.get("chunk_id")
        txt = chunk_text.get(cid, "")

        if not isinstance(txt, str) or not txt.strip():
            continue

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
                "text": txt,
            }
        )

    candidates = _dedup_candidates(candidates)
    candidates.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    selected = _mmr_select(candidates, k=FINAL_K, lambda_mult=0.78)

    return selected, candidates


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
        if _get_groq_key():
            st.caption(f"Generator: Groq ({_get_groq_model()})")
        else:
            st.caption(f"Generator: Ollama ({OLLAMA_MODEL})")

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

    # Generation step (same RAG flow)
    if st.session_state.is_typing:
        user_q = (st.session_state.pending_user_q or "").strip()
        if not user_q:
            st.session_state.is_typing = False
            st.stop()

        with st.spinner("Retrieving relevant chunks..."):
            top_chunks, _ = retrieve_chunks(resources, user_q)

        if not top_chunks:
            answer = "Not found in provided documents."
            final_conf = 0
            st.session_state.chat.append({"role": "assistant", "text": answer, "confidence": final_conf, "sources": []})
            write_jsonl_line(history_path, {"ts": utc_now_iso(), "question": user_q, "answer": answer, "confidence": final_conf, "sources": []})
            st.session_state.is_typing = False
            st.session_state.pending_user_q = ""
            st.rerun()

        conf = compute_confidence(top_chunks)
        context = build_context(top_chunks, per_chunk_chars=PER_CHUNK_CHARS)

        with st.spinner("Generating answer..."):
            try:
                answer, engine = generate_answer(SYSTEM_PROMPT, context, user_q)
            except Exception as e:
                answer = f"Generation error: {type(e).__name__}: {str(e)[:450]}"
                engine = "error"
                conf = {"confidence": 0}

        # ✅ FIX: If answer is Not found, confidence must be 0
        final_conf = int(conf.get("confidence", 0) or 0)
        if is_not_found_answer(answer):
            final_conf = 0

        st.session_state.chat.append(
            {"role": "assistant", "text": answer, "confidence": final_conf, "sources": top_chunks}
        )

        write_jsonl_line(
            history_path,
            {
                "ts": utc_now_iso(),
                "question": user_q,
                "answer": answer,
                "engine": engine,
                "confidence": final_conf,
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
