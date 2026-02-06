# app.py

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional

import numpy as np
import requests
import streamlit as st
import gdown  # ✅ NEW

try:
    import faiss
except ImportError:
    st.error("faiss not installed. Run: pip install faiss-cpu")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("sentence-transformers not installed.")
    raise


# =========================
# CONSTANTS
# =========================
OLLAMA_MODEL = "phi3:mini"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 600
OLLAMA_NUM_CTX = 2048
OLLAMA_NUM_PREDICT = 160

TOP_K = 15
FINAL_K = 5
PER_CHUNK_CHARS = 900
CHAT_HEIGHT_PX = 420

# ✅ GOOGLE DRIVE IDs
FAISS_FILE_ID = "1Zvt2fP0ih70dGFXoIvuDX27427wQUYym"
META_FILE_ID = "1bVrE_JFgdK0kdZaaHCBxPItfPda_xxvo"

SYSTEM_PROMPT = """You are a strict retrieval-grounded QA assistant.

RULES (must follow):
1) Use ONLY the provided CONTEXT.
2) Answer in 1–3 sentences.
3) If not present, reply:
Not found in provided documents.
"""


# =========================
# GOOGLE DRIVE DOWNLOAD
# =========================
def download_from_drive(file_id: str, dest_path: Path, label: str):
    if dest_path.exists():
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {label} from Google Drive...")

    gdown.download(url, str(dest_path), quiet=False)

    if not dest_path.exists():
        raise FileNotFoundError(f"{label} download failed!")

    print(f"{label} ready!")


# =========================
# ROOT DETECT
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(40):
        if (cur / "embeddings").exists():
            return cur
        cur = cur.parent
    return start.resolve()


# =========================
# JSONL HELPERS
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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

    # ✅ Auto-download from Drive
    download_from_drive(FAISS_FILE_ID, index_path, "FAISS index")
    download_from_drive(META_FILE_ID, meta_path, "Metadata")

    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing {chunks_path}")

    index = faiss.read_index(str(index_path))
    meta_rows = list(iter_jsonl(meta_path))
    chunk_text = {r["chunk_id"]: r["text"] for r in iter_jsonl(chunks_path)}

    emb_model = SentenceTransformer(EMBED_MODEL_NAME)

    return index, meta_rows, chunk_text, emb_model


# =========================
# RETRIEVAL
# =========================
def retrieve(index, meta_rows, chunk_text, emb_model, q):

    q_emb = emb_model.encode([q], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, FINAL_K)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(meta_rows):
            m = meta_rows[idx]
            cid = m["chunk_id"]
            results.append({
                "score": float(score),
                "text": chunk_text.get(cid, "")
            })
    return results


# =========================
# OLLAMA CALL
# =========================
def ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    r = requests.post(
        url,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        },
        timeout=OLLAMA_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# =========================
# UI
# =========================
def main():

    st.set_page_config(page_title="Lincoln Chatbot", layout="wide")
    st.title("🇺🇸 Abraham Lincoln Chatbot")

    project_root = find_project_root(Path(__file__).parent)

    try:
        index, meta_rows, chunk_text, emb_model = load_rag_resources(project_root)
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    question = st.text_input("Ask about Abraham Lincoln...")

    if st.button("Send") and question:

        results = retrieve(index, meta_rows, chunk_text, emb_model, question)

        if not results:
            answer = "Not found in provided documents."
        else:
            context = "\n".join(r["text"] for r in results)

            prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
            try:
                answer = ollama_generate(prompt)
            except Exception as e:
                answer = f"Ollama error: {e}"

        st.write("### Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
