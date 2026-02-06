# app.py
"""
Streamlit UI for your RAG (FAISS + Ollama phi3:mini) with:
✅ Ask question -> Top chunks -> Answer
✅ Confidence score
✅ Saves history

+ NEW:
✅ Auto-download FAISS from Google Drive if missing
"""

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
    st.error("faiss not installed.")
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

TOP_K = 15
FINAL_K = 5
PER_CHUNK_CHARS = 900

CHAT_HEIGHT_PX = 420

# ✅ GOOGLE DRIVE FAISS
FAISS_FILE_ID = "1Zvt2fP0ih70dGFXoIvuDX27427wQUYym"

SYSTEM_PROMPT = """You are a strict retrieval-grounded QA assistant.

RULES:
Use ONLY the CONTEXT.
If not found say:
Not found in provided documents.
"""


# =========================
# GOOGLE DRIVE DOWNLOAD
# =========================
def ensure_faiss_downloaded(index_path: Path):
    if index_path.exists():
        return

    index_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={FAISS_FILE_ID}"
    print("Downloading FAISS from Google Drive...")

    gdown.download(url, str(index_path), quiet=False)

    if not index_path.exists():
        raise FileNotFoundError("FAISS download failed!")

    print("FAISS ready!")


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

    # ✅ Auto-download FAISS
    ensure_faiss_downloaded(index_path)

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)

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

    out = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(meta_rows):
            m = meta_rows[idx]
            cid = m["chunk_id"]
            out.append({
                "score": float(score),
                "text": chunk_text.get(cid, "")
            })
    return out


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

    question = st.text_input("Ask something about Lincoln...")

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

            r = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=300,
            )

            answer = r.json()["response"]

        st.write("### Answer")
        st.write(answer)


if __name__ == "__main__":
    main()
