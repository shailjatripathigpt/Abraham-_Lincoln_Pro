# app.py

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
    st.error("faiss not installed")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("sentence-transformers not installed")
    raise


# =========================
# CONSTANTS
# =========================
OLLAMA_MODEL = "phi3:mini"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 600

TOP_K = 15
FINAL_K = 5
PER_CHUNK_CHARS = 900

# ✅ GOOGLE DRIVE FILE IDS
FAISS_FILE_ID = "1Zvt2fP0ih70dGFXoIvuDX27427wQUYym"
META_FILE_ID = "1bVrE_JFgdK0kdZaaHCBxPItfPda_xxvo"
CHUNKS_FILE_ID = "16eTgJEilBGdH6dmgkoH92bY5N87T7-Wm"

SYSTEM_PROMPT = """You are a strict retrieval-grounded QA assistant.
Use ONLY the provided context.
If not found say:
Not found in provided documents.
"""


# =========================
# GOOGLE DRIVE DOWNLOAD
# =========================
def download_from_drive(file_id: str, dest: Path, label: str):
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {label}...")

    gdown.download(url, str(dest), quiet=False)

    if not dest.exists():
        raise FileNotFoundError(f"{label} failed to download")

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
# JSONL LOADER
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
def load_resources(root: Path):

    emb = root / "embeddings"
    ch = root / "chunks"

    index_path = emb / "chunks.faiss"
    meta_path = emb / "chunks_meta.jsonl"
    chunks_path = ch / "all_chunks.jsonl"

    # ✅ Auto download
    download_from_drive(FAISS_FILE_ID, index_path, "FAISS index")
    download_from_drive(META_FILE_ID, meta_path, "Metadata")
    download_from_drive(CHUNKS_FILE_ID, chunks_path, "Chunks")

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
# OLLAMA
# =========================
def ollama(prompt: str):
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=OLLAMA_TIMEOUT_S,
    )
    return r.json()["response"]


# =========================
# UI
# =========================
def main():

    st.set_page_config(page_title="Lincoln Chatbot", layout="wide")
    st.title("🇺🇸 Abraham Lincoln Chatbot")

    root = find_project_root(Path(__file__).parent)

    try:
        index, meta_rows, chunk_text, emb_model = load_resources(root)
    except Exception as e:
        st.error(str(e))
        st.stop()

    question = st.text_input("Ask something about Lincoln")

    if st.button("Send") and question:

        results = retrieve(index, meta_rows, chunk_text, emb_model, question)

        if not results:
            st.write("Not found in provided documents.")
            return

        context = "\n".join(r["text"] for r in results)

        prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        answer = ollama(prompt)

        st.write("### Answer")
        st.write(answer)


if __name__ == "__main__":
    main()

