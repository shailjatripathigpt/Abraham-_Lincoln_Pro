import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import requests
import streamlit as st
import gdown
import os

try:
    import faiss
except:
    st.error("faiss-cpu not installed")
    raise

from sentence_transformers import SentenceTransformer


# =========================
# CONFIG
# =========================
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
FINAL_K = 5

# Google Drive IDs
FAISS_FILE_ID = "1Zvt2fP0ih70dGFXoIvuDX27427wQUYym"
META_FILE_ID = "1bVrE_JFgdK0kdZaaHCBxPItfPda_xxvo"
CHUNKS_FILE_ID = "16eTgJEilBGdH6dmgkoH92bY5N87T7-Wm"

# HuggingFace API
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/Phi-3-mini-4k-instruct"
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

SYSTEM_PROMPT = """You are a strict retrieval QA assistant.
Answer ONLY from context.
If not found say:
Not found in provided documents.
"""


# =========================
# DOWNLOAD HELPERS
# =========================
def download_from_drive(file_id: str, dest: Path, label: str):
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner(f"Downloading {label}..."):
        gdown.download(url, str(dest), quiet=False)

    if not dest.exists():
        raise FileNotFoundError(f"{label} failed to download")


# =========================
# JSONL LOADER
# =========================
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# =========================
# LOAD RAG
# =========================
@st.cache_resource
def load_resources():

    root = Path(".")
    emb = root / "embeddings"
    ch = root / "chunks"

    index_path = emb / "chunks.faiss"
    meta_path = emb / "chunks_meta.jsonl"
    chunks_path = ch / "all_chunks.jsonl"

    download_from_drive(FAISS_FILE_ID, index_path, "FAISS")
    download_from_drive(META_FILE_ID, meta_path, "Metadata")
    download_from_drive(CHUNKS_FILE_ID, chunks_path, "Chunks")

    index = faiss.read_index(str(index_path))
    meta = list(iter_jsonl(meta_path))
    texts = {r["chunk_id"]: r["text"] for r in iter_jsonl(chunks_path)}

    model = SentenceTransformer(EMBED_MODEL_NAME)

    return index, meta, texts, model


# =========================
# RETRIEVE
# =========================
def retrieve(index, meta, texts, model, query):

    emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(emb, FINAL_K)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(meta):
            cid = meta[idx]["chunk_id"]
            results.append(texts.get(cid, ""))

    return results


# =========================
# HF GENERATION
# =========================
def hf_generate(prompt):

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.2,
            "return_full_text": False,
        },
    }

    r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=120)

    if r.status_code != 200:
        return f"HF API Error: {r.text}"

    data = r.json()

    if isinstance(data, list):
        return data[0]["generated_text"]

    return "No response."


# =========================
# UI
# =========================
def main():

    st.set_page_config(page_title="Lincoln RAG", layout="wide")
    st.title("🇺🇸 Abraham Lincoln RAG Chatbot")

    if not HF_TOKEN:
        st.error("HF token missing in secrets!")
        st.stop()

    index, meta, texts, model = load_resources()

    q = st.text_input("Ask Abraham Lincoln something...")

    if st.button("Send") and q:

        ctx_chunks = retrieve(index, meta, texts, model, q)

        if not ctx_chunks:
            st.write("Not found in provided documents.")
            return

        context = "\n".join(ctx_chunks)

        prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{q}

ANSWER:
"""

        with st.spinner("Generating..."):
            ans = hf_generate(prompt)

        st.write("### Answer")
        st.write(ans)


if __name__ == "__main__":
    main()
