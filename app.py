import json
from pathlib import Path
import numpy as np
import requests
import streamlit as st
import gdown
import os

import faiss
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

# ✅ WORKING FREE HF MODEL
HF_API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN"))

SYSTEM_PROMPT = """Answer ONLY using the context.
If answer not present say:
Not found in provided documents.
"""


# =========================
# DOWNLOAD HELPERS
# =========================
def download_from_drive(file_id, dest, label):
    if dest.exists():
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner(f"Downloading {label}..."):
        gdown.download(url, str(dest), quiet=False)


# =========================
# JSONL LOADER
# =========================
def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# =========================
# LOAD RESOURCES
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

    emb = model.encode([query]).astype("float32")
    D, I = index.search(emb, FINAL_K)

    results = []
    for idx in I[0]:
        if idx < len(meta):
            cid = meta[idx]["chunk_id"]
            txt = texts.get(cid)
            if txt:
                results.append(txt)

    return results


# =========================
# HF GENERATION
# =========================
def hf_generate(prompt):

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.3,
        },
    }

    r = requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    if r.status_code != 200:
        return f"HF API Error: {r.text}"

    data = r.json()

    if isinstance(data, list):
        return data[0]["generated_text"]

    return str(data)


# =========================
# UI
# =========================
def main():

    st.set_page_config(page_title="Lincoln RAG", layout="wide")
    st.title("🇺🇸 Abraham Lincoln RAG Chatbot")

    if not HF_TOKEN:
        st.error("HF_TOKEN missing in secrets!")
        st.stop()

    index, meta, texts, model = load_resources()

    q = st.text_input("Ask Abraham Lincoln something...")

    if st.button("Send") and q:

        ctx = retrieve(index, meta, texts, model, q)

        if not ctx:
            st.write("Not found in provided documents.")
            return

        context = "\n".join(ctx)

        prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{q}

Answer:
"""

        with st.spinner("Generating answer..."):
            ans = hf_generate(prompt)

        st.write("### Answer")
        st.write(ans)


if __name__ == "__main__":
    main()
