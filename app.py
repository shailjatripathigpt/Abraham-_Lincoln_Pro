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

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

SYSTEM_PROMPT = """Answer ONLY using the context.
If answer not present say:
Not found in provided documents.
"""


# =========================
# DOWNLOAD FROM DRIVE
# =========================
def download_from_drive(fid, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(f"https://drive.google.com/uc?id={fid}", str(dest), quiet=False)


# =========================
# LOAD RESOURCES
# =========================
@st.cache_resource
def load_resources():

    emb = Path("embeddings")
    ch = Path("chunks")

    index_path = emb / "chunks.faiss"
    meta_path = emb / "chunks_meta.jsonl"
    chunks_path = ch / "all_chunks.jsonl"

    download_from_drive(FAISS_FILE_ID, index_path)
    download_from_drive(META_FILE_ID, meta_path)
    download_from_drive(CHUNKS_FILE_ID, chunks_path)

    index = faiss.read_index(str(index_path))

    meta = [json.loads(l) for l in open(meta_path)]
    texts = {r["chunk_id"]: r["text"] for r in map(json.loads, open(chunks_path))}

    model = SentenceTransformer(EMBED_MODEL_NAME)

    return index, meta, texts, model


# =========================
# RETRIEVE
# =========================
def retrieve(index, meta, texts, model, q):

    emb = model.encode([q]).astype("float32")
    _, I = index.search(emb, FINAL_K)

    out = []
    for idx in I[0]:
        if idx < len(meta):
            cid = meta[idx]["chunk_id"]
            txt = texts.get(cid)
            if txt:
                out.append(txt)

    return out


# =========================
# GROQ GENERATION
# =========================
def groq_generate(prompt):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",  # fast & free
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    r = requests.post(url, headers=headers, json=payload)

    return r.json()["choices"][0]["message"]["content"]


# =========================
# UI
# =========================
def main():

    st.set_page_config(page_title="Lincoln RAG", layout="wide")
    st.title("🇺🇸 Abraham Lincoln RAG Chatbot (Groq Powered)")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing in secrets!")
        st.stop()

    index, meta, texts, model = load_resources()

    q = st.text_input("Ask Abraham Lincoln something...")

    if st.button("Send") and q:

        ctx = retrieve(index, meta, texts, model, q)

        if not ctx:
            st.write("Not found in provided documents.")
            return

        context = "\n".join(ctx)

        prompt = f"""
Context:
{context}

Question:
{q}

Answer:
"""

        with st.spinner("Generating answer..."):
            ans = groq_generate(prompt)

        st.write("### Answer")
        st.write(ans)


if __name__ == "__main__":
    main()
