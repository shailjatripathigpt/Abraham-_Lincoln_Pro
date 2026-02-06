"""
search_query.py (FIXED PATH AUTO-DETECT)
----------------------------------------
✅ Asks query from user in terminal
✅ Searches FAISS index
✅ Shows TOP-K (default 5)
✅ Prints FULL chunk text

Auto-detects PROJECT_ROOT by searching upward for:
  embeddings/chunks.faiss

Run:
  python search_query.py
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable

import numpy as np

try:
    import faiss
except ImportError as e:
    raise SystemExit("❌ faiss not installed. Run: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("❌ sentence-transformers not installed. Run: pip install sentence-transformers") from e


# =========================
# IO
# =========================
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def load_chunks_text_map(chunks_path: Path) -> Dict[str, str]:
    chunk_text = {}
    for r in iter_jsonl(chunks_path):
        cid = r.get("chunk_id")
        txt = r.get("text")
        if isinstance(cid, str) and cid and isinstance(txt, str):
            chunk_text[cid] = txt
    return chunk_text


# =========================
# PROJECT ROOT AUTO-DETECT
# =========================
def find_project_root(start: Path) -> Path:
    """
    Walk upwards until we find embeddings/chunks.faiss.
    """
    cur = start.resolve()
    for _ in range(12):
        if (cur / "embeddings" / "chunks.faiss").exists():
            return cur
        cur = cur.parent
    return start.resolve()


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    emb_dir = project_root / "embeddings"
    chunks_dir = project_root / "chunks"

    index_path = emb_dir / "chunks.faiss"
    meta_path = emb_dir / "chunks_meta.jsonl"
    chunks_path = chunks_dir / "all_chunks.jsonl"

    print("PROJECT_ROOT :", project_root)
    print("INDEX_PATH   :", index_path)

    if not index_path.exists():
        raise SystemExit(f"❌ FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise SystemExit(f"❌ Meta file not found: {meta_path}")
    if not chunks_path.exists():
        raise SystemExit(f"❌ Chunks file not found: {chunks_path}")

    print("🔹 Loading FAISS index...")
    index = faiss.read_index(str(index_path))

    print("🔹 Loading chunk meta...")
    meta_rows = list(iter_jsonl(meta_path))

    print("🔹 Loading chunk texts...")
    chunk_text = load_chunks_text_map(chunks_path)

    print("🔹 Loading model:", args.model)
    model = SentenceTransformer(args.model)

    print("\n✅ READY. Type query (exit to quit)\n")

    while True:
        query = input("❓ Query: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("👋 Bye.")
            break

        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, args.top_k)

        print("\n================= RESULTS =================")
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            if idx < 0 or idx >= len(meta_rows):
                continue

            m = meta_rows[idx]
            cid = m.get("chunk_id")
            text = chunk_text.get(cid, "")

            print(f"\nRank {rank}")
            print(f"Score      : {score:.4f}")
            print(f"Chunk ID   : {cid}")
            print(f"Doc ID     : {m.get('doc_id')}")
            print(f"Title      : {m.get('title')}")
            print(f"Author     : {m.get('author')}")
            print(f"Year       : {m.get('publish_year')}")
            print(f"Page       : {m.get('page_number')}")
            print(f"Source URL : {m.get('source_url')}")
            print("-" * 70)
            print(text)   # FULL TEXT
            print("-" * 70)

        print("\n===========================================\n")


if __name__ == "__main__":
    main()
