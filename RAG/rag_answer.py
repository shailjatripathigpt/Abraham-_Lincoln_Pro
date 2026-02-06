"""
rag_answer.py  (PHI-3 MINI ONLY) - GROUNDED RAG + REAL SOURCES + CONFIDENCE + FILTERING + HISTORY
-----------------------------------------------------------------------------------------------
✅ Uses ONLY phi3:mini (Ollama)
✅ Model is FORBIDDEN to print citations; code prints REAL sources (top-k) from FAISS meta
✅ Confidence score (0–100) computed from retrieval scores + separation + source agreement
✅ Multi-source filtering:
   - include/exclude scan_source
   - include/exclude title keywords
   - include/exclude author keywords
   - year range filtering
✅ Saves query history (question, answer, confidence, sources) to:
   PROJECT_ROOT/rag_history/rag_history.jsonl

Requires:
  pip install faiss-cpu sentence-transformers requests numpy

Files expected:
  PROJECT_ROOT/embeddings/chunks.faiss
  PROJECT_ROOT/embeddings/chunks_meta.jsonl
  PROJECT_ROOT/chunks/all_chunks.jsonl

Run:
  python rag_answer.py
  python rag_answer.py --top_k 5 --show_context
  python rag_answer.py --include_sources "Internet Archive,LOC" --year_min 1800 --year_max 1900
  python rag_answer.py --exclude_sources "Wikipedia JSON Dump" --title_contains "debates"
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional, Tuple
import requests
import numpy as np

try:
    import faiss
except ImportError as e:
    raise SystemExit(" faiss not installed. Run: pip install faiss-cpu") from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(" sentence-transformers not installed. Run: pip install sentence-transformers") from e


# =========================
# CONSTANTS (LOCKED MODEL)
# =========================
OLLAMA_MODEL = "phi3:mini"
OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434"

SYSTEM_PROMPT = """You are a strict retrieval-grounded QA assistant.

RULES (must follow):
1) Use ONLY the provided CONTEXT. Do NOT use outside knowledge.
2) Answer in 1–3 sentences maximum.
3) Do NOT write "CITATIONS:", do NOT mention SOURCE numbers, do NOT mention chunk ids.
4) If the answer is not explicitly present in CONTEXT, reply exactly:
   Not found in provided documents.
"""

# =========================
# PATHS / ROOT DETECT
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(40):
        if (cur / "embeddings" / "chunks.faiss").exists():
            return cur
        cur = cur.parent
    return start.resolve()


# =========================
# TIME / JSONL
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    # robust enough for typical JSONL
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        buf = ""
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # normal
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
                buf = ""
                continue
            except Exception:
                pass
            # multiline recovery
            buf += line
            try:
                obj = json.loads(buf)
                if isinstance(obj, dict):
                    yield obj
                    buf = ""
            except Exception:
                if len(buf) > 5_000_000:
                    buf = ""

def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# CHUNK TEXT MAP
# =========================
def load_chunks_text_map(chunks_path: Path) -> Dict[str, str]:
    """
    chunk_id -> chunk text
    """
    m: Dict[str, str] = {}
    for r in iter_jsonl(chunks_path):
        cid = r.get("chunk_id")
        txt = r.get("text")
        if isinstance(cid, str) and cid and isinstance(txt, str):
            m[cid] = txt
    return m


# =========================
# OLLAMA CALL
# =========================
def ollama_generate(prompt: str, base_url: str, timeout_s: int, num_ctx: int, num_predict: int) -> str:
    url = f"{base_url}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.7,
            "num_ctx": num_ctx,
            "num_predict": num_predict
        }
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


# =========================
# FILTERING
# =========================
def _norm_list_csv(s: Optional[str]) -> List[str]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def _contains_any(hay: str, needles: List[str]) -> bool:
    if not needles:
        return True
    h = (hay or "").lower()
    for n in needles:
        if n.lower() in h:
            return True
    return False

def _contains_none(hay: str, needles: List[str]) -> bool:
    if not needles:
        return True
    h = (hay or "").lower()
    for n in needles:
        if n.lower() in h:
            return False
    return True

def _year_ok(y: Any, y_min: Optional[int], y_max: Optional[int]) -> bool:
    if y is None:
        return True  # allow unknown year unless user wants strict (not needed now)
    try:
        yi = int(y)
    except Exception:
        return True
    if y_min is not None and yi < y_min:
        return False
    if y_max is not None and yi > y_max:
        return False
    return True


# =========================
# CONFIDENCE SCORE (0-100)
# =========================
def compute_confidence(top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Works best when FAISS is cosine via IP with normalized vectors.
    We use:
      - top score level
      - margin between top1 and top2
      - concentration (how many scores are "good")
      - source agreement (same scan_source repeated)
    """
    if not top_chunks:
        return {"confidence": 0, "reason": "no_chunks"}

    scores = np.array([float(c.get("score", 0.0)) for c in top_chunks], dtype=np.float32)
    s1 = float(scores[0])
    s2 = float(scores[1]) if len(scores) > 1 else (s1 - 0.15)

    # Map cosine-ish score [-1..1] -> [0..1] roughly
    # Usually with good models, relevant is ~0.35+ and strong is ~0.55+
    level = np.clip((s1 - 0.20) / 0.55, 0.0, 1.0)  # >0.20 starts becoming meaningful
    margin = np.clip((s1 - s2) / 0.20, 0.0, 1.0)   # big separation => more confident

    # concentration: how many are above a threshold
    good = (scores >= max(0.30, s1 - 0.10)).sum()
    conc = np.clip(good / max(1, len(scores)), 0.0, 1.0)

    # source agreement
    srcs = [str(c.get("scan_source", "") or "") for c in top_chunks]
    srcs = [s for s in srcs if s]
    agree = 0.0
    if srcs:
        # fraction of top-k coming from the most common source
        most = max(srcs.count(x) for x in set(srcs))
        agree = np.clip(most / len(srcs), 0.0, 1.0)

    # weighted blend
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
# CONTEXT BUILDING
# =========================
def truncate_text(s: str, max_chars: int) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ...[TRUNCATED]"

def build_context(top_chunks: List[Dict[str, Any]], per_chunk_chars: int) -> str:
    """
    We tag blocks for the model to read,
    BUT the model is forbidden to print any citations or ids.
    """
    parts = []
    for i, c in enumerate(top_chunks, start=1):
        txt = truncate_text(c.get("text", ""), per_chunk_chars)
        # minimal structured header helps model focus
        parts.append(
            f"SOURCE {i}:\n"
            f"title={c.get('title','Unknown')} | author={c.get('author','Unknown')} | year={c.get('publish_year')} | page={c.get('page_number')}\n"
            f"text={txt}\n"
        )
    return "\n".join(parts)


# =========================
# PRINT REAL SOURCES (BY CODE)
# =========================
def print_sources(top_chunks: List[Dict[str, Any]]) -> None:
    print("\n================= SOURCES (REAL) =================")
    for i, c in enumerate(top_chunks, start=1):
        title = c.get("title") or "Unknown"
        author = c.get("author") or "Unknown"
        page = c.get("page_number")
        url = c.get("source_url") or ""
        chunk_id = c.get("chunk_id") or ""
        score = c.get("score")
        scan_source = c.get("scan_source") or "Unknown"

        print(f"{i}. score={score:.4f} | {scan_source} | {title} | {author} | page={page} | chunk_id={chunk_id}")
        if url:
            print(f"   {url}")
    print("==================================================\n")


# =========================
# HISTORY LOGGER
# =========================
def save_history(history_path: Path, item: Dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8", newline="\n") as f:
        write_jsonl_line(f, item)


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()

    # retrieval / answer
    ap.add_argument("--top_k", type=int, default=5, help="How many chunks to retrieve from FAISS (before filtering re-rank)")
    ap.add_argument("--final_k", type=int, default=5, help="How many chunks to pass to LLM after filtering (default 5)")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="Embedding model used to build FAISS")
    ap.add_argument("--show_context", action="store_true", help="Print retrieved context text before answering")

    # ollama settings (safe for GTX1650)
    ap.add_argument("--timeout", type=int, default=600, help="Ollama HTTP timeout seconds")
    ap.add_argument("--num_ctx", type=int, default=2048, help="Ollama context window")
    ap.add_argument("--per_chunk_chars", type=int, default=900, help="Max chars sent per chunk")
    ap.add_argument("--num_predict", type=int, default=160, help="Max tokens to generate")
    ap.add_argument("--ollama_url", default=OLLAMA_URL_DEFAULT, help="Ollama base url")

    # multi-source filtering
    ap.add_argument("--include_sources", default="", help='Comma list of scan_source allowed (e.g., "Internet Archive,Local Dump")')
    ap.add_argument("--exclude_sources", default="", help='Comma list of scan_source to block')
    ap.add_argument("--title_contains", default="", help='Comma list of keywords; keep sources whose title contains any')
    ap.add_argument("--title_exclude", default="", help='Comma list of keywords; drop sources whose title contains any')
    ap.add_argument("--author_contains", default="", help='Comma list of keywords; keep sources whose author contains any')
    ap.add_argument("--author_exclude", default="", help='Comma list of keywords; drop sources whose author contains any')
    ap.add_argument("--year_min", type=int, default=None, help="Minimum publish_year (inclusive)")
    ap.add_argument("--year_max", type=int, default=None, help="Maximum publish_year (inclusive)")

    # history
    ap.add_argument("--history_name", default="rag_history", help="History file base name (jsonl)")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    emb_dir = project_root / "embeddings"
    chunks_dir = project_root / "chunks"
    history_dir = project_root / "rag_history"
    history_path = history_dir / f"{args.history_name}.jsonl"

    index_path = emb_dir / "chunks.faiss"
    meta_path = emb_dir / "chunks_meta.jsonl"
    chunks_path = chunks_dir / "all_chunks.jsonl"

    if not index_path.exists():
        raise SystemExit(f" FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise SystemExit(f" Meta file not found: {meta_path}")
    if not chunks_path.exists():
        raise SystemExit(f" Chunks file not found: {chunks_path}")

    include_sources = _norm_list_csv(args.include_sources)
    exclude_sources = _norm_list_csv(args.exclude_sources)
    title_contains = _norm_list_csv(args.title_contains)
    title_exclude = _norm_list_csv(args.title_exclude)
    author_contains = _norm_list_csv(args.author_contains)
    author_exclude = _norm_list_csv(args.author_exclude)

    print("PROJECT_ROOT :", project_root)
    print("OLLAMA MODEL :", OLLAMA_MODEL)
    print("FAISS INDEX  :", index_path)
    print("META FILE    :", meta_path)
    print("CHUNKS FILE  :", chunks_path)
    print("HISTORY FILE :", history_path)

    if include_sources:
        print("FILTER include_sources:", include_sources)
    if exclude_sources:
        print("FILTER exclude_sources:", exclude_sources)
    if title_contains:
        print("FILTER title_contains:", title_contains)
    if title_exclude:
        print("FILTER title_exclude:", title_exclude)
    if author_contains:
        print("FILTER author_contains:", author_contains)
    if author_exclude:
        print("FILTER author_exclude:", author_exclude)
    if args.year_min is not None or args.year_max is not None:
        print("FILTER year:", args.year_min, "to", args.year_max)

    print("\n🔹 Loading FAISS index...")
    index = faiss.read_index(str(index_path))

    print("🔹 Loading meta rows...")
    meta_rows = list(iter_jsonl(meta_path))

    print("🔹 Loading chunk text map...")
    chunk_text = load_chunks_text_map(chunks_path)

    print("🔹 Loading embedding model:", args.embed_model)
    emb_model = SentenceTransformer(args.embed_model)

    print("\n✅ RAG READY (Ollama + FAISS). Type query (exit to quit)\n")

    while True:
        question = input(" Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print(" Bye.")
            break

        # Embed query (normalize for cosine/IP)
        q_emb = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        # Search FAISS (overfetch then filter)
        D, I = index.search(q_emb, max(args.top_k, args.final_k))

        # Build candidate chunks
        candidates: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(meta_rows):
                continue
            m = meta_rows[idx]
            cid = m.get("chunk_id")
            candidates.append({
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
                "text": chunk_text.get(cid, "")
            })

        # Apply filters
        filtered: List[Dict[str, Any]] = []
        for c in candidates:
            ss = str(c.get("scan_source") or "")
            title = str(c.get("title") or "")
            author = str(c.get("author") or "")
            year = c.get("publish_year")

            # include/exclude scan_source
            if include_sources and ss not in include_sources:
                continue
            if exclude_sources and ss in exclude_sources:
                continue

            # title/author filters
            if not _contains_any(title, title_contains):
                continue
            if not _contains_none(title, title_exclude):
                continue
            if not _contains_any(author, author_contains):
                continue
            if not _contains_none(author, author_exclude):
                continue

            # year range
            if not _year_ok(year, args.year_min, args.year_max):
                continue

            # must have text
            if not isinstance(c.get("text"), str) or not c["text"].strip():
                continue

            filtered.append(c)

        # Final top chunks
        top_chunks = filtered[:args.final_k]

        if not top_chunks:
            print("\n No relevant chunks found after filtering.\n")
            # log failed query too
            save_history(history_path, {
                "ts": utc_now_iso(),
                "question": question,
                "answer": "Not found in provided documents.",
                "confidence": 0,
                "filters": {
                    "include_sources": include_sources,
                    "exclude_sources": exclude_sources,
                    "title_contains": title_contains,
                    "title_exclude": title_exclude,
                    "author_contains": author_contains,
                    "author_exclude": author_exclude,
                    "year_min": args.year_min,
                    "year_max": args.year_max,
                },
                "sources": []
            })
            continue

        # Optional: show retrieved context
        if args.show_context:
            print("\n================= RETRIEVED CONTEXT (FULL) =================")
            for i, c in enumerate(top_chunks, start=1):
                print(f"\n[SOURCE {i}] score={c['score']:.4f} | {c.get('scan_source')}")
                print(f"title={c.get('title')} | author={c.get('author')} | year={c.get('publish_year')} | page={c.get('page_number')} | chunk_id={c.get('chunk_id')}")
                print("-" * 70)
                print(c.get("text", ""))
                print("-" * 70)
            print("=============================================================\n")

        # Confidence score (before generation)
        conf = compute_confidence(top_chunks)

        # Build prompt
        context = build_context(top_chunks, per_chunk_chars=args.per_chunk_chars)
        prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        # Generate answer
        try:
            answer = ollama_generate(
                prompt=prompt,
                base_url=args.ollama_url,
                timeout_s=args.timeout,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict
            )
        except Exception as e:
            print(f"\n Ollama error: {type(e).__name__}: {str(e)[:200]}\n")
            print("Tips:")
            print(" - Ensure Ollama is running:  ollama serve")
            print(" - Ensure model exists:       ollama pull phi3:mini")
            print(" - Reduce context:            --per_chunk_chars 600 --num_ctx 1024")
            continue

        # Print answer
        print("\n================= ANSWER =================")
        print(answer)
        print("==========================================")

        # Print confidence
        print(f"✅ CONFIDENCE: {conf['confidence']} / 100  (top={conf['top_score']:.4f}, margin={conf['margin']:.2f}, agree={conf['source_agreement']:.2f})")

        # Print real sources
        print_sources(top_chunks)

        # Save history (question + answer + confidence + sources)
        sources_for_history = []
        for c in top_chunks:
            sources_for_history.append({
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
            })

        save_history(history_path, {
            "ts": utc_now_iso(),
            "question": question,
            "answer": answer,
            "confidence": conf.get("confidence", 0),
            "confidence_details": conf,
            "filters": {
                "include_sources": include_sources,
                "exclude_sources": exclude_sources,
                "title_contains": title_contains,
                "title_exclude": title_exclude,
                "author_contains": author_contains,
                "author_exclude": author_exclude,
                "year_min": args.year_min,
                "year_max": args.year_max,
                "top_k": args.top_k,
                "final_k": args.final_k,
            },
            "sources": sources_for_history
        })


if __name__ == "__main__":
    main()
