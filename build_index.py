"""Step 1 — Build the RAG index.

Pipeline (every stage resumable):
  1. Map every PDF in pdf_cache/ to its Scopus record (DOI filename match;
     fuzzy title-slug match for no-doi_ files). Tag stream via the
     Stream-B-subset rule.
  2. Extract text per page (pdfplumber, pypdf fallback) -> index/pagecache/.
  3. Chunk ~3600 chars (~900 tokens) with ~600-char overlap, sentence-aware,
     keeping page ranges -> index/chunks.jsonl.
  4. Embed all chunks with gemini-embedding-001 in batches
     -> index/emb_batches/batch_NNNNN.npy (resume = skip existing batches).
  5. Concatenate to index/embeddings.npy (float32, L2-normalized, row i = chunk i).

Run: python build_index.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from rag_common import (INDEX_DIR, PDF_DIR, embed_texts, load_records,
                        match_pdfs, paper_key)

PAGECACHE = INDEX_DIR / "pagecache"
EMB_BATCH_DIR = INDEX_DIR / "emb_batches"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMB_PATH = INDEX_DIR / "embeddings.npy"
PAPERS_PATH = INDEX_DIR / "papers.json"

CHUNK_CHARS = 3600     # ~900 tokens
OVERLAP_CHARS = 600    # ~150 tokens
EMB_BATCH = 24


def safe_name(key: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in key)


# ---------------------------------------------------------------------------
# Page extraction
# ---------------------------------------------------------------------------

def extract_pages(pdf_path: Path) -> list[str]:
    # pypdf first: it follows content-stream order, which keeps two-column
    # papers readable; pdfplumber's line grouping interleaves columns.
    pages = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = [(pg.extract_text() or "") for pg in reader.pages]
        if sum(len(t) for t in pages) > 200:
            return pages
    except Exception as e:  # noqa: BLE001
        print(f"  pypdf failed ({e}); trying pdfplumber", flush=True)
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [(p.extract_text() or "") for p in pdf.pages]
    except Exception as e:  # noqa: BLE001
        print(f"  pdfplumber failed too: {e}", flush=True)
        pages = []
    return pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_SENT_END = (". ", ".\n", "! ", "!\n", "? ", "?\n", ".” ", ".) ")


def chunk_pages(pages: list[str]) -> list[dict]:
    """Return [{text, page_start, page_end}] with sentence-aware boundaries."""
    # Concatenate pages, remembering char offset -> page number
    full, bounds = [], []  # bounds[i] = (start_offset, page_no)
    off = 0
    for i, ptext in enumerate(pages):
        bounds.append((off, i + 1))
        full.append(ptext)
        off += len(ptext) + 1
    text = "\n".join(full)

    def page_of(o: int) -> int:
        pg = 1
        for start, pno in bounds:
            if o >= start:
                pg = pno
            else:
                break
        return pg

    chunks = []
    pos, n = 0, len(text)
    while pos < n:
        end = min(pos + CHUNK_CHARS, n)
        if end < n:
            # try to end at a sentence boundary in the last 25% of the window
            window = text[pos + int(CHUNK_CHARS * 0.75): end]
            best = -1
            for mark in _SENT_END:
                idx = window.rfind(mark)
                if idx > best:
                    best = idx + len(mark)
            if best > 0:
                end = pos + int(CHUNK_CHARS * 0.75) + best
        chunk_text = text[pos:end].strip()
        if len(chunk_text) > 80:  # skip near-empty fragments
            chunks.append({
                "text": chunk_text,
                "page_start": page_of(pos),
                "page_end": page_of(end - 1),
            })
        if end >= n:
            break
        pos = max(end - OVERLAP_CHARS, pos + 1)
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    PAGECACHE.mkdir(exist_ok=True)
    EMB_BATCH_DIR.mkdir(exist_ok=True)

    records = load_records()
    matched, unmatched_pdfs = match_pdfs(records)
    n_ind = sum(1 for r in matched.values() if r["stream"] == "industrial")
    print(f"PDFs in cache: {len(list(PDF_DIR.glob('*.pdf')))}")
    print(f"PDFs matched to CSV records: {len(matched)} "
          f"(industrial {n_ind}, generic {len(matched) - n_ind})")
    if unmatched_pdfs:
        print("UNMATCHED PDFs:")
        for name in unmatched_pdfs:
            print("  -", name)

    papers = {}
    for pdf_name, rec in sorted(matched.items()):
        key = paper_key(rec)
        papers[key] = {"pdf": pdf_name, "doi": rec["doi"], "title": rec["title"],
                       "year": rec["year"], "stream": rec["stream"], "eid": rec["eid"]}
    PAPERS_PATH.write_text(json.dumps(papers, indent=1), encoding="utf-8")

    # ---- Extract + chunk (or reload existing chunks.jsonl) -----------------
    if CHUNKS_PATH.exists():
        all_chunks = [json.loads(l) for l in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]
        print(f"Reusing existing {CHUNKS_PATH.name}: {len(all_chunks)} chunks")
    else:
        all_chunks = []
        empty_papers = []
        for i, (key, meta) in enumerate(sorted(papers.items()), 1):
            cache = PAGECACHE / (safe_name(key) + ".json")
            if cache.exists():
                pages = json.loads(cache.read_text(encoding="utf-8"))
            else:
                print(f"Extracting {i}/{len(papers)}: {meta['pdf']}", flush=True)
                pages = extract_pages(PDF_DIR / meta["pdf"])
                cache.write_text(json.dumps(pages), encoding="utf-8")
            if sum(len(t) for t in pages) < 200:
                empty_papers.append(key)
                continue
            for j, ch in enumerate(chunk_pages(pages)):
                all_chunks.append({
                    "chunk_id": f"{key}::c{j}",
                    "doi": meta["doi"], "title": meta["title"],
                    "stream": meta["stream"], "year": meta["year"],
                    "page_start": ch["page_start"], "page_end": ch["page_end"],
                    "text": ch["text"],
                })
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            for ch in all_chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        if empty_papers:
            print(f"WARNING: {len(empty_papers)} papers produced no text: {empty_papers}")
        print(f"Chunked: {len(all_chunks)} chunks from {len(papers)} papers")

    # ---- Embed --------------------------------------------------------------
    n_batches = (len(all_chunks) + EMB_BATCH - 1) // EMB_BATCH
    for b in range(n_batches):
        out = EMB_BATCH_DIR / f"batch_{b:05d}.npy"
        if out.exists():
            continue
        batch = all_chunks[b * EMB_BATCH:(b + 1) * EMB_BATCH]
        texts = [c["text"] for c in batch]
        print(f"Embedding batch {b + 1}/{n_batches} ({len(texts)} chunks)", flush=True)
        vecs = np.asarray(embed_texts(texts, "RETRIEVAL_DOCUMENT"), dtype=np.float32)
        np.save(out, vecs)

    parts = [np.load(EMB_BATCH_DIR / f"batch_{b:05d}.npy") for b in range(n_batches)]
    emb = np.vstack(parts).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    assert emb.shape == (len(all_chunks), emb.shape[1])
    np.save(EMB_PATH, emb)

    streams = {}
    for c in all_chunks:
        streams[c["stream"]] = streams.get(c["stream"], 0) + 1
    print("\n=== INDEX BUILD SUMMARY ===")
    print(f"Papers indexed:   {len(papers)}")
    print(f"Unmatched PDFs:   {len(unmatched_pdfs)} {unmatched_pdfs if unmatched_pdfs else ''}")
    print(f"Chunks:           {len(all_chunks)} (by stream: {streams})")
    print(f"Embeddings:       {emb.shape} -> {EMB_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
