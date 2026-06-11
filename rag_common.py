"""Shared utilities for the corpus-grounded RAG system.

- Gemini client construction (key read from ~/.config/academic-research/config.toml)
- Rate-limited, retrying wrappers around embedding and generation calls
- Text normalization used by the quote-verification pass
- CSV loading / stream tagging / PDF<->record matching helpers
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdf_cache"
INDEX_DIR = ROOT / "index"
RESULTS_DIR = ROOT / "rag_results"

STREAM_A_CSV = DATA_DIR / "scopus_export_May 25-2026_f2f33b4c-1651-4a3d-8536-e4662d3266f5.csv"
STREAM_B_CSV = DATA_DIR / "scopus_export_May 25-2026_9234a3e9-cbf1-472f-8b80-8eb18be4e56f.csv"

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
EMBED_DIM = 768  # output_dimensionality (vectors L2-normalized by us)

# ---------------------------------------------------------------------------
# Gemini client + retry
# ---------------------------------------------------------------------------

def load_gemini_key() -> str:
    text = (Path.home() / ".config" / "academic-research" / "config.toml").read_text(encoding="utf-8")
    return re.search(r'\[gemini\].*?api_key\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL).group(1)


_client = None

def get_client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=load_gemini_key())
    return _client


_RETRYABLE = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL",
              "quota", "overload", "Deadline", "timeout", "timed out")
_BACKOFFS = [30, 60, 120, 120, 120]  # up to 5 retries


def _with_retries(fn, what: str):
    for attempt in range(len(_BACKOFFS) + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            retryable = any(t.lower() in msg.lower() for t in _RETRYABLE)
            if not retryable or attempt >= len(_BACKOFFS):
                raise
            wait = _BACKOFFS[attempt]
            print(f"  [retry] {what}: {msg[:140]} -> sleeping {wait}s "
                  f"(attempt {attempt + 1}/{len(_BACKOFFS)})", flush=True)
            time.sleep(wait)


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed a batch of texts. task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY."""
    from google.genai import types
    client = get_client()

    def call():
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type,
                                            output_dimensionality=EMBED_DIM),
        )
        return [e.values for e in resp.embeddings]

    out = _with_retries(call, f"embed x{len(texts)}")
    time.sleep(1.0)
    return out


def generate(prompt: str, system_instruction: str, json_mode: bool = False) -> str:
    """Grounded generation: temperature 0.0, thinking disabled."""
    from google.genai import types
    client = get_client()
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=system_instruction,
        response_mime_type="application/json" if json_mode else None,
    )

    def call():
        resp = client.models.generate_content(model=GEN_MODEL, contents=prompt, config=cfg)
        if resp.text is None:
            raise RuntimeError(f"empty response (finish: {resp.candidates[0].finish_reason if resp.candidates else 'none'})")
        return resp.text

    out = _with_retries(call, "generate")
    time.sleep(1.0)
    return out


# ---------------------------------------------------------------------------
# Normalization for verbatim-quote checking
# ---------------------------------------------------------------------------

_LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "­": "", "‐": "-", "‑": "-"}


def normalize_for_match(s: str) -> str:
    """Whitespace/ligature/quote-mark normalization so PDF-extraction artifacts
    do not break verbatim comparison. Case-folded; hyphen-linebreaks healed."""
    s = unicodedata.normalize("NFKC", s)
    for k, v in _LIG.items():
        s = s.replace(k, v)
    s = s.replace("-\n", "").replace("- \n", "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    # drop residual soft hyphenation artifacts like "word- breaks"
    s = s.replace("- ", "")
    s = s.replace("-", "")  # hyphenation at line breaks is unrecoverable; ignore hyphens entirely
    return s


def quote_in_text(quote: str, text: str) -> bool:
    """True if `quote` appears in `text` verbatim after normalization, OR as an
    in-order word subsequence with >=90% coverage inside a tight window.

    The fallback exists because pdfplumber interleaves columns of two-column
    PDFs: a sentence in the source may have fragments of the neighbouring
    column injected mid-stream, breaking contiguous substring matching even
    though the quote is genuinely verbatim on the page. Requiring nearly all
    quote words in order within a window ~3x the quote length still rules out
    fabricated quotes while tolerating that extraction artifact.
    """
    nq, nt = normalize_for_match(quote), normalize_for_match(text)
    if nq in nt:
        return True
    # Letters-only comparison: immune to spacing pathologies of PDF extraction
    # (stray spaces inside words, or all spaces missing). The exact character
    # sequence of a >=5-word quote still cannot be fabricated.
    lq = re.sub(r"[^a-z0-9]", "", nq)
    if len(lq) >= 25 and lq in re.sub(r"[^a-z0-9]", "", nt):
        return True
    qw, tw = nq.split(), nt.split()
    if len(qw) < 5 or not tw:
        return False
    window = max(len(qw) * 3, 30)
    anchors = [i for i, w in enumerate(tw) if w == qw[0] or (len(qw) > 1 and w == qw[1])]
    for a in anchors:
        seg = tw[a:a + window]
        qi = 0
        for w in seg:
            if qi < len(qw) and w == qw[qi]:
                qi += 1
        if qi / len(qw) >= 0.90:
            return True
    return False


# ---------------------------------------------------------------------------
# Corpus loading / matching
# ---------------------------------------------------------------------------

def slugify(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s


def load_records() -> list[dict]:
    """Stream A records tagged with stream = industrial/generic via Stream B subset rule."""
    with open(STREAM_B_CSV, encoding="utf-8-sig") as f:
        b_rows = list(csv.DictReader(f))
    b_dois = {r["DOI"].strip().lower() for r in b_rows if r.get("DOI", "").strip()}
    b_eids = {r["EID"].strip() for r in b_rows if r.get("EID", "").strip()}

    with open(STREAM_A_CSV, encoding="utf-8-sig") as f:
        a_rows = list(csv.DictReader(f))

    records = []
    for r in a_rows:
        doi = r.get("DOI", "").strip()
        eid = r.get("EID", "").strip()
        industrial = (doi and doi.lower() in b_dois) or (eid and eid in b_eids)
        records.append({
            "doi": doi,
            "eid": eid,
            "title": r.get("Title", "").strip(),
            "year": r.get("Year", "").strip(),
            "source": r.get("Source title", "").strip(),
            "doc_type": r.get("Document Type", "").strip(),
            "stream": "industrial" if industrial else "generic",
        })
    return records


# Anomalously named PDFs, resolved by first-page title inspection (see README).
# Duplicates of an already DOI-/slug-named PDF are skipped; manual matches map
# filename -> DOI of the CSV record.
DUPLICATE_PDFS = {
    "Advancing_MLOps_from_Ad_hoc_to_Kaizen.pdf",      # = 10.1109_SEAA60479.2023.00023.pdf
    "electronics-12-03940-v2.pdf",                    # = 10.3390_electronics12183940.pdf
    "EKVU4MPS.pdf",   # = no-doi_streamlining-the-operation-of-ai-systems-...
    "K5E4FS98.pdf",   # = no-doi_explainable-mlops-a-methodological-framework-...
    "QSIIHD6D.pdf",   # = no-doi_towards-machine-learning-based-digital-twins-...
}
MANUAL_MATCHES = {
    "9781003454663_previewpdf.pdf": "10.1201/9781003454663",  # AI/ML for Healthcare (book preview)
}


def match_pdfs(records: list[dict]) -> tuple[dict, list[str]]:
    """Return ({pdf_filename: record}, [unmatched pdf filenames])."""
    pdfs = sorted(p.name for p in PDF_DIR.glob("*.pdf") if p.name not in DUPLICATE_PDFS)
    by_doi = {r["doi"].lower(): r for r in records if r["doi"]}
    by_doiname = {}
    for rec in records:
        if rec["doi"]:
            by_doiname[(rec["doi"].replace("/", "_") + ".pdf").lower()] = rec

    matched, unmatched = {}, []
    nodoi_recs = [r for r in records if not r["doi"]]
    for name in pdfs:
        if name.lower() in by_doiname:
            matched[name] = by_doiname[name.lower()]
            continue
        if name in MANUAL_MATCHES and MANUAL_MATCHES[name].lower() in by_doi:
            matched[name] = by_doi[MANUAL_MATCHES[name].lower()]
            continue
        if name.startswith("no-doi_"):
            slug_part = name[len("no-doi_"):-len(".pdf")]
            best, best_score = None, 0.0
            for rec in nodoi_recs + [r for r in records if r["doi"]]:
                rs = slugify(rec["title"])[: len(slug_part)]
                score = difflib.SequenceMatcher(None, slug_part, rs).ratio()
                if score > best_score:
                    best, best_score = rec, score
            if best is not None and best_score >= 0.75:
                matched[name] = best
                continue
        unmatched.append(name)
    return matched, unmatched


def paper_key(rec: dict) -> str:
    """Stable per-paper key: DOI if present, else slug of title."""
    return rec["doi"].lower() if rec["doi"] else "no-doi:" + slugify(rec["title"])[:60]


# ---------------------------------------------------------------------------
# Index loading (used by ask.py and answer_rqs.py)
# ---------------------------------------------------------------------------

def load_index():
    import numpy as np
    chunks = []
    with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    emb = np.load(INDEX_DIR / "embeddings.npy")
    assert emb.shape[0] == len(chunks), f"{emb.shape[0]} embeddings vs {len(chunks)} chunks"
    return chunks, emb
