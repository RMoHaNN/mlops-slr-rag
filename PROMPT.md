# Task: Build a corpus-grounded RAG system and answer the research questions from the corpus ONLY

You are working in a folder containing a systematic-literature-review corpus.
Build a Retrieval-Augmented Generation (RAG) system whose answers come
EXCLUSIVELY from this corpus — never from the LLM's pretrained knowledge —
and then use it to answer the two research questions in `screening_config.py`.

## Files you have

- `data/` — two Scopus CSV exports (open with `encoding="utf-8-sig"`;
  columns include Title, Year, Source title, DOI, Abstract, Document Type, EID):
  - `scopus_export_May 25-2026_f2f33b4c-...csv` — Stream A, 268 records (Generic MLOps)
  - `scopus_export_May 25-2026_9234a3e9-...csv` — Stream B, 56 records (Industrial AI).
    Stream B is a topical subset of Stream A: any Stream A record whose DOI or EID
    appears in Stream B has stream = "industrial"; all others "generic".
- `pdf_cache/` — 214 full-text PDFs. Files are named by DOI with `/` replaced
  by `_` (e.g. `10.1109_ACCESS.2023.3311713.pdf`). Files named
  `no-doi_<title-slug>.pdf` belong to papers without a DOI — match them to
  CSV records by fuzzy title comparison with the slug.
- `screening_config.py` — research questions RQ1 and RQ2, stream definitions,
  and 10 coding-field definitions (sanitized: contains no example findings).

## API

Only Google Gemini. Read the key like this:

```python
import re
from pathlib import Path
text = (Path.home()/".config"/"academic-research"/"config.toml").read_text(encoding="utf-8")
gemini_key = re.search(r'\[gemini\].*?api_key\s*=\s*["\']([^"\']+)["\']', text, re.DOTALL).group(1)
```

Use `google-genai` (`from google import genai; from google.genai import types`).
Models: `gemini-embedding-001` for embeddings, `gemini-2.5-flash` for generation
(temperature 0.0, `thinking_config=types.ThinkingConfig(thinking_budget=0)`).
Install what you need: `pip install google-genai pdfplumber pypdf numpy`.
Rate limiting: sleep ~1s between calls; on 429/500/503/quota errors back off
30s → 60s → 120s, up to 5 retries; never crash on a rate limit. Make every
long-running script resumable (skip already-processed items on restart).

## Step 1 — Build the index (`build_index.py`)

1. Map every PDF to its CSV record (DOI from filename; title-slug matching for
   `no-doi_` files). Tag each paper with `stream` (industrial/generic via the
   Stream B subset rule), `doi`, `title`, `year`.
2. Extract text per page (pdfplumber, pypdf fallback). Record page numbers.
3. Chunk: ~900 tokens (~3,600 chars) per chunk, ~150-token overlap, never
   splitting mid-sentence where avoidable. Each chunk keeps: chunk_id, doi,
   title, stream, year, page_start, page_end, text.
4. Embed all chunks with `gemini-embedding-001` (batch where the API allows).
5. Persist: `index/chunks.jsonl` + `index/embeddings.npy` (float32, row i =
   chunk i). Print: papers indexed, papers unmatched (list them), chunk count.

## Step 2 — Grounded query engine (`ask.py`)

`python ask.py "question" [--stream industrial|generic] [--k 12] [--quotes-only]`

1. Embed the question, cosine-similarity against all chunks (NumPy exact
   search is fine at this scale), take top-k (default 12), optional stream filter.
2. Build the generation prompt with the retrieved chunks as NUMBERED excerpts,
   each headed by `[n] DOI — Title (p. X–Y)`.
3. System instruction (grounding contract — enforce all of it):
   - Answer ONLY from the numbered excerpts. The model's own knowledge of the
     literature MUST NOT be used; treat the excerpts as the entire universe of
     available information.
   - Every factual sentence must end with citation(s) like [3] or [3,7].
   - Any direct quotation must be verbatim from an excerpt.
   - If the excerpts do not contain enough information, reply exactly:
     "NOT FOUND IN CORPUS" plus one sentence on what is missing. Never fill
     gaps from prior knowledge.
4. Verification pass (programmatic, after generation):
   - Every cited number must exist in the retrieved set; flag uncited factual
     sentences.
   - Every quoted span (text inside quotation marks, ≥5 words) must appear
     verbatim in the cited chunk (normalize whitespace before comparing).
     If a quote fails, regenerate once with the failure pointed out; if it
     fails again, drop the claim and note it.
   - `--quotes-only` mode: skip synthesis entirely; output the top passages
     verbatim with DOI + page references.
5. Output: the answer, then a "Sources" table mapping citation numbers to
   DOI, title, pages.

## Step 3 — Answer the research questions (`answer_rqs.py`)

RQ1 and RQ2 are too broad for one retrieval round. Use map-reduce over the
corpus so every paper is consulted, not just the top-k of one query:

**Map (per paper):** For each paper, retrieve that paper's own chunks only and
extract with the grounding contract above: (a) operational constraints for
deploying/operating AI in its context, (b) explicitly acknowledged gaps in
current practice — each item with a verbatim supporting quote and page number.
Definitions of these fields are in `screening_config.py`. If a paper discusses
neither, record "none stated". Save per-paper extractions to
`rag_results/paper_extractions.jsonl` (resumable).

**Reduce (RQ1):** Aggregate the per-paper constraint extractions by stream.
Ask the model to group recurring constraints WITHIN each stream (industrial:
46-ish papers; generic: the rest), using only the extracted items as input,
every group citing its supporting DOIs. Then compare the two streams: which
constraint groups appear (a) only in the industrial stream, (b) in both but
materially harder in industrial, (c) equally in both. Every claim must cite
DOIs from the extractions. Write `rag_results/RQ1_answer.md`.

**Reduce (RQ2):** From the cross-stream differences (categories a and b only)
and the per-paper gap extractions, derive design principles for Industrial AI.
Each principle must state: the constraint difference it addresses, the
supporting DOIs with one verbatim quote each, and what to do about it. A
principle with fewer than 2 supporting papers must be labelled "single-source".
Write `rag_results/RQ2_answer.md`.

Both answer files end with a complete reference list (DOI → title, year,
stream) covering every citation used.

## Ground rules

- Never let the model see a question without retrieved excerpts attached.
- Never use the model's prior knowledge of papers, tools, or vendors — if it
  is not in an excerpt, it does not exist.
- Temperature 0.0 everywhere; archive every prompt template in the scripts.
- Print progress (`Processing 12/214: ...`) and end-of-run summaries.
- Verify before reporting: run the citation/quote checks and state the pass
  rate in the final summary.

## Deliverables

1. `build_index.py`, `ask.py`, `answer_rqs.py` — working, resumable.
2. `index/` built; `rag_results/paper_extractions.jsonl`,
   `rag_results/RQ1_answer.md`, `rag_results/RQ2_answer.md` written.
3. A short `README.md` describing the architecture and how to run it.
4. Final chat summary: index stats, verification pass rates, and the headline
   RQ1/RQ2 findings — each with its citation count.
