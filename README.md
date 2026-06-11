# Corpus-Grounded RAG Synthesis — MLOps / Industrial AI SLR

Companion repository for a systematic-literature-review synthesis that answers
two research questions using Retrieval-Augmented Generation whose answers come
**exclusively from the review corpus** — never from the LLM's pretrained
knowledge — with programmatic, string-level verification of every citation and
quote.

**RQ1.** What operational constraints distinguish Industrial AI implementation
(manufacturing/maintenance environments) from Generic AI implementation
(standard software-centric ML deployment contexts)?

**RQ2.** What design principles for Industrial AI can be identified from the
gaps between the two literatures?

The corpus is 268 Scopus records (two exports dated 2026-05-25; the
56-record Industrial stream is a subset of the 268-record Generic-MLOps
stream), of which 209 full texts were retrievable and analyzed. No
title/abstract relevance screening was applied; the analysis consulted every
retrievable full text, with relevance enforced at extraction time through
corpus-grounded, citation-verified prompting. See
`rag_results/PRISMA_numbers.md` for the full flow.

## Repository structure

```
data/                          Two Scopus exports, metadata-only (Title, Year,
                               Source title, DOI, Document Type, EID; abstracts
                               removed for licensing). encoding="utf-8-sig"
screening_config.py            Research questions, stream definitions, coding fields
rag_common.py                  Shared utilities: Gemini client, retry, quote
                               verification, CSV/PDF matching
build_index.py                 Step 1 — extract, chunk (~900 tokens, 150 overlap,
                               page-tracked), embed (gemini-embedding-001, 768-d)
ask.py                         Step 2 — grounded Q&A: top-k cosine retrieval,
                               grounding contract, citation/quote verification
answer_rqs.py                  Step 3 — map-reduce over every paper: per-paper
                               constraint/gap extraction, per-stream grouping,
                               cross-stream comparison (RQ1), design principles (RQ2)
rag_results/
  paper_extractions.jsonl      Per-paper evidence: every item has a verbatim quote,
                               page range, and stream tag
  RQ1_answer.md                Constraint groups per stream + cross-stream comparison
  RQ2_answer.md                Design principles with verbatim supporting quotes
  PRISMA_numbers.md            PRISMA 2020 flow for this analysis
PROMPT.md                      The build specification this repository implements
```

Not included (copyright): the 209 source PDFs and the derived full-text index.

## Grounding and verification (what "verified" means here)

- The model never sees a question without retrieved excerpts attached, and is
  instructed to treat the excerpts as the entire universe of information,
  answer `NOT FOUND IN CORPUS` when they are insufficient, cite excerpt
  numbers on every factual sentence, and quote only verbatim. Temperature 0.0,
  thinking disabled. All prompt templates are archived in the scripts.
- After every generation, code (not the LLM) checks that each cited excerpt
  exists, each cited DOI is in the corpus and in the claimed stream, and each
  quoted span ≥ 5 words appears in the cited source text
  (whitespace/ligature/hyphenation-normalized). A failed quote triggers one
  regeneration; a second failure drops the claim and notes it.
- This run: 96.9% of 2,730 extracted evidence items verbatim-verified (the
  3.1% that failed were dropped, not used); a 250-item random re-audit passed
  100%; 0 invalid and 0 wrong-stream citations in the final answers.

## Reproducibility

**Tier 1 — verify the claims (no API, no PDFs needed).**
Every claim in `RQ1_answer.md` and `RQ2_answer.md` traces to
`rag_results/paper_extractions.jsonl`, where each evidence item carries the
paper's DOI, a page range, and a verbatim quote. To check any claim: find the
cited DOI in the JSONL, read the quote, and confirm it against the published
paper at that DOI and page. No part of the pipeline needs to be re-run.

**Tier 2 — full replication.**
1. Obtain the 209 full texts by DOI (DOIs are in `data/*.csv` and in the
   reference lists of both answer files).
2. Place them in `pdf_cache/`, named by the convention in `rag_common.py`:
   `DOI.replace("/", "_") + ".pdf"` (e.g. `10.1109_ACCESS.2023.3311713.pdf`);
   for papers without a DOI, `no-doi_<title-slug>.pdf`.
3. Configure the Gemini API key (see below), then:

```bash
pip install google-genai pdfplumber pypdf numpy
python build_index.py     # extract -> chunk -> embed (resumable at every stage)
python answer_rqs.py      # map (resumable) + reduce -> rag_results/
python ask.py "your question" [--stream industrial|generic] [--k 12] [--quotes-only]
```

Generation is deterministic in configuration (temperature 0.0, fixed prompts,
fixed chunking); minor output variation across model versions is possible.

### Gemini API key

The scripts read the key from `~/.config/academic-research/config.toml`:

```toml
[gemini]
api_key = "YOUR_KEY"
```

Models used: `gemini-embedding-001` (embeddings, 768-d) and `gemini-2.5-flash`
(generation, temperature 0.0, thinking disabled). Rate limits are handled with
1 s pacing and 30→60→120 s backoff; all long-running stages resume after
interruption.
