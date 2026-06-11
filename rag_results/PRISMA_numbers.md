# PRISMA 2020 flow — corpus-grounded RAG analysis

All figures computed directly from `data/*.csv`, `index/papers.json`,
`index/chunks.jsonl`, `rag_results/paper_extractions.jsonl`,
`rag_results/RQ1_answer.md`, and `rag_results/RQ2_answer.md`.

**Design statement.** No title/abstract relevance screening was applied; the
analysis consulted every retrievable full text from the search exports, with
relevance enforced at extraction time through corpus-grounded,
citation-verified prompting.

## Identification

| | n |
|---|---|
| Records identified — Scopus export, Stream A (Generic MLOps) | 268 |
| Records identified — Scopus export, Stream B (Industrial AI) | 56 |
| Duplicates across exports (Stream B is a full subset of Stream A, by EID) | 56 |
| **Unique records after deduplication** | **268** |
| — tagged industrial (Stream-B subset rule) | 56 |
| — tagged generic | 212 |

Internal duplicates within each export: 0 (268 and 56 unique EIDs respectively).

## Retrieval

| | n |
|---|---|
| Reports sought | 268 |
| Reports not retrieved | 59 |
| **Reports retrieved (full text)** | **209** (268 − 59 = 209) |
| — industrial | 48 / 56 (85.7%) |
| — generic | 161 / 212 (75.9%) |

Document types of the 59 reports not retrieved:
Conference review 20 · Conference paper 14 · Book 10 · Article 8 ·
Book chapter 6 · Review 1.

(`pdf_cache/` holds 214 PDF files; 5 are duplicate copies of another file and
were not indexed separately, leaving 209 unique full-text documents.)

## Analysis

| | n |
|---|---|
| Full texts analyzed (chunked, embedded, consulted in map phase) | 209 |
| — chunks indexed | 4,660 |
| Papers yielding zero verified evidence | 2 |
| **Papers contributing verified constraint/gap evidence** | **207** (209 − 2 = 207) |
| Papers cited in `RQ1_answer.md` | 203 (industrial 47, generic 156) |
| Papers cited in `RQ2_answer.md` | 20 (industrial 15, generic 5) |

The two zero-evidence papers:
- `10.1016/j.procs.2024.01.168` (industrial) — full text analyzed; the
  extraction model found no operational-constraint or gap statements to
  extract ("none stated").
- `10.1109/SEAA60479.2023.00023` (generic) — scanned, image-only PDF; no
  machine-extractable text (both available copies are image-only).

## Provenance

The two Scopus exports are dated 2026-05-25; the search strings are documented
in the originating review. This analysis applied no relevance screening by
design — every retrievable full text was analyzed, and relevance was enforced
at extraction time by the corpus-grounded, citation-verified prompting
described in `README.md`.
