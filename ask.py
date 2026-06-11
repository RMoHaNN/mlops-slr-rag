"""Step 2 — Grounded query engine.

Usage:
  python ask.py "question" [--stream industrial|generic] [--k 12] [--quotes-only]

Pipeline:
  1. Embed the question (gemini-embedding-001, RETRIEVAL_QUERY), exact cosine
     search over index/embeddings.npy, take top-k (optional stream filter).
  2. Generate with gemini-2.5-flash (temp 0.0, no thinking) under a strict
     grounding contract: answer ONLY from the numbered excerpts, every factual
     sentence cited [n], quotes verbatim, otherwise "NOT FOUND IN CORPUS".
  3. Programmatic verification: cited numbers must exist in the retrieved set;
     uncited factual sentences are flagged; every quoted span (>=5 words) must
     appear verbatim (whitespace-normalized) in a cited chunk. One failed quote
     triggers a single regeneration with the failure pointed out; a second
     failure drops the claim with a note.
  4. Output: answer + Sources table (citation number -> DOI, title, pages).
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from rag_common import embed_texts, generate, load_index, quote_in_text

# --------------------------------------------------------------------------
# Prompt templates (archived here per ground rules)
# --------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You are a literature-grounded assistant for a systematic review corpus.

STRICT GROUNDING CONTRACT — follow every rule:
1. Answer ONLY from the numbered excerpts provided in the user message. Treat
   the excerpts as the ENTIRE universe of available information. Your own
   knowledge of the literature, of tools, vendors, papers, or authors MUST NOT
   be used. If it is not in an excerpt, it does not exist.
2. Every factual sentence in your answer must end with one or more citations
   in square brackets, e.g. [3] or [3,7], referring to excerpt numbers.
3. Any text you place inside quotation marks must be copied VERBATIM from an
   excerpt (the excerpt you cite for that sentence).
4. If the excerpts do not contain enough information to answer, reply exactly:
   NOT FOUND IN CORPUS
   followed by one sentence stating what information is missing. Never fill
   gaps from prior knowledge.
5. Do not cite excerpt numbers that were not provided.
"""

USER_PROMPT_TEMPLATE = """\
Question: {question}

Numbered excerpts from the corpus (your ONLY source of information):

{excerpts}

Answer the question using only these excerpts, following the grounding contract.
"""

REGEN_SUFFIX = """

IMPORTANT — your previous answer failed verification:
{failures}
Regenerate the answer. Either quote the text EXACTLY as it appears in the cited
excerpt, or remove the quotation. Do not introduce new unverifiable quotes.
"""


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def retrieve(question: str, chunks: list[dict], emb: np.ndarray,
             k: int = 12, stream: str | None = None) -> list[tuple[int, dict, float]]:
    q = np.asarray(embed_texts([question], "RETRIEVAL_QUERY")[0], dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-9)
    sims = emb @ q
    if stream:
        mask = np.array([c["stream"] == stream for c in chunks])
        sims = np.where(mask, sims, -1.0)
    top = np.argsort(-sims)[:k]
    return [(int(i), chunks[int(i)], float(sims[int(i)])) for i in top if sims[int(i)] > -1.0]


def format_excerpts(hits) -> str:
    parts = []
    for n, (_, ch, _) in enumerate(hits, 1):
        doi = ch["doi"] or "no-doi"
        parts.append(f"[{n}] {doi} — {ch['title']} (p. {ch['page_start']}–{ch['page_end']})\n"
                     f"{ch['text']}")
    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

_QUOTE_RE = re.compile(r'["“]([^"“”]{20,}?)["”]')
_CIT_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str) -> set[int]:
    out = set()
    for m in _CIT_RE.finditer(answer):
        out.update(int(x) for x in m.group(1).split(","))
    return out


def verify_answer(answer: str, hits) -> dict:
    """Return {invalid_citations, uncited_sentences, failed_quotes}."""
    n_hits = len(hits)
    cited = extract_citations(answer)
    invalid = sorted(c for c in cited if c < 1 or c > n_hits)

    # Uncited factual sentences (heuristic: prose sentences > 40 chars without [n])
    uncited = []
    if "NOT FOUND IN CORPUS" not in answer:
        for line in answer.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("|"):
                continue
            for sent in re.split(r"(?<=[.!?])\s+", line):
                s = sent.strip().rstrip("*").strip()
                if len(s) > 60 and not _CIT_RE.search(s) and not s.endswith(":"):
                    uncited.append(s[:110])

    # Verbatim-quote check: each quoted span >=5 words must appear in a chunk
    # cited in the same sentence (fallback: any retrieved chunk).
    failed_quotes = []
    for sent in re.split(r"(?<=[.!?])\s+", answer):
        for m in _QUOTE_RE.finditer(sent):
            quote = m.group(1)
            if len(quote.split()) < 5:
                continue
            cits = [c for c in extract_citations(sent) if 1 <= c <= n_hits]
            pool = [hits[c - 1][1]["text"] for c in cits] or [h[1]["text"] for h in hits]
            if not any(quote_in_text(quote, t) for t in pool):
                failed_quotes.append(quote[:140])
    return {"invalid_citations": invalid, "uncited_sentences": uncited,
            "failed_quotes": failed_quotes}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def ask(question: str, k: int = 12, stream: str | None = None,
        quotes_only: bool = False) -> str:
    chunks, emb = load_index()
    hits = retrieve(question, chunks, emb, k=k, stream=stream)

    if quotes_only:
        out = [f"Top {len(hits)} passages for: {question}\n"]
        for n, (_, ch, score) in enumerate(hits, 1):
            out.append(f"[{n}] {ch['doi'] or 'no-doi'} — {ch['title']} "
                       f"(p. {ch['page_start']}–{ch['page_end']}, sim {score:.3f})\n"
                       f"{ch['text']}\n")
        return "\n".join(out)

    prompt = USER_PROMPT_TEMPLATE.format(question=question, excerpts=format_excerpts(hits))
    answer = generate(prompt, SYSTEM_INSTRUCTION)
    report = verify_answer(answer, hits)

    if report["failed_quotes"] or report["invalid_citations"]:
        failures = []
        for q in report["failed_quotes"]:
            failures.append(f'- quote not found verbatim in cited excerpt: "{q}..."')
        for c in report["invalid_citations"]:
            failures.append(f"- citation [{c}] does not exist in the excerpt list")
        answer = generate(prompt + REGEN_SUFFIX.format(failures="\n".join(failures)),
                          SYSTEM_INSTRUCTION)
        report = verify_answer(answer, hits)
        if report["failed_quotes"]:
            answer += ("\n\n> NOTE (verification): the following quoted claims could not "
                       "be verified verbatim against the corpus and should be discounted:\n"
                       + "\n".join(f'> - "{q}..."' for q in report["failed_quotes"]))

    # Sources table
    lines = ["", "Sources", "-------",
             "| # | DOI | Title | Pages |", "|---|-----|-------|-------|"]
    for n, (_, ch, _) in enumerate(hits, 1):
        lines.append(f"| {n} | {ch['doi'] or 'no-doi'} | {ch['title'][:80]} "
                     f"| {ch['page_start']}–{ch['page_end']} |")
    verif = (f"\nVerification: invalid citations: {len(report['invalid_citations'])}; "
             f"unverified quotes: {len(report['failed_quotes'])}; "
             f"uncited factual sentences flagged: {len(report['uncited_sentences'])}")
    return answer + "\n" + "\n".join(lines) + "\n" + verif


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus-grounded question answering")
    ap.add_argument("question")
    ap.add_argument("--stream", choices=["industrial", "generic"], default=None)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--quotes-only", action="store_true")
    args = ap.parse_args()
    print(ask(args.question, k=args.k, stream=args.stream, quotes_only=args.quotes_only))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
