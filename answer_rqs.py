"""Step 3 — Answer RQ1 and RQ2 with map-reduce over the whole corpus.

MAP   (per paper): feed the paper's own chunks (all of them; for very long
      papers the chunks most similar to the RQ topics) to gemini-2.5-flash
      under the grounding contract and extract (a) operational constraints,
      (b) explicitly acknowledged gaps — each with a verbatim quote tied to an
      excerpt number, mapped back to page numbers programmatically. Quotes are
      verified verbatim (whitespace-normalized) against the cited chunk; one
      retry on failure, then the item is dropped and noted.
      -> rag_results/paper_extractions.jsonl  (resumable)

REDUCE RQ1: group recurring constraints WITHIN each stream (model input =
      extracted items only), then compare streams into categories
      (a) industrial-only, (b) both-but-harder-in-industrial, (c) equal.
      Every DOI cited is checked against the extraction set.
      -> rag_results/RQ1_answer.md

REDUCE RQ2: from categories (a)+(b) and the per-paper gap extractions, derive
      design principles, each with supporting DOIs + one verified verbatim
      quote each; <2 supporting papers => labelled "single-source".
      -> rag_results/RQ2_answer.md

Run: python answer_rqs.py [--map-only]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from rag_common import (INDEX_DIR, RESULTS_DIR, embed_texts, generate,
                        load_index, quote_in_text)

EXTRACTIONS_PATH = RESULTS_DIR / "paper_extractions.jsonl"
RQ1_PATH = RESULTS_DIR / "RQ1_answer.md"
RQ2_PATH = RESULTS_DIR / "RQ2_answer.md"
MAX_CHUNKS_PER_PAPER = 45  # very long papers (books): keep the most RQ-relevant chunks

RQ1 = ("What operational constraints distinguish Industrial AI implementation "
       "(manufacturing/maintenance environments) from Generic AI implementation "
       "(standard software-centric ML deployment contexts)?")
RQ2 = ("What design principles for Industrial AI can be identified from the "
       "gaps between the two literatures?")

# --------------------------------------------------------------------------
# Prompt templates (archived per ground rules)
# --------------------------------------------------------------------------

MAP_SYSTEM = """\
You are an extraction engine for a systematic literature review.

STRICT GROUNDING CONTRACT:
1. Use ONLY the numbered excerpts provided. They are the entire universe of
   available information. Never use your own knowledge of the literature,
   tools, or vendors.
2. Every extracted item must include a supporting quote copied VERBATIM from
   one excerpt, and the number of that excerpt.
3. If the paper states nothing for a field, return an empty list for it.
   Never invent content.
Return ONLY valid JSON, no markdown fences.
"""

MAP_PROMPT = """\
The excerpts below all come from ONE paper:
  DOI: {doi}
  Title: {title}

Extract two fields:

1. "operational_constraints": constraints and challenges the paper identifies
   for deploying or operating AI/ML in its specific context (technical,
   organizational, environmental, regulatory, data-related, ...). Concrete
   constraints only, stated by THIS paper.
2. "gaps_identified": gaps in current MLOps/AI-operations practice that the
   paper EXPLICITLY acknowledges — limitations, unsolved problems, or
   statements about what current tools/frameworks/processes cannot handle.

Return JSON exactly in this shape:
{{
  "operational_constraints": [
    {{"item": "<one-sentence constraint, close to the paper's wording>",
      "quote": "<verbatim supporting quote from one excerpt, 10-40 words>",
      "excerpt": <excerpt number>}}
  ],
  "gaps_identified": [
    {{"item": "<one-sentence gap>",
      "quote": "<verbatim supporting quote>",
      "excerpt": <excerpt number>}}
  ]
}}
Use [] for a field the paper does not address. Maximum 8 items per field —
pick the most significant. Quotes must be copied character-for-character.

Numbered excerpts:

{excerpts}
"""

MAP_RETRY_SUFFIX = """

IMPORTANT — in your previous attempt these quotes were NOT verbatim copies of
the cited excerpt and failed verification:
{failures}
Re-extract. Copy quotes character-for-character from the excerpt text, or drop
the item if no verbatim support exists.
"""

GROUP_SYSTEM = """\
You are an aggregation engine for a systematic literature review. You receive
constraint statements extracted from papers (each tagged with the paper's DOI).
These statements are your ONLY source of information — do not use any prior
knowledge of the literature. Every group you output must list the DOIs of the
papers whose statements support it, and only DOIs that appear in the input.
Return ONLY valid JSON, no markdown fences.
"""

GROUP_PROMPT = """\
Below are operational-constraint statements extracted from the {label} stream
of the corpus ({stream_def}).

Group recurring constraints into 8-15 thematic groups. Do not merge
constraints with different root causes into one broad group merely because
they co-occur in the same papers. For each group give:
- "group": short name (3-8 words)
- "description": 1-2 sentences summarizing the constraint, synthesized ONLY
  from the statements below
- "dois": every DOI whose statement(s) support this group

Statements that fit no group go to a final group named "Other". Return JSON:
{{"groups": [{{"group": "...", "description": "...", "dois": ["..."]}}]}}

Constraint statements ({n} items):

{items}
"""

COMPARE_SYSTEM = GROUP_SYSTEM

COMPARE_PROMPT = """\
Research question: {rq1}

You are given constraint GROUPS derived from two literature streams.
INDUSTRIAL stream = {ind_def}
GENERIC stream = {gen_def}

INDUSTRIAL groups:
{ind_groups}

GENERIC groups:
{gen_groups}

Compare the streams. Classify every industrial constraint group into exactly
one category:
(a) "industrial_only" — no materially similar group exists in the generic stream
(b) "shared_but_harder" — a similar group exists in both, but the industrial
    statements show it is materially harder/stricter there (say why, based
    only on the group descriptions)
(c) "equal" — appears in both with similar severity

Return JSON:
{{"comparison": [
  {{"category": "industrial_only" | "shared_but_harder" | "equal",
    "constraint": "<group name>",
    "explanation": "<2-3 sentences grounded in the group descriptions>",
    "industrial_dois": ["..."],
    "generic_dois": ["..."]}}
]}}
Use only DOIs present in the groups above.
"""

PRINCIPLES_SYSTEM = """\
You are a synthesis engine for a systematic literature review. Your ONLY
sources are (1) the cross-stream constraint differences and (2) the gap
statements with verbatim quotes, both provided below. Do not use any prior
knowledge. Every principle must be traceable: cite supporting DOIs and reuse
the provided quotes EXACTLY as given (character-for-character).
Return ONLY valid JSON, no markdown fences.
"""

PRINCIPLES_PROMPT = """\
Research question: {rq2}

INPUT 1 — constraint differences between the industrial and generic streams
(categories: industrial_only and shared_but_harder):
{differences}

INPUT 2 — gap statements explicitly acknowledged by papers (DOI, gap, verbatim
quote). Quotes from industrial-stream papers are marked [industrial]:
{gaps}

Derive 5-10 design principles for Industrial AI (MLOps for manufacturing /
maintenance / physical environments). Each principle must:
- address a specific constraint difference from INPUT 1
- be supported by gap/constraint evidence from INPUT 2 (prefer >=2 papers)
- state what to DO about it (actionable design guidance)

Return JSON:
{{"principles": [
  {{"principle": "<imperative one-liner>",
    "constraint_difference": "<which INPUT-1 difference it addresses>",
    "rationale": "<2-3 sentences from the inputs only>",
    "action": "<what a system designer should do>",
    "support": [{{"doi": "...", "quote": "<copy one quote EXACTLY from INPUT 2>"}}]
  }}
]}}
Use only DOIs and quotes that appear in the inputs.
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def parse_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def load_papers() -> dict:
    return json.loads((INDEX_DIR / "papers.json").read_text(encoding="utf-8"))


def chunks_by_paper(chunks: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    out = defaultdict(list)
    for i, ch in enumerate(chunks):
        out[ch["chunk_id"].rsplit("::", 1)[0]].append((i, ch))
    return out


# --------------------------------------------------------------------------
# MAP phase
# --------------------------------------------------------------------------

def select_paper_chunks(idx_chunks, emb, rq_vec) -> list[tuple[int, dict]]:
    if len(idx_chunks) <= MAX_CHUNKS_PER_PAPER:
        return idx_chunks
    ids = np.array([i for i, _ in idx_chunks])
    sims = emb[ids] @ rq_vec
    keep = set(ids[np.argsort(-sims)[:MAX_CHUNKS_PER_PAPER]].tolist())
    return [(i, ch) for i, ch in idx_chunks if i in keep]  # original order


def verify_items(items: list[dict], excerpts: list[dict]) -> tuple[list[dict], list[str]]:
    """Verbatim-check each item's quote against its cited excerpt (fallback:
    any excerpt of the paper). Returns (verified items with pages, failures)."""
    ok, failed = [], []
    for it in items:
        quote = (it.get("quote") or "").strip()
        n = it.get("excerpt")
        if not quote or len(quote.split()) < 4:
            failed.append(quote or "<empty quote>")
            continue
        target = None
        if isinstance(n, int) and 1 <= n <= len(excerpts):
            if quote_in_text(quote, excerpts[n - 1]["text"]):
                target = excerpts[n - 1]
        if target is None:
            for ch in excerpts:
                if quote_in_text(quote, ch["text"]):
                    target = ch
                    break
        if target is None:
            failed.append(quote)
            continue
        ok.append({"item": it.get("item", "").strip(), "quote": quote,
                   "page_start": target["page_start"], "page_end": target["page_end"]})
    return ok, failed


def run_map() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    papers = load_papers()
    chunks, emb = load_index()
    by_paper = chunks_by_paper(chunks)

    rq_vec = np.asarray(
        embed_texts(["operational constraints, challenges, and limitations of deploying "
                     "and operating machine learning systems; gaps in current MLOps practice"],
                    "RETRIEVAL_QUERY")[0], dtype=np.float32)
    rq_vec /= max(np.linalg.norm(rq_vec), 1e-9)

    done = set()
    if EXTRACTIONS_PATH.exists():
        for line in EXTRACTIONS_PATH.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["key"])

    todo = [k for k in sorted(papers) if k not in done]
    print(f"Map phase: {len(done)} done, {len(todo)} to go", flush=True)
    stats = {"quotes_total": 0, "quotes_verified": 0, "dropped": 0}

    with open(EXTRACTIONS_PATH, "a", encoding="utf-8") as out:
        for i, key in enumerate(todo, 1):
            meta = papers[key]
            print(f"Processing {len(done) + i}/{len(papers)}: {meta['title'][:70]}", flush=True)
            paper_chunks = [ch for _, ch in select_paper_chunks(by_paper.get(key, []), emb, rq_vec)]
            rec = {"key": key, "doi": meta["doi"], "title": meta["title"],
                   "year": meta["year"], "stream": meta["stream"],
                   "constraints": [], "gaps": [], "dropped_quotes": []}
            if paper_chunks:
                excerpts_str = "\n\n---\n\n".join(
                    f"[{n}] (p. {ch['page_start']}–{ch['page_end']})\n{ch['text']}"
                    for n, ch in enumerate(paper_chunks, 1))
                prompt = MAP_PROMPT.format(doi=meta["doi"] or "no-doi",
                                           title=meta["title"], excerpts=excerpts_str)
                try:
                    data = parse_json(generate(prompt, MAP_SYSTEM, json_mode=True))
                except Exception as e:  # noqa: BLE001
                    print(f"  parse failure: {e}; recording empty", flush=True)
                    data = {"operational_constraints": [], "gaps_identified": []}

                cons, f1 = verify_items(data.get("operational_constraints") or [], paper_chunks)
                gaps, f2 = verify_items(data.get("gaps_identified") or [], paper_chunks)
                fails = f1 + f2
                if fails:  # one retry with the failures pointed out
                    retry = prompt + MAP_RETRY_SUFFIX.format(
                        failures="\n".join(f'- "{q[:120]}"' for q in fails))
                    try:
                        data = parse_json(generate(retry, MAP_SYSTEM, json_mode=True))
                        cons, f1 = verify_items(data.get("operational_constraints") or [], paper_chunks)
                        gaps, f2 = verify_items(data.get("gaps_identified") or [], paper_chunks)
                    except Exception as e:  # noqa: BLE001
                        print(f"  retry parse failure: {e}", flush=True)
                    rec["dropped_quotes"] = [q[:160] for q in (f1 + f2)]
                rec["constraints"], rec["gaps"] = cons, gaps
                n_ok = len(cons) + len(gaps)
                stats["quotes_verified"] += n_ok
                stats["quotes_total"] += n_ok + len(rec["dropped_quotes"])
                stats["dropped"] += len(rec["dropped_quotes"])
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    print(f"\nMap done. Quote verification: {stats['quotes_verified']}/{stats['quotes_total']} "
          f"verbatim-verified ({stats['dropped']} dropped after retry)", flush=True)


# --------------------------------------------------------------------------
# REDUCE phase
# --------------------------------------------------------------------------

IND_DEF = ("AI/ML deployment in manufacturing, Industry 4.0, predictive maintenance, "
           "industrial IoT, cyber-physical systems, process industries, energy/utilities, "
           "or safety-critical physical environments")
GEN_DEF = ("AI/ML deployment in cloud-native, SaaS, enterprise software, data platforms, "
           "or general engineering contexts not tied to physical production processes")


def load_extractions() -> list[dict]:
    return [json.loads(l) for l in EXTRACTIONS_PATH.read_text(encoding="utf-8").splitlines()]


def doi_check(text_or_dois, valid: set[str]) -> tuple[list[str], list[str]]:
    """Split paper identifiers (DOIs or no-doi keys) into (valid, invalid).

    `valid` may be restricted to one stream's papers, which also enforces that
    evidence lists only cite papers from the correct stream."""
    if isinstance(text_or_dois, list):
        dois = text_or_dois
    else:
        dois = re.findall(r"(?:10\.\d{4,9}/[^\s\]\),;\"']+|no-doi:[a-z0-9-]+)", text_or_dois)
    good, bad = [], []
    for d in dois:
        (good if d.strip().lower().rstrip(".") in valid else bad).append(d.strip())
    return good, bad


def group_stream(extractions, stream: str) -> dict:
    items, n_papers = [], 0
    for rec in extractions:
        if rec["stream"] != stream or not rec["constraints"]:
            continue
        n_papers += 1
        for c in rec["constraints"]:
            items.append(f"- [{rec['doi'] or rec['key']}] {c['item']}")
    print(f"Grouping {stream}: {len(items)} constraint items from {n_papers} papers", flush=True)
    prompt = GROUP_PROMPT.format(
        label=stream.upper(),
        stream_def=IND_DEF if stream == "industrial" else GEN_DEF,
        n=len(items), items="\n".join(items))
    return parse_json(generate(prompt, GROUP_SYSTEM, json_mode=True))


def reference_list(dois_used: set[str], papers: dict) -> str:
    by_doi = {}
    for key, meta in papers.items():
        by_doi[key.lower()] = meta
        if meta["doi"]:
            by_doi[meta["doi"].lower()] = meta
    lines = ["", "## References (DOI — title, year, stream)", ""]
    for d in sorted(dois_used):
        meta = by_doi.get(d.lower())
        if meta:
            lines.append(f"- `{d}` — {meta['title']} ({meta['year']}, {meta['stream']})")
        else:
            lines.append(f"- `{d}` — [not resolved in corpus]")
    return "\n".join(lines)


def run_reduce() -> None:
    extractions = load_extractions()
    papers = load_papers()
    pid = lambda r: (r["doi"] or r["key"]).lower()  # noqa: E731
    valid_dois = {pid(r) for r in extractions}
    valid_by_stream = {
        "industrial": {pid(r) for r in extractions if r["stream"] == "industrial"},
        "generic": {pid(r) for r in extractions if r["stream"] == "generic"},
    }

    # ---- RQ1 ----------------------------------------------------------------
    ind_groups = group_stream(extractions, "industrial")
    gen_groups = group_stream(extractions, "generic")

    cmp_prompt = COMPARE_PROMPT.format(
        rq1=RQ1, ind_def=IND_DEF, gen_def=GEN_DEF,
        ind_groups=json.dumps(ind_groups, indent=1),
        gen_groups=json.dumps(gen_groups, indent=1))
    comparison = parse_json(generate(cmp_prompt, COMPARE_SYSTEM, json_mode=True))

    # programmatic DOI verification
    cited, bad_all = set(), []
    for entry in comparison.get("comparison", []):
        for fld, strm in (("industrial_dois", "industrial"), ("generic_dois", "generic")):
            # stream-restricted set: also rejects citing a paper in the wrong stream
            good, bad = doi_check(entry.get(fld, []), valid_by_stream[strm])
            entry[fld] = good
            bad_all += bad
            cited.update(good)
    print(f"RQ1 comparison: {len(cited)} DOIs cited, {len(bad_all)} invalid removed", flush=True)

    cat_names = {"industrial_only": "(a) Constraints found ONLY in the industrial stream",
                 "shared_but_harder": "(b) Constraints in both streams but materially harder in industrial",
                 "equal": "(c) Constraints appearing equally in both streams"}
    md = [f"# RQ1 — {RQ1}", "",
          f"_Corpus: {len(extractions)} full-text papers "
          f"({sum(1 for r in extractions if r['stream'] == 'industrial')} industrial, "
          f"{sum(1 for r in extractions if r['stream'] == 'generic')} generic). "
          "All claims derive from per-paper extractions with verbatim-verified quotes; "
          "every group cites its supporting DOIs._", ""]
    for cat in ("industrial_only", "shared_but_harder", "equal"):
        md += [f"## {cat_names[cat]}", ""]
        # the catch-all "Other" buckets carry no comparable claim — skip them
        for e in [e for e in comparison.get("comparison", [])
                  if e.get("category") == cat and e.get("constraint", "").strip().lower() != "other"]:
            md += [f"### {e['constraint']}",
                   e.get("explanation", ""),
                   f"- Industrial evidence: {', '.join('`' + d + '`' for d in e['industrial_dois']) or '—'}",
                   f"- Generic evidence: {', '.join('`' + d + '`' for d in e['generic_dois']) or '—'}", ""]
    md += ["", "## Constraint groups per stream (aggregation detail)", ""]
    for label, grp in (("Industrial", ind_groups), ("Generic", gen_groups)):
        md += [f"### {label} stream", ""]
        for g in grp.get("groups", []):
            good, _ = doi_check(g.get("dois", []), valid_by_stream[label.lower()])
            cited.update(good)
            md += [f"- **{g['group']}** — {g['description']} "
                   f"({len(good)} papers: {', '.join('`' + d + '`' for d in good[:12])}"
                   f"{', …' if len(good) > 12 else ''})"]
        md += [""]
    md.append(reference_list(cited, papers))
    RQ1_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {RQ1_PATH}", flush=True)

    # ---- RQ2 ----------------------------------------------------------------
    differences = [e for e in comparison.get("comparison", [])
                   if e.get("category") in ("industrial_only", "shared_but_harder")]
    gap_lines, quote_pool = [], {}
    for rec in extractions:
        tag = " [industrial]" if rec["stream"] == "industrial" else ""
        for g in rec["gaps"] + rec["constraints"]:
            d = rec["doi"] or rec["key"]
            gap_lines.append(f'- [{d}]{tag} {g["item"]} | quote: "{g["quote"]}"')
            quote_pool.setdefault(d.lower(), []).append(g["quote"])

    pr_prompt = PRINCIPLES_PROMPT.format(
        rq2=RQ2, differences=json.dumps(differences, indent=1),
        gaps="\n".join(gap_lines))
    principles = parse_json(generate(pr_prompt, PRINCIPLES_SYSTEM, json_mode=True))

    # verification: quotes must match the extraction pool of the cited DOI
    n_q, n_q_ok = 0, 0
    cited2 = set()
    for p in principles.get("principles", []):
        verified = []
        for s in p.get("support", []):
            n_q += 1
            d = (s.get("doi") or "").strip()
            pool = quote_pool.get(d.lower(), [])
            if any(quote_in_text(s.get("quote", ""), q) or quote_in_text(q, s.get("quote", ""))
                   for q in pool):
                n_q_ok += 1
                verified.append(s)
                cited2.add(d)
            else:
                p.setdefault("unverified_support", []).append(s)
        p["support"] = verified
    print(f"RQ2 principles: {n_q_ok}/{n_q} support quotes verified against extractions", flush=True)

    md2 = [f"# RQ2 — {RQ2}", "",
           "_Each principle addresses a verified cross-stream constraint difference "
           "(RQ1 categories a/b) and is supported by verbatim-verified quotes from "
           "the per-paper gap extractions. Principles with fewer than 2 supporting "
           "papers are labelled **single-source**._", ""]
    for i, p in enumerate(principles.get("principles", []), 1):
        n_support = len({s["doi"].lower() for s in p["support"]})
        tag = " — **single-source**" if n_support < 2 else ""
        md2 += [f"## P{i}. {p['principle']}{tag}", "",
                f"**Constraint difference addressed:** {p.get('constraint_difference', '')}", "",
                f"**Rationale:** {p.get('rationale', '')}", "",
                f"**What to do:** {p.get('action', '')}", "",
                "**Supporting evidence:**"]
        for s in p["support"]:
            md2.append(f'- `{s["doi"]}`: "{s["quote"]}"')
        if p.get("unverified_support"):
            md2.append(f"- _({len(p['unverified_support'])} further support item(s) dropped: "
                       "quote could not be verified verbatim)_")
        md2.append("")
    md2.append(reference_list(cited2, papers))
    RQ2_PATH.write_text("\n".join(md2), encoding="utf-8")
    print(f"Wrote {RQ2_PATH}", flush=True)

    # summary
    print("\n=== REDUCE SUMMARY ===")
    print(f"RQ1: {len(comparison.get('comparison', []))} compared constraint groups; "
          f"{len(cited)} distinct DOIs cited; {len(bad_all)} invalid DOIs removed")
    print(f"RQ2: {len(principles.get('principles', []))} principles; "
          f"quote verification {n_q_ok}/{n_q}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-only", action="store_true")
    ap.add_argument("--reduce-only", action="store_true")
    args = ap.parse_args()
    if not args.reduce_only:
        run_map()
    if not args.map_only:
        run_reduce()
