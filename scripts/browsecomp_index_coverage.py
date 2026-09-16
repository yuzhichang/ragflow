#!/usr/bin/env python3
"""Is the markdown原文 fully covered by the chunks the corpus index serves?

`list_chunks` reads ES chunk records, so a document whose indexed chunks are a SUBSET of
its .md file is a document no run can ever read in full - the evidence may simply not be
there. Length comparison is not enough to see that (an index can hold the same number of
characters and still have dropped a passage), so this probe does an INTERVAL check instead:

  1. fetch every chunk of the document, ordered by chunk_order_int (the order list_chunks
     serves them in) and concatenate their content_with_weight;
  2. normalise both sides by collapsing whitespace runs (the ingest re-wraps text and adds a
     Document/Source URL/title/date header, so byte equality is not the question);
  3. run difflib over the two and take the MATCHING BLOCKS as covered intervals of the FILE.

Whatever part of the file no block covers is content the index does not serve. Reported per
document as a coverage percentage plus the uncovered spans with their context, because a gap
is only interesting when something in it matters.

Usage:
    python3 scripts/browsecomp_index_coverage.py --ids 223,233,... [--questions ...] [--corpus ...]
        [--es http://localhost:1200] [--min-gap 40] [--out outputs/index_coverage]
"""

from __future__ import annotations

import argparse
import base64
import difflib
import glob
import json
import os
import re
import sys
import urllib.request
from typing import Any

DEFAULT_QUESTIONS = "/home/zhichyu/Downloads/data/browsecomp-plus/questions.jsonl"
DEFAULT_CORPUS = "/home/zhichyu/Downloads/data/browsecomp-plus/corpus"

WHITESPACE_RE = re.compile(r"\s+")


def es_request(base: str, auth: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def discover_scope(es: str, auth: str) -> tuple[str, str]:
    for path in sorted(glob.glob("outputs/reachability_*/summary.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            continue
        if summary.get("index") and summary.get("kb_id"):
            return summary["index"], summary["kb_id"]
    raise SystemExit("no outputs/reachability_*/summary.json found; pass --index and --kb-id")


def load_questions(path: str) -> dict[str, dict[str, Any]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                q = json.loads(line)
                out[str(q.get("id"))] = q
    return out


def indexed_chunks(es: str, auth: str, index: str, kb_id: str, stem: str, cache: dict[str, list[str]]) -> list[str]:
    """One document's chunk texts in chunk_order_int order — what list_chunks serves."""
    if stem in cache:
        return cache[stem]
    texts: list[tuple[int, str]] = []
    body = {
        "size": 500,
        "track_total_hits": True,
        "_source": ["content_with_weight", "chunk_order_int"],
        "query": {"bool": {"filter": [{"term": {"kb_id": kb_id}}, {"term": {"docnm_kwd": f"{stem}.md"}}]}},
    }
    res = es_request(es, auth, f"/{index}/_search", body)
    hits = res.get("hits", {})
    for hit in hits.get("hits", []):
        src = hit.get("_source") or {}
        order = src.get("chunk_order_int")
        texts.append((order if isinstance(order, int) else 10**9, src.get("content_with_weight") or ""))
    texts.sort(key=lambda pair: pair[0])
    cache[stem] = [text for _, text in texts]
    total = (hits.get("total") or {}).get("value")
    if isinstance(total, int) and total > len(texts):
        print(f"  warn: {stem} reports {total} chunks, fetched {len(texts)} (raise size)", file=sys.stderr)
    return cache[stem]


def normalise(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def coverage(file_text: str, chunks: list[str], min_gap: int) -> dict[str, Any]:
    """Covered intervals of the FILE, from matching blocks against the chunk text."""
    file_norm = normalise(file_text)
    joined_norm = normalise("\n".join(chunks))
    if not file_norm:
        return {"file_chars": 0, "covered_chars": 0, "coverage_pct": None, "gaps": []}
    matcher = difflib.SequenceMatcher(None, file_norm, joined_norm, autojunk=False)
    covered = bytearray(len(file_norm))
    for block in matcher.get_matching_blocks():
        for pos in range(block.a, block.a + block.size):
            covered[pos] = 1
    gaps: list[dict[str, Any]] = []
    start: int | None = None
    for pos, flag in enumerate(covered):
        if not flag and start is None:
            start = pos
        elif flag and start is not None:
            if pos - start >= min_gap:
                gaps.append(
                    {
                        "start": start,
                        "length": pos - start,
                        "context": file_norm[max(0, start - 40) : pos + 40],
                    }
                )
            start = None
    if start is not None and len(file_norm) - start >= min_gap:
        gaps.append({"start": start, "length": len(file_norm) - start, "context": file_norm[max(0, start - 40) :]})
    covered_chars = sum(covered)
    return {
        "file_chars": len(file_norm),
        "covered_chars": covered_chars,
        "coverage_pct": round(100.0 * covered_chars / len(file_norm), 1),
        "chunk_count": len(chunks),
        "gap_chars": len(file_norm) - covered_chars,
        "gaps": gaps[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="comma-separated question ids")
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--es", default="http://localhost:1200")
    ap.add_argument("--es-user", default="elastic")
    ap.add_argument("--es-password", default="infini_rag_flow")
    ap.add_argument("--index")
    ap.add_argument("--kb-id")
    ap.add_argument("--min-gap", type=int, default=40, help="only report uncovered spans at least this long")
    ap.add_argument("--gold", action="store_true", help="also report whether the gold string lands in a gap")
    ap.add_argument("--out", default="outputs/index_coverage")
    args = ap.parse_args()

    args.auth = base64.b64encode(f"{args.es_user}:{args.es_password}".encode()).decode()
    if args.index and args.kb_id:
        pass
    else:
        args.index, args.kb_id = discover_scope(args.es, args.auth)

    questions = load_questions(args.questions)
    cache: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    ids = [x.strip() for x in args.ids.replace("\n", ",").split(",") if x.strip()]
    print(f"{'doc':>8} {'chunks':>6} {'file':>7} {'covered':>8} {'cover%':>7} {'gap':>6}  note")
    for qid in ids:
        question = questions.get(qid) or {}
        gold = normalise(str(question.get("gold_answer") or ""))
        expected = [str(x) for x in (question.get("expected_doc_ids") or [])]
        for stem in expected:
            path = os.path.join(args.corpus, f"{stem}.md")
            if not os.path.exists(path):
                print(f"{stem:>8} {'-':>6} {'-':>7} {'-':>8} {'-':>7} {'-':>6}  FILE MISSING")
                continue
            with open(path, encoding="utf-8", errors="ignore") as fh:
                file_text = fh.read()
            chunks = indexed_chunks(args.es, args.auth, args.index, args.kb_id, stem, cache)
            stats = coverage(file_text, chunks, args.min_gap)
            note = ""
            if args.gold and gold and stats["gaps"]:
                # Is the gold in the part the index does not serve?
                flat = normalise(file_text)
                gold_hit_in_file = gold.lower() in flat.lower()
                gold_in_gap = any(gold.lower() in gap["context"].lower() for gap in stats["gaps"])
                if gold_hit_in_file and gold_in_gap:
                    note = "GOLD IS IN AN UNCOVERED SPAN"
                elif gold_hit_in_file:
                    note = "gold present, not in a reported gap"
                else:
                    note = "gold not in the file at all"
            print(f"{stem:>8} {stats.get('chunk_count', 0):>6} {stats['file_chars']:>7} {stats['covered_chars']:>8} {stats['coverage_pct']!s:>7} {stats.get('gap_chars', 0):>6}  {note}")
            rows.append({"qid": qid, "doc": stem, **stats, "note": note, "gaps": stats["gaps"]})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "coverage.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    incomplete = [r for r in rows if (r.get("coverage_pct") or 100) < 99.0]
    print("\n=== summary ===")
    print(f"  documents checked: {len(rows)}")
    print(f"  fully covered (>=99%): {len(rows) - len(incomplete)}")
    print(f"  with uncovered spans: {len(incomplete)} -> {[(r['qid'], r['doc'], r['coverage_pct']) for r in incomplete]}")
    if incomplete:
        print("\n  the uncovered spans themselves (a gap only matters when it matters):")
        for r in incomplete:
            print(f"  q{r['qid']} {r['doc']}.md coverage={r['coverage_pct']}% gap_chars={r['gap_chars']}")
            for gap in r["gaps"][:3]:
                print(f"      [{gap['start']}:+{gap['length']}] …{gap['context'][:150]}…")
    print(f"\nwrote {args.out}/coverage.jsonl")


if __name__ == "__main__":
    main()
