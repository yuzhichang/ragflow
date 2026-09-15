#!/usr/bin/env python3
"""Did the answer-bearing chunk ever reach the model, and was it cited?

Zero-LLM diagnostic over an archived answers.jsonl that was produced by a backend
carrying the chunk-id read ledger (`deep_read_chunk_ids` / `shallow_read_chunk_ids`).
It answers, per question, the question the ledger was added for:

    the model read N chunks; the deliverable cites M of them; the chunks that
    hold the GOLD string are among the read ones / the uncited ones / neither.

That split is the whole diagnosis for the evidence_in_hand class. Measured on the
2026-09-15 retry batch (before the ledger existed), 13 of 20 such failures had a
gold-bearing DOCUMENT served to the run, only 6 of those cited it and 0 shipped it
- which left "read and silently dropped" indistinguishable from "never read". The
ledger settles it at chunk granularity:

  * gold chunk was never read            -> the loss is upstream (retrieval/ranking)
  * gold chunk was read, never cited     -> the loss is candidate formation:
                                            the passage was in front of the model
                                            and no matrix line carried it
  * gold chunk was read AND cited        -> the loss is the answer/verdict step

Usage:
    python3 scripts/browsecomp_read_chunk_probe.py \
        --answers outputs/browsecomp_eih3/answers.jsonl \
        [--questions /home/zhichyu/Downloads/data/browsecomp-plus/questions.jsonl] \
        [--es http://localhost:1200] [--top-docs 0] [--out outputs/read_chunk_probe]

ES defaults mirror scripts/browsecomp_reachability_probe.py; the index and kb_id
are auto-discovered from a reachability run when one exists, so the two probes
speak about the same corpus.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import re
import sys
import urllib.request
from typing import Any

DEFAULT_QUESTIONS = "/home/zhichyu/Downloads/data/browsecomp-plus/questions.jsonl"

# The deliverable cites `chunk_id: <id>` on every matrix/chain line (the auditor
# verifies the field), so the cited set is read off the text, not inferred.
CITED_CHUNK_RE = re.compile(r"chunk_id[:\s`\"']+([0-9a-fA-F]{8,})")


def es_request(base: str, auth: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def discover_scope(es: str, auth: str) -> tuple[str, str]:
    """Reuse the reachability probe's index/kb so both probes read one corpus."""
    for path in sorted(glob.glob("outputs/reachability_*/summary.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            continue
        if summary.get("index") and summary.get("kb_id"):
            return summary["index"], summary["kb_id"]
    raise SystemExit("no outputs/reachability_*/summary.json found; pass --index and --kb-id")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_questions(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    return {str(q.get("id")): q for q in load_jsonl(path)}


def doc_chunks(es: str, auth: str, index: str, kb_id: str, stem: str, cache: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    """Every chunk of one corpus document, as {"id", "text"}.

    One document is one .md file and its chunks are keyed by docnm_kwd (e.g.
    "14497.md"), exactly the name the deliverable cites in its `doc:` field.
    """
    if stem in cache:
        return cache[stem]
    body = {
        "size": 1000,
        "track_total_hits": True,
        "_source": ["content_with_weight", "docnm_kwd", "doc_id"],
        "query": {"bool": {"filter": [{"term": {"kb_id": kb_id}}, {"term": {"docnm_kwd": f"{stem}.md"}}]}},
    }
    res = es_request(es, auth, f"/{index}/_search", body)
    hits = res.get("hits", {})
    chunks = [{"id": hit.get("_id") or "", "text": (hit.get("_source") or {}).get("content_with_weight") or ""} for hit in hits.get("hits", [])]
    total = (hits.get("total") or {}).get("value")
    if isinstance(total, int) and total > len(chunks):
        print(f"  warn: {stem} has {total} chunks, fetched {len(chunks)} (raise size)", file=sys.stderr)
    cache[stem] = chunks
    return chunks


def probe_row(
    row: dict[str, Any],
    question: dict[str, Any] | None,
    args: argparse.Namespace,
    cache: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    qid = str(row.get("question_id"))
    answer = str(row.get("ragflow_answer") or "")
    deep = [str(x) for x in (row.get("deep_read_chunk_ids") or [])]
    shallow = [str(x) for x in (row.get("shallow_read_chunk_ids") or [])]
    cited = {m.group(1) for m in CITED_CHUNK_RE.finditer(answer)}
    uncited_deep = [c for c in deep if c not in cited]

    out: dict[str, Any] = {
        "id": qid,
        "instrumented": bool(deep or shallow),
        "deep_read": len(deep),
        "shallow_read": len(shallow),
        "cited": len(cited),
        "uncited_deep": len(uncited_deep),
        "deep_read_chunks": row.get("deep_read_chunks"),
        "gold_answer": (question or {}).get("gold_answer") or row.get("gold_answer"),
        "expected_docs": [str(x) for x in ((question or {}).get("expected_doc_ids") or row.get("expected_doc_ids") or [])],
    }
    # The ledger's own accounting should agree with the counts the backend has
    # been emitting all along; a mismatch means the ids and the totals describe
    # different populations, which would silently invalidate every rate below.
    out["counts_agree"] = (not deep) or (isinstance(row.get("deep_read_chunks"), int) and row["deep_read_chunks"] == len(deep))

    gold = str(out["gold_answer"] or "").strip().lower()
    gold_chunks: list[dict[str, Any]] = []
    if gold and out["expected_docs"]:
        for stem in out["expected_docs"]:
            for chunk in doc_chunks(args.es, args.auth, args.index, args.kb_id, stem, cache):
                if gold in chunk["text"].lower():
                    gold_chunks.append(
                        {
                            "doc": stem,
                            "chunk_id": chunk["id"],
                            "in_deep_read": chunk["id"] in deep,
                            "in_cited": chunk["id"] in cited,
                        }
                    )
    out["gold_chunks"] = gold_chunks
    out["gold_chunks_total"] = len(gold_chunks)
    out["gold_chunks_read"] = sum(1 for c in gold_chunks if c["in_deep_read"])
    out["gold_chunks_cited"] = sum(1 for c in gold_chunks if c["in_cited"])
    # An uninstrumented row (a backend built before the ledger) has an EMPTY read
    # set, which would read as "the gold chunk was never read" - the opposite of
    # what an absent ledger can support. Such a row is undecided, not a finding.
    if not out["instrumented"]:
        out["verdict"] = "uninstrumented_no_ledger"
    elif not gold_chunks:
        out["verdict"] = "no_gold_chunk_in_expected_docs"
    elif out["gold_chunks_cited"]:
        out["verdict"] = "gold_read_and_cited__loss_is_the_answer_step"
    elif out["gold_chunks_read"]:
        out["verdict"] = "gold_read_never_cited__loss_is_candidate_formation"
    else:
        out["verdict"] = "gold_never_read__loss_is_upstream"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True, help="answers.jsonl produced with the chunk-id ledger")
    ap.add_argument("--questions", default=DEFAULT_QUESTIONS, help="dataset questions.jsonl (for gold + expected docs)")
    ap.add_argument("--es", default="http://localhost:1200")
    ap.add_argument("--es-user", default="elastic")
    ap.add_argument("--es-password", default="infini_rag_flow")
    ap.add_argument("--index", help="override the auto-discovered ES index")
    ap.add_argument("--kb-id", help="override the auto-discovered knowledge-base id")
    ap.add_argument("--out", default="outputs/read_chunk_probe", help="output directory")
    ap.add_argument("--no-es", action="store_true", help="skip the gold-chunk lookup (ledger arithmetic only)")
    args = ap.parse_args()

    args.auth = base64.b64encode(f"{args.es_user}:{args.es_password}".encode()).decode()
    if args.index and args.kb_id:
        pass
    elif args.no_es:
        args.index = args.kb_id = ""
    else:
        args.index, args.kb_id = discover_scope(args.es, args.auth)

    rows = load_jsonl(args.answers)
    questions = load_questions(args.questions)
    cache: dict[str, list[dict[str, str]]] = {}
    out_rows = []
    for row in rows:
        qid = str(row.get("question_id"))
        out_rows.append(probe_row(row, questions.get(qid), args, cache))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "probe.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)

    print(f"{'id':>6} {'deep':>5} {'cited':>6} {'uncited':>8} {'goldCk':>7} {'goldRead':>9} {'goldCited':>10}  verdict")
    for r in out_rows:
        print(f"{r['id']:>6} {r['deep_read']:>5} {r['cited']:>6} {r['uncited_deep']:>8} {r['gold_chunks_total']:>7} {r['gold_chunks_read']:>9} {r['gold_chunks_cited']:>10}  {r['verdict']}")

    uninstrumented = [r["id"] for r in out_rows if not r["instrumented"]]
    disagree = [r["id"] for r in out_rows if not r["counts_agree"]]
    summary = {
        "rows": len(out_rows),
        "uninstrumented": uninstrumented,
        "counts_disagree_with_ids": disagree,
        "verdicts": {v: sum(1 for r in out_rows if r["verdict"] == v) for v in sorted({r["verdict"] for r in out_rows})},
        "median_deep_read": sorted(r["deep_read"] for r in out_rows)[len(out_rows) // 2] if out_rows else None,
        "median_uncited_deep": sorted(r["uncited_deep"] for r in out_rows)[len(out_rows) // 2] if out_rows else None,
        "answers": args.answers,
        "index": args.index,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\n=== summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    if uninstrumented:
        print("\nNOTE: those rows carry no chunk ids (a backend built before the ledger),")
        print("      so they can only be read with the doc-level retrieved_docids.")
    print(f"\nwrote {args.out}/probe.jsonl and summary.json")


if __name__ == "__main__":
    main()
