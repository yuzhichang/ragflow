#!/usr/bin/env python3
"""Do the reference documents' chunks cover the reference text - and can the agent see them?

list_chunks reads a document through one narrow filter (grep_service.go:89-101):

    pattern=".*", sorted by (doc_id, page_num_int, chunk_order_int), filtered to
    available_int == 1 and doc_id in scope, with `exists(compile_kwd)` excluded,
    then skipping chunks whose content is empty or is a graph chunk.

So for every reference document there are THREE populations, and conflating them
is what makes a coverage question unanswerable:

  * indexed   - every chunk the index holds for the document;
  * visible   - the chunks that survive list_chunks' filter, i.e. what an agent
                can actually deep-read;
  * the file  - the corpus .md the benchmark calls the reference.

This script measures the file's word n-grams against both the indexed and the
visible reassembly, so a document can be classified instead of merely flagged:

    both ~1.0                      -> complete
    indexed ~1.0, visible < 1.0    -> HIDDEN: indexed but filtered out of
                                      list_chunks' view (available_int / compile_kwd)
    indexed < 1.0                  -> MISSING: the text is not in the index at all

Word n-grams (8 tokens, one every 4) rather than character windows on purpose:
the corpus files carry MediaWiki-era markup (`| budget = $8 million`, `====`
rules, pipe tables) which the ingest rewrites (`budget: $8 million`), and a
character-window metric reported ~90% coverage for documents that are complete -
every "missing" span sat on markup the index had normalised.

Usage:
    python3 scripts/browsecomp_chunk_coverage.py \
        --ids failed [--config scripts/browsecompplus_retry_conf.json] \
        [--workers 8] [--out outputs/chunk_coverage]
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import glob
import json
import os
import re
import sys
import time
import urllib.request
from typing import Any

CORPUS = "/home/zhichyu/Downloads/data/browsecomp-plus/corpus"
QUESTIONS = "/home/zhichyu/Downloads/data/browsecomp-plus/questions.jsonl"

NGRAM = 8
STRIDE_TOKENS = 4
TOKEN_RE = re.compile(r"[a-z0-9]+")


def es_request(base: str, auth: str, path: str, body: dict[str, Any], attempts: int = 3) -> dict[str, Any]:
    """One ES call, retried on transient failures.

    The first concurrent sweep lost two documents to `_ssl.c:3135 unknown error`
    while the very same documents succeeded when checked one at a time, so the
    failure is transport flakiness under concurrency, not data. A sweep that
    silently drops such documents would report a CLEAN corpus exactly where it
    failed to look, which is the one error this script must not make.
    """
    url = base.rstrip("/") + path
    payload = json.dumps(body).encode()
    headers = {"Authorization": "Basic " + auth, "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.load(resp)
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised when exhausted
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"ES request failed after {attempts} attempts: {last}")


def is_graph_chunk(content: str) -> bool:
    """Mirror of agentic_rag.isGraphChunkContent: a JSON chunk that is graph data."""
    trimmed = content.strip()
    if len(trimmed) < 2 or not trimmed.startswith("{") or not trimmed.endswith("}"):
        return False
    try:
        obj = json.loads(trimmed)
    except ValueError:
        return False
    if not isinstance(obj, dict):
        return False
    if obj.get("type") in ("relation", "entity", "location"):
        return True
    return "head" in obj


def doc_chunks(es: str, auth: str, index: str, kb_id: str, stem: str) -> list[dict[str, Any]]:
    """EVERY chunk the index holds for the document, in reading order.

    Deliberately unfiltered by available_int / compile_kwd: this is the "indexed"
    population, and the visible one is derived from it here rather than being
    fetched separately, so the two answers cannot drift apart.
    """
    body = {
        "size": 2000,
        "track_total_hits": True,
        "_source": [
            "content_with_weight",
            "doc_id",
            "docnm_kwd",
            "chunk_order_int",
            "page_num_int",
            "available_int",
            "compile_kwd",
        ],
        "query": {"bool": {"filter": [{"term": {"kb_id": kb_id}}, {"term": {"docnm_kwd": f"{stem}.md"}}]}},
        "sort": [{"doc_id": "asc"}, {"page_num_int": "asc"}, {"chunk_order_int": "asc"}],
    }
    res = es_request(es, auth, f"/{index}/_search", body)
    hits = res.get("hits", {})
    total = (hits.get("total") or {}).get("value")
    out: list[dict[str, Any]] = []
    for hit in hits.get("hits", []):
        src = hit.get("_source") or {}
        content = src.get("content_with_weight") or ""
        out.append(
            {
                "id": hit.get("_id") or "",
                "content": content,
                "available": src.get("available_int"),
                "compiled": src.get("compile_kwd") is not None,
                "order": src.get("chunk_order_int"),
            }
        )
    if isinstance(total, int) and total > len(hits.get("hits", [])):
        print(f"  warn: {stem} reported {total} chunks, fetched {len(hits.get('hits', []))}", file=sys.stderr)
    return out


def visible_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset list_chunks would hand the model.

    The engine does NOT test `available_int == 1`; it translates that filter into
    `must_not {range: {available_int: {lt: 1}}}` (elasticsearch/chunk.go:2058-2082),
    which keeps every chunk whose `available_int` is ABSENT as well as those equal
    to 1. Reading the flag as an equality marked all 34 chunks of one document
    invisible and would have reported a visibility defect that does not exist.
    """
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        available = chunk.get("available")
        if isinstance(available, (int, float)) and available < 1:
            continue
        if chunk.get("compiled"):
            continue
        if not chunk["content"].strip() or is_graph_chunk(chunk["content"]):
            continue
        out.append(chunk)
    return out


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= STRIDE_TOKENS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def coverage_spans(file_tokens: list[str], chunk_text: str) -> tuple[int, int, list[tuple[int, int]]]:
    """(matched, total, missing token spans) for the file's n-grams against chunk_text."""
    chunk_tokens = TOKEN_RE.findall(chunk_text.lower())
    grams = {tuple(chunk_tokens[i : i + NGRAM]) for i in range(max(1, len(chunk_tokens) - NGRAM + 1))}
    if len(file_tokens) < NGRAM:
        return 0, 0, []
    matched = 0
    total = 0
    missing: list[tuple[int, int]] = []
    for start in range(0, len(file_tokens) - NGRAM + 1, STRIDE_TOKENS):
        total += 1
        if tuple(file_tokens[start : start + NGRAM]) in grams:
            matched += 1
        else:
            missing.append((start, start + NGRAM))
    return matched, total, merge_spans(missing)


def check_doc(es: str, auth: str, index: str, kb_id: str, stem: str) -> dict[str, Any]:
    path = os.path.join(CORPUS, f"{stem}.md")
    row: dict[str, Any] = {"doc": stem, "exists": os.path.exists(path)}
    if not row["exists"]:
        row["coverage_visible"] = row["coverage_indexed"] = None
        return row
    with open(path, encoding="utf-8", errors="ignore") as fh:
        file_text = fh.read()
    file_tokens = TOKEN_RE.findall(file_text.lower())

    chunks = doc_chunks(es, auth, index, kb_id, stem)
    visible = visible_chunks(chunks)
    vis_matched, total, vis_missing = coverage_spans(file_tokens, " ".join(c["content"] for c in visible))
    idx_matched, idx_total, idx_missing = coverage_spans(file_tokens, " ".join(c["content"] for c in chunks))

    # The reverse direction: how much of the INDEXED text is absent from the file.
    # A document whose indexed text is LONGER than the file yet still misses 8% of
    # the file's n-grams cannot be explained by "the ingest dropped a section" -
    # the two texts simply differ, which is what two snapshots of one URL look
    # like. Measuring one direction cannot tell those apart, so both are measured.
    rev_matched, rev_total, _ = coverage_spans(TOKEN_RE.findall(" ".join(c["content"] for c in chunks).lower()), file_text)
    row.update(
        {
            "coverage_index_in_file": round(rev_matched / max(rev_total, 1), 4),
            "chunks_indexed": len(chunks),
            "chunks_visible": len(visible),
            "chunks_unavailable": sum(1 for c in chunks if c.get("available") != 1),
            "chunks_compiled": sum(1 for c in chunks if c.get("compiled")),
            "chunks_empty": sum(1 for c in chunks if not c["content"].strip()),
            "file_chars": len(file_text),
            "file_tokens": len(file_tokens),
            "indexed_chars": sum(len(c["content"]) for c in chunks),
            "visible_chars": sum(len(c["content"]) for c in visible),
            "ngrams": total,
            "coverage_visible": round(vis_matched / max(total, 1), 4),
            "coverage_indexed": round(idx_matched / max(idx_total, 1), 4),
            "uncovered_visible_tokens": sum(b - a for a, b in vis_missing),
            "uncovered_indexed_tokens": sum(b - a for a, b in idx_missing),
            "missing_spans": [{"start": a, "end": b, "excerpt": " ".join(file_tokens[a : a + 18])} for a, b in vis_missing[:3]],
        }
    )
    # The classification the three populations exist to produce. The ASYMMETRY
    # between the two directions is what separates the cases: a dropped section
    # leaves the file far more uncovered than the index (q912's document: 8% of
    # the file present, 87% of the index present), while a reflowed table or an
    # infobox rewrite gaps BOTH directions by similar amounts and loses nothing.
    visible_cov = row["coverage_visible"]
    indexed_cov = row["coverage_indexed"]
    reverse_cov = row["coverage_index_in_file"]
    if visible_cov >= 0.99 and indexed_cov >= 0.99:
        row["class"] = "complete"
    elif visible_cov < 0.99 <= indexed_cov:
        row["class"] = "hidden_from_list_chunks"
    elif visible_cov < 0.8 * reverse_cov:
        row["class"] = "content_missing_from_index"
    elif visible_cov >= 0.85 and reverse_cov >= 0.85:
        row["class"] = "index_and_file_differ"
    else:
        row["class"] = "partial_overlap"
    return row


def failed_ids(config: str) -> list[str]:
    with open(config, encoding="utf-8") as fh:
        conf = json.load(fh)
    ann = conf.get("annotations") or {}
    ids: list[str] = []
    for key, value in ann.items():
        if key.startswith("failed_"):
            ids.extend(str(x) for x in value)
    return sorted(set(ids), key=int)


def expected_docs(ids: list[str], questions_path: str) -> dict[str, list[str]]:
    wanted = set(ids)
    out: dict[str, list[str]] = {}
    with open(questions_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            q = json.loads(line)
            qid = str(q.get("id"))
            if qid in wanted:
                out[qid] = [str(x) for x in (q.get("expected_doc_ids") or [])]
    return out


def discover_scope() -> tuple[str, str]:
    for summary in sorted(glob.glob("outputs/reachability_*/summary.json"), reverse=True):
        with open(summary, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("index") and data.get("kb_id"):
            return data["index"], data["kb_id"]
    raise SystemExit("no reachability summary found; pass --index and --kb-id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", default="failed", help="'failed' (every id in a failed_* bucket) or a comma list")
    ap.add_argument(
        "--docs",
        help="document stems (e.g. '33637,34297'), checked directly instead of via questions - used to re-verify specific documents after a re-parse",
    )
    ap.add_argument("--config", default="scripts/browsecompplus_retry_conf.json")
    ap.add_argument("--questions", default=QUESTIONS, help="override the questions jsonl (expected docs)")
    ap.add_argument("--es", default="http://localhost:1200")
    ap.add_argument("--es-user", default="elastic")
    ap.add_argument("--es-password", default="infini_rag_flow")
    ap.add_argument("--index")
    ap.add_argument("--kb-id")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/chunk_coverage")
    args = ap.parse_args()

    args.auth = base64.b64encode(f"{args.es_user}:{args.es_password}".encode()).decode()
    if not (args.index and args.kb_id):
        args.index, args.kb_id = discover_scope()

    if args.docs:
        docs = sorted({d.strip() for d in args.docs.split(",") if d.strip()})
        per_question = {"_docs": docs}
    else:
        ids = failed_ids(args.config) if args.ids == "failed" else [x.strip() for x in args.ids.split(",") if x.strip()]
        per_question = expected_docs(ids, args.questions)
        docs = sorted({d for ds in per_question.values() for d in ds})
    print(f"questions={len(per_question)} reference documents={len(docs)} workers={args.workers}")
    print("scope = every chunk for the doc; visible = list_chunks' own filter (available_int:1, no compile_kwd)")
    print(f"metric = word {NGRAM}-grams, one every {STRIDE_TOKENS} tokens")

    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_doc, args.es, args.auth, args.index, args.kb_id, stem): stem for stem in docs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            stem = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - one bad document must not stop the sweep
                rows.append({"doc": stem, "error": str(exc), "coverage_visible": None, "coverage_indexed": None})
            if done % 25 == 0:
                print(f"  checked {done}/{len(docs)}", flush=True)

    rows.sort(key=lambda r: (r.get("coverage_visible") if r.get("coverage_visible") is not None else 1.0, r["doc"]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "coverage.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    flagged = [r for r in rows if r.get("class") and r["class"] != "complete"]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("class") or "error"] = counts.get(r.get("class") or "error", 0) + 1
    print("\nclassification:", counts)
    print(f"{'doc':>8} {'class':<22} {'vis':>7} {'idx':>7} {'rev':>7} {'chunks(vis/all)':>16}  first uncovered excerpt")
    for r in flagged:
        vis = r.get("coverage_visible")
        idx = r.get("coverage_indexed")
        span = (r.get("missing_spans") or [{}])[0].get("excerpt", "")
        note = f"  ERROR: {r['error']}" if r.get("error") else f"  {span[:70]!r}"
        print(f"{r['doc']:>8} {r.get('class')!s:<24} {vis!s:>7} {idx!s:>7} {r.get('coverage_index_in_file')!s:>7} {str(r.get('chunks_visible')) + '/' + str(r.get('chunks_indexed')):>16}{note}")

    per_q = {}
    for qid, ds in per_question.items():
        qrows = [r for r in rows if r["doc"] in ds and r.get("class")]
        per_q[qid] = {
            "docs": len(ds),
            "worst_visible": min((r["coverage_visible"] for r in qrows), default=None),
            "classes": sorted({r["class"] for r in qrows}),
        }
    summary = {
        "questions": len(per_question),
        "documents": len(rows),
        "classification": counts,
        "not_complete": [r["doc"] for r in flagged],
        "errors": [r["doc"] for r in rows if r.get("error")],
        "ngram_tokens": NGRAM,
        "stride_tokens": STRIDE_TOKENS,
        "per_question": per_q,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}/coverage.jsonl and summary.json")


if __name__ == "__main__":
    main()
