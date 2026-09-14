#!/usr/bin/env python3
"""BrowseComp-Plus reachability probe: can a question's own wording reach its evidence?

This is a zero-LLM diagnostic over the whole question set. For every question it
measures three things against the corpus index (BM25 only, tokenised exactly the
way RAGFlow tokenises it, so the numbers mean what the retrieval leg means):

  1. literal  - does the QUESTION'S OWN WORDING retrieve any expected_doc_id?
  2. gold     - does the GOLD ANSWER string retrieve any expected_doc_id? (the
                ceiling: if the answer's own name cannot reach its evidence, no
                amount of agent reasoning can be blamed for the miss)
  3. in_doc   - is the gold string actually PRESENT in the expected documents?
                (the label control: a gold that appears nowhere in its own
                evidence set is an unanswerable or mis-generated question, not a
                retrieval failure)

The cross-tab of (1) x (2) is the point of the probe. It separates:

  A  literal hit                     -> the question's wording reaches the
                                        evidence directly; any failure is the
                                        agent's.
  B  literal miss, gold hit          -> PARAPHRASE-LIMITED: the evidence is
                                        reachable, but only through vocabulary
                                        the question deliberately rewrites away
                                        (the q350 class: "family business /
                                        manufacturing a product made from a
                                        fruit" describes a restaurant).
  C  gold miss, in_doc > 0           -> keyword ranking alone cannot reach a
                                        document that contains the answer; the
                                        vector leg or the doc's own wording is
                                        doing the work (hybrid-only).
  D  gold miss, in_doc == 0          -> the answer is not written in its own
                                        evidence set: unanswerable as worded, a
                                        mis-mapping of gold to documents, or a
                                        derived (computed) answer.

Usage:
    python3 scripts/browsecomp_reachability_probe.py \
        --config scripts/browsecompplus_full_conf.json \
        [--limit N] [--workers 8] [--top-k 30] [--out outputs/reachability]

ES connection defaults come from conf/service_conf.yaml's local dev values and
can be overridden with --es/--es-user/--es-password. The index and the kb_id are
auto-discovered from the first expected document name found in the corpus.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.nlp import rag_tokenizer


def es_request(base: str, auth: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def load_questions(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def discover_scope(es: str, auth: str, sample_doc: str) -> tuple[str, str]:
    """Return (index, kb_id) by finding one expected document in the corpus."""
    idx = es_request(es, auth, "/_cat/indices/ragflow*?h=index&format=json")
    for row in idx:
        index = row.get("index", "")
        if index.endswith("_doc_meta"):
            continue
        body = {
            "size": 1,
            "_source": ["kb_id"],
            "query": {"term": {"docnm_kwd": f"{sample_doc}.md"}},
        }
        try:
            res = es_request(es, auth, f"/{index}/_search", body)
        except urllib.error.HTTPError:
            continue
        hits = res.get("hits", {}).get("hits", [])
        if hits:
            return index, hits[0]["_source"]["kb_id"]
    raise SystemExit(f"cannot locate document {sample_doc}.md in any ragflow_* index")


def bm25_docs(es: str, auth: str, index: str, kb_id: str, text: str, top_k: int) -> list[str]:
    """BM25 the text (tokenised by rag_tokenizer, like RAGFlow does) and return
    the matching document stems in rank order."""
    tokens = rag_tokenizer.tokenize(text or "")
    if not tokens.strip():
        return []
    body = {
        "size": top_k,
        "_source": ["docnm_kwd"],
        "query": {
            "bool": {
                "filter": [{"term": {"kb_id": kb_id}}],
                "must": [{"match": {"content_ltks": {"query": tokens, "operator": "or"}}}],
            }
        },
    }
    res = es_request(es, auth, f"/{index}/_search", body)
    stems: list[str] = []
    for hit in res.get("hits", {}).get("hits", []):
        name = hit.get("_source", {}).get("docnm_kwd") or ""
        stem = os.path.splitext(os.path.basename(name))[0]
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def rank_of(expected: set[str], ranked: list[str]) -> int | None:
    for i, stem in enumerate(ranked, 1):
        if stem in expected:
            return i
    return None


def gold_in_docs(corpus_path: str, expected: list[str], gold: str) -> tuple[int, int]:
    """(documents containing the gold string, documents readable)."""
    needle = gold.strip().lower()
    if not needle:
        return 0, 0
    found = readable = 0
    for stem in expected:
        path = os.path.join(corpus_path, f"{stem}.md")
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read().lower()
        except OSError:
            continue
        readable += 1
        if needle in text:
            found += 1
    return found, readable


def probe_one(q: dict[str, Any], args: argparse.Namespace, index: str, kb_id: str) -> dict[str, Any]:
    expected = {str(d) for d in (q.get("expected_doc_ids") or [])}
    gold = str(q.get("gold_answer") or "")
    literal_ranked = bm25_docs(args.es, args.auth, index, kb_id, q.get("question", ""), args.top_k)
    gold_ranked = bm25_docs(args.es, args.auth, index, kb_id, gold, args.top_k)
    in_doc, readable = gold_in_docs(args.corpus_path, sorted(expected), gold)

    literal_rank = rank_of(expected, literal_ranked)
    gold_rank = rank_of(expected, gold_ranked)

    if literal_rank:
        klass = "A_literal"
    elif gold_rank:
        klass = "B_paraphrase_limited"
    elif in_doc:
        klass = "C_hybrid_only"
    else:
        klass = "D_unanswerable_or_mismapped"

    return {
        "id": q.get("id"),
        "class": klass,
        "literal_rank": literal_rank,
        "gold_rank": gold_rank,
        "literal_top": literal_ranked[:3],
        "gold_top": gold_ranked[:3],
        "expected": sorted(expected),
        "expected_count": len(expected),
        "gold_in_expected_docs": in_doc,
        "expected_docs_readable": readable,
        "gold_answer": gold,
        "question": q.get("question", "")[:400],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="scripts/browsecompplus_full_conf.json")
    ap.add_argument("--questions", help="override questions jsonl path")
    ap.add_argument("--corpus", help="override corpus directory")
    ap.add_argument("--es", default="http://localhost:1200")
    ap.add_argument("--es-user", default="elastic")
    ap.add_argument("--es-password", default="infini_rag_flow")
    ap.add_argument("--limit", type=int, default=0, help="probe only the first N questions (0 = all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--out", help="output directory (default outputs/reachability_<timestamp>)")
    args = ap.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
    ds = cfg.get("dataset", {})
    args.questions = args.questions or ds.get("questions_path")
    args.corpus_path = args.corpus or ds.get("corpus_path")
    if not args.questions or not args.corpus_path:
        raise SystemExit("questions path and corpus path are required (config or flags)")
    args.auth = base64.b64encode(f"{args.es_user}:{args.es_password}".encode()).decode()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    sample = str((questions[0].get("expected_doc_ids") or ["0"])[0])
    index, kb_id = discover_scope(args.es, args.auth, sample)
    print(f"index={index} kb_id={kb_id} questions={len(questions)} top_k={args.top_k}", flush=True)

    out_dir = args.out or os.path.join("outputs", f"reachability_{datetime.now().astimezone():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe_one, q, args, index, kb_id): q for q in questions}
        for done, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - one bad question must not stop the sweep
                q = futures[fut]
                rows.append({"id": q.get("id"), "class": "E_error", "error": str(exc)})
            if done % 50 == 0:
                print(f"  probed {done}/{len(questions)}", flush=True)

    rows.sort(key=lambda r: str(r.get("id")))
    with open(os.path.join(out_dir, "probe.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    total = len(rows)
    summary = {
        "total": total,
        "class_counts": counts,
        "class_pct": {k: round(100.0 * v / max(total, 1), 1) for k, v in counts.items()},
        "literal_hit": sum(1 for r in rows if r.get("literal_rank")),
        "gold_hit": sum(1 for r in rows if r.get("gold_rank")),
        "gold_in_evidence": sum(1 for r in rows if r.get("gold_in_expected_docs")),
        "index": index,
        "kb_id": kb_id,
        "top_k": args.top_k,
        "questions_path": args.questions,
        "corpus_path": args.corpus_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    for k, v in summary["class_pct"].items():
        print(f"  {k:28s} {counts[k]:4d}  ({v}%)")
    print(f"  literal_hit@{args.top_k}: {summary['literal_hit']}/{total}")
    print(f"  gold_hit@{args.top_k}:    {summary['gold_hit']}/{total}")
    print(f"  gold present in own evidence docs: {summary['gold_in_evidence']}/{total}")

    for klass in ("B_paraphrase_limited", "D_unanswerable_or_mismapped"):
        sample_rows = [r for r in rows if r["class"] == klass][:5]
        if sample_rows:
            print(f"\n--- samples: {klass} ---")
            for r in sample_rows:
                print(f"  #{r['id']} gold={r['gold_answer'][:40]!r} in_doc={r.get('gold_in_expected_docs')} gold_top={r.get('gold_top')}")
                print(f"      q: {r['question'][:150]}")
    print(f"\nwrote {out_dir}/probe.jsonl and summary.json")


if __name__ == "__main__":
    main()
