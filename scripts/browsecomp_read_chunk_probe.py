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

# The deliverable names each cited document as ``doc: `49409.md` `` (the field the
# auditor checks against the chunk's own doc_name), so the cited document set is
# read off the text exactly like the cited chunk set. The backtick matters: the
# value rides in a code span and a pattern anchored straight after `doc:` misses
# every citation (which made a passing run look like it cited nothing at all).
CITED_DOC_RE = re.compile(r"doc:\s*[`\"']*(\d+)\.md")


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


def load_grades(path: str) -> dict[str, bool]:
    """question id -> graded outcome, from a run's own leaderboard.json."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            board = json.load(fh)
    except (OSError, ValueError):
        return {}
    out: dict[str, bool] = {}
    for metric in board.get("per_query_metrics") or []:
        correct = metric.get("correct")
        if isinstance(correct, bool):
            out[str(metric.get("query_id"))] = correct
    return out


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


# A question's ANCHORS: the tokens a locate query would use as a rare term - a
# capitalized word (a proper noun, the question's own named entities) or a
# 4-digit number (a year). The producer prompt asks for exactly this class of
# query ("an ENTITY-ANCHORED single-term query on the rarest proper noun"), so a
# mechanical read-but-uncited check would have to lean on the same vocabulary.
# Capitalized function words are dropped; capitalized words at the START of a
# sentence survive, which inflates the anchor set slightly and is therefore the
# conservative direction for the noise floor this probe is measuring.
ANCHOR_RE = re.compile(r"\b[A-Z][A-Za-z'\-]{2,}\b|\b\d{4}\b")
# The same shape without the years: NAMES, for the "did you read something you
# never accounted for" rule below.
NOVEL_NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]{2,}\b")
ANCHOR_STOPWORDS = {
    "the",
    "this",
    "that",
    "these",
    "those",
    "a",
    "an",
    "in",
    "on",
    "at",
    "of",
    "for",
    "and",
    "or",
    "but",
    "as",
    "by",
    "to",
    "from",
    "with",
    "without",
    "into",
    "over",
    "under",
    "between",
    "during",
    "after",
    "before",
    "it",
    "its",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "you",
    "your",
    "i",
    "we",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "does",
    "did",
    "do",
    "not",
    "no",
    "yes",
    "if",
    "than",
    "then",
    "there",
    "here",
    "one",
    "two",
    "three",
    "first",
    "last",
    "same",
    "other",
    "another",
    "also",
    "both",
    "each",
    "some",
    "any",
    "all",
    "more",
    "most",
    "less",
    "least",
    "many",
    "much",
    "such",
    "own",
}


def question_anchors(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in ANCHOR_RE.findall(text or ""):
        token = match.lower()
        if token not in ANCHOR_STOPWORDS and token not in seen:
            seen[token] = None
    return list(seen)


def chunks_by_id(es: str, auth: str, index: str, kb_id: str, ids: list[str], cache: dict[str, Any]) -> dict[str, str]:
    """Fetch the text of specific chunks by their ids (the ids deliverables cite).

    Fetched in one terms query: the uncited set is the population this whole probe
    exists to characterise, and it is only known AFTER the read set is known, so
    it cannot ride the per-document fetch above.
    """
    key = "byid:" + ",".join(sorted(ids))
    if key in cache:
        return cache[key]
    out: dict[str, str] = {}
    for start in range(0, len(ids), 200):
        batch = ids[start : start + 200]
        body = {
            "size": len(batch),
            "_source": ["content_with_weight"],
            "query": {"bool": {"filter": [{"term": {"kb_id": kb_id}}], "must": [{"terms": {"_id": batch}}]}},
        }
        res = es_request(es, auth, f"/{index}/_search", body)
        for hit in res.get("hits", {}).get("hits", []):
            out[hit.get("_id") or ""] = (hit.get("_source") or {}).get("content_with_weight") or ""
    cache[key] = out
    return out


def _median(values: list[Any]) -> Any:
    """Median of the numbers present, or None when nothing was measured."""
    nums = sorted(v for v in values if isinstance(v, (int, float)))
    return nums[len(nums) // 2] if nums else None


def chunk_docs_by_id(es: str, auth: str, index: str, kb_id: str, ids: list[str], cache: dict[str, Any]) -> dict[str, str]:
    """Map chunk ids to their document stem (the `doc:` a deliverable cites)."""
    key = "docs:" + ",".join(sorted(ids))
    if key in cache:
        return cache[key]
    out: dict[str, str] = {}
    for start in range(0, len(ids), 200):
        batch = ids[start : start + 200]
        body = {
            "size": len(batch),
            "_source": ["docnm_kwd"],
            "query": {"bool": {"filter": [{"term": {"kb_id": kb_id}}], "must": [{"terms": {"_id": batch}}]}},
        }
        res = es_request(es, auth, f"/{index}/_search", body)
        for hit in res.get("hits", {}).get("hits", []):
            name = str((hit.get("_source") or {}).get("docnm_kwd") or "")
            out[hit.get("_id") or ""] = name.removesuffix(".md")
    cache[key] = out
    return out


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
    # The two accounts are different populations on purpose: `deep_read_chunks`
    # counts RENDERS (the same chunk read twice counts twice, which is what its
    # docstring has always said) while the id list dedupes. The invariant the
    # ledger can be checked against is therefore ⊆, not =. A unique-id count
    # ABOVE the render count would mean the ledger records something the counter
    # never saw, which would invalidate every rate below.
    renders = row.get("deep_read_chunks")
    out["deep_read_renders"] = renders
    out["counts_agree"] = (not deep) or (isinstance(renders, int) and len(deep) <= renders)

    gold = str(out["gold_answer"] or "").strip().lower()
    gold_chunks: list[dict[str, Any]] = []
    if args.index and args.kb_id and gold and out["expected_docs"]:
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

    # The uncited set, split by what a CHECK could know about it at run time.
    # "contains the gold string" is only available afterwards (it is the check's
    # ceiling, not its behaviour), while "contains one of the question's anchor
    # terms" is what a mechanical read-but-uncited rule would actually fire on -
    # and therefore the number that decides whether such a rule would drown in
    # its own noise. Measured without this split, a passing question with 62
    # uncited deep reads (q575) looks the same as a failing one with 43 (q297).
    uncited_texts = chunks_by_id(args.es, args.auth, args.index, args.kb_id, uncited_deep, cache) if args.index and args.kb_id else {}
    anchors = question_anchors((question or {}).get("question") or "")
    with_gold = [cid for cid in uncited_deep if gold and gold in uncited_texts.get(cid, "").lower()]
    with_anchor = [cid for cid in uncited_deep if any(anchor in uncited_texts.get(cid, "").lower() for anchor in anchors)]
    out["anchors"] = anchors
    out["uncited_deep_with_gold"] = len(with_gold) if uncited_texts else None
    out["uncited_deep_with_anchor"] = len(with_anchor) if uncited_texts else None
    out["uncited_deep_fetched"] = len(uncited_texts) if uncited_texts else None

    # A cheaper criterion than "does it hold the answer": does the uncited chunk
    # carry a NAME the deliverable never mentions at all? The lexical-anchor split
    # above fails on its own (a passing run showed 55 such chunks against a failing
    # run's 18), because the question's own anchors are vocabulary the run already
    # HAS. A name that appears nowhere in the deliverable is different in kind: it
    # is evidence the run read something it never accounted for. Measured here as
    # the noise floor of that rule - capitalized tokens in the chunk that are
    # absent (case-insensitively) from the whole deliverable text.
    def novel_names(text: str) -> int:
        names = {token.lower() for token in NOVEL_NAME_RE.findall(text)} - ANCHOR_STOPWORDS
        return sum(1 for name in names if name not in answer.lower())

    novel = [cid for cid in uncited_deep if uncited_texts.get(cid) and novel_names(uncited_texts[cid])]
    out["uncited_deep_with_novel_name"] = len(novel) if uncited_texts else None

    # The document-level version of the same question, which is the cheapest form
    # of it: whole documents that were read (a chunk deep-read) and then never
    # cited anywhere. A run that cites everything it opened is using its reading;
    # one that opens whole documents and drops them is not. Read from the
    # deliverable's own `doc:` fields for the cited side, so no extra judgement is
    # smuggled in.
    if args.index and args.kb_id and deep:
        read_docs = {d for d in chunk_docs_by_id(args.es, args.auth, args.index, args.kb_id, deep, cache).values() if d}
        cited_docs = {m.group(1) for m in CITED_DOC_RE.finditer(answer)}
        out["read_docs"] = len(read_docs)
        out["cited_docs"] = len(cited_docs)
        out["uncited_docs"] = len(read_docs - cited_docs)
    else:
        out["read_docs"] = out["cited_docs"] = out["uncited_docs"] = None
    # An uninstrumented row (a backend built before the ledger) has an EMPTY read
    # set, which would read as "the gold chunk was never read" - the opposite of
    # what an absent ledger can support. Such a row is undecided, not a finding.
    if not out["instrumented"]:
        out["verdict"] = "uninstrumented_no_ledger"
    elif not args.index or not args.kb_id:
        out["verdict"] = "es_skipped__ledger_only"
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
    ap.add_argument(
        "--leaderboard",
        help="the run's leaderboard.json: joins each row's grade onto its verdict, so the mechanism can be cross-tabulated against the outcome (default: sibling of --answers)",
    )
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

    # The grade, when the run's own leaderboard is available: the verdict's value
    # is that it can be checked against an outcome, so a run whose grade is absent
    # (still being judged) says so rather than silently reporting a mechanism.
    grades = load_grades(args.leaderboard or os.path.join(os.path.dirname(args.answers), "leaderboard.json"))
    for r in out_rows:
        r["correct"] = grades.get(r["id"])
    graded = [r for r in out_rows if r["correct"] is not None]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "probe.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)

    print(f"{'id':>6} {'deep':>5} {'cited':>6} {'uncited':>8} {'goldCk':>7} {'goldRead':>9} {'goldCited':>10}  {'grade':<5} verdict")
    for r in out_rows:
        grade = "-" if r["correct"] is None else ("PASS" if r["correct"] else "FAIL")
        print(f"{r['id']:>6} {r['deep_read']:>5} {r['cited']:>6} {r['uncited_deep']:>8} {r['gold_chunks_total']:>7} {r['gold_chunks_read']:>9} {r['gold_chunks_cited']:>10}  {grade:<5} {r['verdict']}")

    verdict_x_grade: dict[str, dict[str, int]] = {}
    verdict_mismatch: list[str] = []
    if graded:
        print("\nverdict x grade (the check that decides whether the mechanism is a finding):")
        for r in graded:
            bucket = verdict_x_grade.setdefault(r["verdict"], {"PASS": 0, "FAIL": 0})
            bucket["PASS" if r["correct"] else "FAIL"] += 1
        for verdict in sorted(verdict_x_grade, key=lambda v: -(verdict_x_grade[v]["PASS"] + verdict_x_grade[v]["FAIL"])):
            bucket = verdict_x_grade[verdict]
            print(f"  {verdict:<50} PASS={bucket['PASS']:<3} FAIL={bucket['FAIL']}")
        verdict_mismatch = [r["id"] for r in graded if r["verdict"].startswith("gold_read_and_cited") != bool(r["correct"])]
        print(f"  rows where verdict and grade DISAGREE: {verdict_mismatch or 'none'}")
    else:
        print("\n(no grades yet: the run's leaderboard.json is missing or still being judged)")
    for r in out_rows:
        print(f"{r['id']:>6} {r['deep_read']:>5} {r['cited']:>6} {r['uncited_deep']:>8} {r['gold_chunks_total']:>7} {r['gold_chunks_read']:>9} {r['gold_chunks_cited']:>10}  {r['verdict']}")

    print("\nuncited deep reads, split by what a CHECK could know about them: 'gold' = contains")
    print("the gold string (the check's ceiling, knowable only afterwards); 'anchor' = contains")
    print("one of the question's proper nouns/years (what a mechanical rule would fire on = its")
    print("noise floor). Without this split a PASSING run with 62 uncited reads (q575) looks")
    print("exactly like a FAILING one with 43 (q297).")
    print(f"{'id':>6} {'uncited':>8} {'uncWithGold':>12} {'uncWithAnchor':>14} {'uncNovelName':>13} {'uncitedDocs':>11} {'readDocs':>9} {'anchors':>8}")
    for r in out_rows:
        print(
            f"{r['id']:>6} {r['uncited_deep']:>8} {r['uncited_deep_with_gold']!s:>12} {r['uncited_deep_with_anchor']!s:>14} {r['uncited_deep_with_novel_name']!s:>13} {r['uncited_docs']!s:>11} {r['read_docs']!s:>9} {len(r['anchors']):>8}"
        )
    print("\n(uncitedDocs = documents with at least one chunk deep-read that the deliverable never")
    print(" cites; readDocs = documents the run opened at all. Both are cheap, deliverable-side")
    print(" reads - the one shape of this question a gate could actually compute.)")

    uninstrumented = [r["id"] for r in out_rows if not r["instrumented"]]
    disagree = [r["id"] for r in out_rows if not r["counts_agree"]]
    summary = {
        "rows": len(out_rows),
        "uninstrumented": uninstrumented,
        "counts_disagree_with_ids": disagree,
        "verdicts": {v: sum(1 for r in out_rows if r["verdict"] == v) for v in sorted({r["verdict"] for r in out_rows})},
        "median_deep_read": sorted(r["deep_read"] for r in out_rows)[len(out_rows) // 2] if out_rows else None,
        "median_uncited_deep": sorted(r["uncited_deep"] for r in out_rows)[len(out_rows) // 2] if out_rows else None,
        "verdict_x_grade": verdict_x_grade,
        "verdict_grade_mismatch": verdict_mismatch,
        "median_uncited_with_gold": _median([r["uncited_deep_with_gold"] for r in out_rows]),
        "median_uncited_with_anchor": _median([r["uncited_deep_with_anchor"] for r in out_rows]),
        "median_uncited_with_novel_name": _median([r["uncited_deep_with_novel_name"] for r in out_rows]),
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
