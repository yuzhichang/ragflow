#!/usr/bin/env python3
"""Full-corpus coverage sweep: does the index still serve every passage of every document?

`browsecomp_index_coverage.py` answers this for a handful of documents picked out by question ids.
This script answers it for the whole re-parse target set (the 8,398 documents recorded in the
re-parse state file), because "MySQL chunk_num equals the ES doc count" only proves the two
counters agree - it cannot see a document whose chunks dropped a passage.

Pipeline (both halves run at the same time, which is the point):

  producer (parent, one thread)   walks ES with search_after, page by page, and emits a document
                                  as soon as all of its chunks have been seen (the expected count
                                  comes from MySQL document.chunk_num, so a document never waits
                                  for the whole sweep);
  workers (process pool)          answer, per document, whether the file's text is covered, by
                                  sliding a window over the whitespace-normalised FILE and asking
                                  whether that window occurs in the whitespace-normalised chunk
                                  text. str.__contains__ is C-level, so this is ~2 orders of
                                  magnitude faster than a difflib pass over the same pair, and it
                                  answers the same question: a run of consecutive uncovered
                                  windows at least `--min-gap` long is a passage the index does not
                                  serve. (Markup the ingest strips, such as table pipes or bold
                                  markers, leaves only short uncovered runs and is filtered out by
                                  that threshold - same rule the difflib probe uses.)

`--method difflib` keeps the original interval check for validating the fast one on a sample.

Results are appended to `--out/coverage.jsonl` as they arrive; documents already present are
skipped, so a long sweep can be resumed.

Usage:
    python3 scripts/browsecomp_full_coverage.py --limit 30            # calibrate
    python3 scripts/browsecomp_full_coverage.py --workers 12          # full sweep
    python3 scripts/browsecomp_full_coverage.py --limit 30 --method difflib   # cross-check
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import urllib.request

STATE_DEFAULT = os.path.expanduser("~/.cache/browsecomp_reparse_state.json")
CORPUS_DEFAULT = "/home/zhichyu/Downloads/data/browsecomp-plus/corpus"
OUT_DEFAULT = "outputs/full_coverage_20260915"
ES_PAGE = 4000

WHITESPACE_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def es_request(base: str, auth: str, path: str, body: dict | None = None) -> dict:
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Authorization": "Basic " + auth, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def es_password() -> str:
    out = subprocess.run(
        "grep -E '^ELASTIC_PASSWORD' docker/.env | cut -d= -f2",
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def discover_scope() -> tuple[str, str]:
    for path in sorted(glob.glob("outputs/reachability_*/summary.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            continue
        if summary.get("index") and summary.get("kb_id"):
            return summary["index"], summary["kb_id"]
    raise SystemExit("no outputs/reachability_*/summary.json found; pass --index and --kb-id")


def target_documents(kb: str, state_path: str) -> dict[str, tuple[str, int]]:
    """name -> (doc id, mysql chunk_num) for the documents in the re-parse target set."""
    with open(state_path, encoding="utf-8") as fh:
        triggered = set(json.load(fh)["triggered"])
    sql = f"SELECT id, name, chunk_num FROM document WHERE kb_id='{kb}';"
    out = subprocess.run(
        ["docker", "exec", "docker-mysql-1", "sh", "-c", f"MYSQL_PWD=infini_rag_flow mysql -N -uroot rag_flow -e {json.dumps(sql)}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    docs: dict[str, tuple[str, int]] = {}
    for line in out.splitlines():
        fields = line.split("\t")
        if len(fields) >= 3 and fields[0] in triggered:
            docs[fields[1]] = (fields[0], int(fields[2] or 0))
    return docs


def stream_documents(es: str, auth: str, index: str, kb: str, wanted: dict[str, tuple[str, int]]):
    """Yield (name, doc_id, mysql_chunks, chunk texts) as soon as a document is complete.

    One ES pass, paged with search_after; a document is emitted once as many chunks have been seen
    as MySQL says it has (or immediately when its name leaves the pending set, so the tail of a
    page is not held back).
    """
    pending = {name: {"id": did, "mysql": mysql, "texts": []} for name, (did, mysql) in wanted.items()}
    body = {
        "size": ES_PAGE,
        "_source": ["docnm_kwd", "content_with_weight"],
        "query": {"bool": {"filter": [{"term": {"kb_id": kb}}, {"terms": {"docnm_kwd": sorted(pending)}}]}},
        "sort": [{"_doc": "asc"}],
    }
    after = None
    while True:
        if after is not None:
            body["search_after"] = after
        res = es_request(es, auth, f"/{index}/_search", body)
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            break
        for hit in hits:
            src = hit.get("_source") or {}
            entry = pending.get(src.get("docnm_kwd"))
            if entry is not None:
                entry["texts"].append(src.get("content_with_weight") or "")
        after = hits[-1].get("sort")
        ready = [n for n, e in pending.items() if len(e["texts"]) >= e["mysql"]]
        for name in ready:
            entry = pending.pop(name)
            yield name, entry["id"], entry["mysql"], entry["texts"]
        if len(hits) < ES_PAGE:
            break
    for name, entry in pending.items():  # documents MySQL expected nothing from, or short counts
        yield name, entry["id"], entry["mysql"], entry["texts"]


def uncovered_spans(file_norm: str, covered: bytearray, min_gap: int) -> list[dict]:
    gaps: list[dict] = []
    start = None
    for pos, flag in enumerate(covered):
        if not flag and start is None:
            start = pos
        elif flag and start is not None:
            if pos - start >= min_gap:
                gaps.append({"start": start, "length": pos - start, "context": file_norm[max(0, start - 40) : pos + 40]})
            start = None
    if start is not None and len(file_norm) - start >= min_gap:
        gaps.append({"start": start, "length": len(file_norm) - start, "context": file_norm[max(0, start - 40) :]})
    return gaps


def coverage_of(job: tuple) -> dict:
    name, doc_id, mysql_chunks, texts, corpus_dir, min_gap, method, window, stride = job
    path = os.path.join(corpus_dir, name)
    if not os.path.exists(path):
        return {"name": name, "id": doc_id, "mysql_chunks": mysql_chunks, "error": "FILE MISSING"}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        file_norm = normalise(fh.read())
    joined = normalise("\n".join(texts))
    base = {"name": name, "id": doc_id, "mysql_chunks": mysql_chunks, "chunk_count": len(texts), "file_chars": len(file_norm)}
    if not file_norm:
        return {**base, "error": "EMPTY FILE"}
    if not joined:
        return {**base, "covered_chars": 0, "coverage_pct": 0.0, "gap_chars": len(file_norm), "gaps": []}

    if method == "difflib":
        import difflib

        matcher = difflib.SequenceMatcher(None, file_norm, joined, autojunk=False)
        covered = bytearray(len(file_norm))
        for block in matcher.get_matching_blocks():
            if block.size > 8:
                covered[block.a : block.a + block.size] = b"\x01" * block.size
    else:
        covered = bytearray(len(file_norm))
        pos = 0
        end = len(file_norm)
        while pos < end:
            piece = file_norm[pos : pos + window]
            if piece in joined:
                covered[pos : pos + len(piece)] = b"\x01" * len(piece)
                pos += max(len(piece) - stride, 1)  # a match is covered; step to its tail
            else:
                pos += stride

    gaps = uncovered_spans(file_norm, covered, min_gap)
    covered_chars = sum(covered)
    return {
        **base,
        "covered_chars": covered_chars,
        "coverage_pct": round(100.0 * covered_chars / len(file_norm), 2),
        "gap_chars": len(file_norm) - covered_chars,
        "gaps": gaps[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--es", default="http://localhost:1200")
    ap.add_argument("--es-user", default="elastic")
    ap.add_argument("--es-password")
    ap.add_argument("--index")
    ap.add_argument("--kb-id")
    ap.add_argument("--state", default=STATE_DEFAULT)
    ap.add_argument("--corpus", default=CORPUS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, help="only the first N documents (calibration)")
    ap.add_argument("--method", choices=["window", "difflib"], default="window")
    ap.add_argument("--window", type=int, default=200, help="window length for --method window")
    ap.add_argument("--stride", type=int, default=50, help="window step for --method window")
    ap.add_argument("--min-gap", type=int, default=40, help="report uncovered runs at least this long")
    ap.add_argument("--every", type=int, default=200)
    args = ap.parse_args()

    args.es_password = args.es_password or es_password()
    auth = base64.b64encode(f"{args.es_user}:{args.es_password}".encode()).decode()
    index, kb = (args.index, args.kb_id) if args.index and args.kb_id else discover_scope()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"coverage_{args.method}.jsonl")
    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["name"])
                except (ValueError, KeyError):
                    continue
    docs = target_documents(kb, args.state)
    todo = {n: v for n, v in sorted(docs.items()) if n not in done}
    if args.limit:
        todo = dict(list(todo.items())[: args.limit])
    print(f"scope index={index} kb={kb} | target {len(docs)} docs | measured already {len(done)} | to measure {len(todo)} | method {args.method} | workers {args.workers}", flush=True)

    started = time.time()
    measured = 0
    pool_jobs = ((name, did, mysql, texts, args.corpus, args.min_gap, args.method, args.window, args.stride) for name, did, mysql, texts in stream_documents(args.es, auth, index, kb, todo))
    with mp.Pool(args.workers) as pool, open(out_path, "a", encoding="utf-8") as sink:
        for row in pool.imap_unordered(coverage_of, pool_jobs, chunksize=1):
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            measured += 1
            if measured % args.every == 0 or measured == len(todo):
                elapsed = time.time() - started
                rate = measured / max(elapsed, 1e-9)
                print(f"  {measured}/{len(todo)}  {rate:.1f} doc/s  ETA {(len(todo) - measured) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    with open(out_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    errs = [r for r in rows if r.get("error")]
    low = sorted([r for r in rows if r.get("coverage_pct") is not None and r["coverage_pct"] < 99.0], key=lambda r: r["coverage_pct"])
    print(f"\n=== summary: {len(rows)} documents in {(time.time() - started) / 60:.1f} min ===")
    print(f"  file missing / errors : {len(errs)}")
    print(f"  covered >=99%         : {len(rows) - len(errs) - len(low)}")
    print(f"  below 99%             : {len(low)}")
    for row in low[:30]:
        print(f"    {row['name']:<18} coverage={row['coverage_pct']}% gap={row['gap_chars']} chars chunks={row.get('chunk_count')}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
