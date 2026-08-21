#!/usr/bin/env python3
"""Re-parse a list of documents in the BrowseComp-Plus KB and, optionally, wait
for the ingest to finish.

The list is produced by the corpus damage sweep
(``outputs/markdown_text_loss_docs.json``): documents whose indexed text was
truncated by the markdown parser text-loss bugs, so re-parsing them is what
actually repairs the evidence in the KB.

Re-parsing is a queue operation: the API deletes the old chunks and publishes a
task that the *ingestor* consumes. Without a running ingestor the documents end
up with run=3 and chunk_num=0 - see the ledger entry
``_reparse_experiment_20260915`` in scripts/browsecompplus_retry_conf.json.

The run is resumable: every triggered batch is recorded in the state file, so a
second invocation skips what is already queued.

Usage:
    python3 scripts/browsecomp_reparse_docs.py --list outputs/markdown_text_loss_docs.json
    python3 scripts/browsecomp_reparse_docs.py --docs 34297,64576 --dry-run
    python3 scripts/browsecomp_reparse_docs.py --list ... --wait
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests

KB_DEFAULT = "5abdf3f1b8954735aa3fa2e03041e479"
API_DEFAULT = "http://localhost:9384"
TOKEN_DEFAULT = "ragflow-xGRCgHIMf6AsBy286eX8idKWQFY5qSrzpzFRKjIu8mk"
STATE_DEFAULT = os.path.expanduser("~/.cache/browsecomp_reparse_state.json")
MYSQL_CONTAINER = "docker-mysql-1"
MYSQL_PASSWORD = "infini_rag_flow"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def mysql(sql: str) -> list[list[str]]:
    """Run SQL through the mysql container and return the tab-separated rows."""
    out = subprocess.run(
        ["docker", "exec", MYSQL_CONTAINER, "sh", "-c", f"MYSQL_PWD={MYSQL_PASSWORD} mysql -N -uroot rag_flow -e {json.dumps(sql)}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"mysql failed: {out.stderr.strip()}")
    return [line.split("\t") for line in out.stdout.splitlines() if line.strip()]


def load_stems(args) -> list[str]:
    if args.docs:
        return [d.strip().removesuffix(".md") for d in args.docs.split(",") if d.strip()]
    with open(args.list, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.values()
    stems = []
    for row in data:
        name = row["doc"] if isinstance(row, dict) else row
        stems.append(str(name).removesuffix(".md"))
    return stems


def kb_documents(kb: str) -> dict[str, dict]:
    """Map document name (``<stem>.md``) to its id / chunk_num / run state."""
    rows = mysql(f"SELECT name, id, chunk_num, run FROM document WHERE kb_id='{kb}';")
    out = {}
    for name, doc_id, chunk_num, run in rows:
        out[name] = {"id": doc_id, "chunk_num": int(chunk_num or 0), "run": run}
    return out


def load_state(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"triggered": [], "before": {}}


def save_state(path: str, state: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)


def trigger(api: str, token: str, kb: str, doc_ids: list[str]) -> tuple[bool, str]:
    try:
        resp = requests.post(
            f"{api}/api/v1/datasets/{kb}/chunks",
            headers={"Authorization": f"Bearer {token}"},
            json={"document_ids": doc_ids},
            timeout=600,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"request error: {exc}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:160]}"
    body = resp.json()
    if body.get("code") != 0:
        return False, f"code={body.get('code')} message={body.get('message')}"
    return True, "success"


def _queue(args, state: dict, doc_ids: list[str], label: str) -> list[str]:
    if not doc_ids:
        log(f"{label}: nothing to queue")
        return []
    batches = [doc_ids[i : i + args.batch] for i in range(0, len(doc_ids), args.batch)]
    queued = 0
    for i, batch in enumerate(batches, 1):
        if args.dry_run:
            log(f"dry-run {label} batch {i}/{len(batches)}: {len(batch)} docs (e.g. {batch[:3]})")
            queued += len(batch)
            continue
        for attempt in range(1, args.retries + 1):
            ok, detail = trigger(args.api, args.token, args.kb, batch)
            if ok:
                break
            log(f"{label} batch {i}/{len(batches)} attempt {attempt} failed: {detail}")
            time.sleep(min(30, 3 * attempt))
        else:
            log(f"{label} batch {i}/{len(batches)} GIVING UP; rerun the script to resume")
            break
        state["triggered"].extend(batch)
        save_state(args.state, state)
        queued += len(batch)
        if i % 5 == 0 or i == len(batches):
            log(f"{label} batch {i}/{len(batches)} queued ({queued}/{len(doc_ids)} docs this run)")
        time.sleep(args.gap)
    return doc_ids[:queued]


def queue_all(args, state: dict) -> list[str]:
    stems = load_stems(args)
    log(f"targets in list: {len(stems)}")
    docs = kb_documents(args.kb)
    log(f"documents in kb {args.kb}: {len(docs)}")

    missing = [s for s in stems if f"{s}.md" not in docs]
    if missing:
        log(f"WARNING: {len(missing)} stems have no document row (first 5: {missing[:5]})")

    for stem in stems:
        row = docs.get(f"{stem}.md")
        if row and row["id"] not in state["before"]:
            state["before"][row["id"]] = row["chunk_num"]

    todo = [docs[f"{s}.md"]["id"] for s in stems if f"{s}.md" in docs and docs[f"{s}.md"]["id"] not in set(state["triggered"])]
    log(f"already queued in an earlier run: {len(state['triggered'])}; to queue now: {len(todo)}")
    return _queue(args, state, todo, "queue")


def retry_failed(args, state: dict) -> list[str]:
    """Re-queue documents whose latest ingestion task FAILED.

    A re-parse deletes the old chunks before parsing, so a document that failed
    afterwards (typically the embedding backend answering 429 Too Many Requests
    once a very large document needed thousands of embeddings) is left with
    nothing. document.chunk_num cannot be used as the failure signal: a document
    that is merely queued also sits at chunk_num=0, so retrying on that would
    re-queue the whole backlog over and over. The ingestion task status is the
    honest signal.
    """
    failed = sorted(failed_documents(args.kb))
    log(f"retry: {len(failed)} documents in kb {args.kb} have a FAILED latest task")
    return _queue(args, state, failed, "retry")


def failed_documents(kb: str) -> set[str]:
    """Document ids whose latest ingestion task FAILED."""
    rows = mysql(
        "SELECT t.document_id FROM ingestion_task t "
        "JOIN (SELECT document_id, MAX(create_time) AS mx FROM ingestion_task "
        f"WHERE dataset_id='{kb}' GROUP BY document_id) x "
        "ON x.document_id = t.document_id AND x.mx = t.create_time "
        "WHERE t.status='FAILED';"
    )
    return {r[0] for r in rows}


def wait_for(args, doc_ids: list[str]) -> None:
    """Poll until no queued document is still waiting for chunks, then summarise.

    Completion is measured by chunk_num, not by document.run: a re-parse deletes
    the chunks first, and a document that is merely queued already reads run=3
    with chunk_num=0, so run alone cannot tell "queued" from "finished". A
    document whose latest task FAILED is not pending either - it needs the retry
    pass, not more waiting.
    """
    ids = set(doc_ids)
    state_by_id: dict[str, tuple[str, int]] = {}
    failed: set[str] = set()
    while True:
        rows = mysql(f"SELECT id, run, chunk_num FROM document WHERE kb_id='{args.kb}';")
        state_by_id = {r[0]: (r[1], int(r[2] or 0)) for r in rows if r[0] in ids}
        failed = failed_documents(args.kb) & ids
        pending = sum(1 for i, v in state_by_id.items() if v[1] == 0 and i not in failed)
        with_chunks = sum(1 for v in state_by_id.values() if v[1] > 0)
        log(f"queued={len(state_by_id)} with_chunks={with_chunks} pending={pending} failed={len(failed)}")
        if pending == 0:
            break
        time.sleep(args.poll)

    # Post-run verification: how many documents gained chunks (the fix recovered
    # text, so the chunker emits more chunks than the damaged parse did).
    before = load_state(args.state).get("before", {})
    improved = same = 0
    for doc_id, (_run, chunk_num) in state_by_id.items():
        was = before.get(doc_id)
        if was is None:
            continue
        if chunk_num > was:
            improved += 1
        else:
            same += 1
    log(f"RESULT: {improved} documents gained chunks, {same} unchanged, {len(failed)} failed")
    if failed:
        log(f"failed ids: {','.join(sorted(failed)[:20])}{' ...' if len(failed) > 20 else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="outputs/markdown_text_loss_docs.json", help="JSON list of damaged documents")
    ap.add_argument("--docs", help="comma separated stems, overrides --list")
    ap.add_argument("--kb", default=KB_DEFAULT)
    ap.add_argument("--api", default=API_DEFAULT)
    ap.add_argument("--token", default=TOKEN_DEFAULT)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--gap", type=float, default=0.5, help="seconds to sleep between batches")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--state", default=STATE_DEFAULT)
    ap.add_argument("--wait", action="store_true", help="poll until the queued documents finish")
    ap.add_argument("--poll", type=int, default=60, help="seconds between status polls with --wait")
    ap.add_argument(
        "--retry-failed",
        action="store_true",
        help="instead of queueing new work, re-queue target documents that ended up with zero chunks",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.list = os.path.abspath(args.list)
    state = load_state(args.state)
    queued = retry_failed(args, state) if args.retry_failed else queue_all(args, state)
    log(f"queued {len(queued)} documents this run")
    if args.wait and not args.dry_run and state["triggered"]:
        wait_for(args, state["triggered"])


if __name__ == "__main__":
    sys.exit(main())
