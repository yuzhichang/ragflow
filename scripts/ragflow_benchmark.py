#!/usr/bin/env python3
"""Unified, dataset-agnostic QA benchmark runner for RAGFlow.

Replaces the day-to-day use of browseComp_benchmark.py and frame_benchmark.py.
Both of those scripts stay untouched; this one is where new behaviour lands.

Unified:
  * One question loader - .jsonl, JSON list, or JSON {id: {...}} mapping.
  * Serial answer and judge phases with resume.
  * One judge scorer: the 0/2/4 rule book wins, tolerant fallbacks after.
    (browseComp rejected any score above 1.0; frame divided 1..100 by 100.)
  * One result artefact: leaderboard.json (the BrowseComp-Plus submission JSON
    plus per-question judgement/usage extensions) carries everything a run
    produces; answers.jsonl stays a pure record of what the agent returned.

Dropped (only ever existed in frame_benchmark.py):
  * the retry / repair subcommands,
  * --bad-cases files, error-category filters and the _annotations tables,
  * per-dataset helpers (_load_bad_case_questions, _filter_by_category,
    _deduplicate_jsonl, _replace_jsonl_rows).
Dropped (superseded by leaderboard.json):
  * judged_answers.jsonl and report.json - the judge verdicts now live in the
    leaderboard's per_query_judgements extension, and the diagnostic totals in
    its _diagnostics block.

Config keys (optional unless marked required):
  dataset          required - {name, description, questions_path, corpus_path?}
  chat_id          required - chat that answers the questions
  judge_chat_id    required - chat that judges the answers. Deliberately kept
                   separate from chat_id so judging is not polluted by the
                   answer chat's corpus retrieval.
  ragflow_api_key / backend
  ragflow_chat     {agent_mode, stream, fresh_session_per_question, llm_id,
                   quote, refine_multiturn, temperature, top_p, max_tokens,
                   reasoning}
  judge_stream     optional - stream the judge chat
  judge_stream     is nested under the judge chat only; the judge PROMPT is
                   deliberately NOT configurable (see GRADER_TEMPLATE).
  leaderboard      {llm, retriever, link, evaluation_date, search_tools,
                   calibration_bin_size} - identity fields for the submission
                   JSON plus which tools count as a search call.
  question_ids     {include: [..], random: N, exclude: [..]}
                   include and random are alternative strategies - if both are
                   set include wins, if neither is effective every question is
                   selected. exclude is dropped from the result of either
                   strategy (the random sample is drawn after exclusion).
                   random is seeded with the current epoch second, so every run
                   draws a fresh sample; use include for reproducibility.
  concurrency      optional - how many questions to answer (and judge) in
                   parallel; the CLI --concurrency flag overrides it. 1 (the
                   default) keeps strictly serial execution with the
                   serial-phase ordering guarantees. Above 1: shared sessions
                   are disabled (a session's history would interleave), answers
                   land in answers.jsonl out of order (resume is id-based, so
                   nothing breaks), and the quota circuit breaker counts TOTAL
                   consecutive Token Plan failures across workers instead of
                   strictly consecutive rows.
  annotations      optional - curated run history (passed ids, notes on hard
                   questions); informational only, not written anywhere.
  output           {output_dir, answers_jsonl, leaderboard_json}

Two artefacts are written at the end of a run:
  * answers.jsonl    - one row per question: the question, the agent's raw
                       return (answer, session, tool/token/gate accounting)
                       and the dataset's reference annotations. No scored
                       fields - everything derived from the reference answer
                       lives in leaderboard.json.
  * leaderboard.json - the BrowseComp-Plus submission JSON (binary accuracy,
                       retrieval recall, search calls, calibration error,
                       per_query_metrics) plus per_query_usage /
                       per_query_judgements extensions carrying each
                       question's tokens, time and full judge verdict.

Deliberately NOT configurable (constants at the top of this file):
  timeout_seconds / retry -> DEFAULT_TIMEOUT_SECONDS / DEFAULT_MAX_RETRIES
  judge prompt            -> GRADER_TEMPLATE, identical for every dataset (see
                             its comment for provenance)
  execution               -> concurrency 1 (serial) unless the config or the
                             CLI --concurrency flag says otherwise
  random seed             -> current epoch second
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

DEFAULT_CONFIG_PATH = Path("qa_benchmark_conf.json")

DEFAULT_OUTPUT_DIR = "outputs/qa_benchmark_<timestamp>"
DEFAULT_ANSWERS_NAME = "answers.jsonl"
DEFAULT_LEADERBOARD_NAME = "leaderboard.json"

# Transport defaults. Intentionally not per-dataset config: every dataset talks
# to the same backend, so retries and timeouts are tuned in one place.
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_MAX_RETRIES = 0
DEFAULT_BACKOFF_SECONDS = 2.0

# THE judge prompt, used for every dataset. Provenance:
#   * BrowseComp-Plus scoring script judge prompt - copied verbatim from
#     GRADER_TEMPLATE in
#     https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_evaluation/evaluate_run.py
#     byte-for-byte identical (including the 0|\%| LaTeX percent escapes the
#     official script sends as-is), because a submission is only comparable
#     with the published numbers when the same judge prompt produced the
#     verdicts. Note this is the evaluate_run.py variant, NOT the older
#     search_agent/prompts.py one: the reasoning criteria judge SEMANTIC
#     equivalence and explicitly allow a more precise or verbose extracted
#     answer. The verdict is BINARY (correct: yes/no) and the judge reports
#     the confidence it found in the answer (0-100, 100 when the answer
#     states none) - the leaderboard's accuracy and calibration error are
#     computed from exactly these two fields.
#   * FRAMES judge prompt - FRAMES ships no official scoring script: google/
#     frames-benchmark discussion #18
#       https://huggingface.co/datasets/google/frames-benchmark/discussions/18
#     asks for one; the original paper's judge prompt lives in its appendix
#     (judge model Gemini-Pro-1.5-0514, since retired), and the thread's
#     consensus points at community re-implementations (e.g. optillm's
#     scripts/eval_frames_benchmark.py) that re-use the paper appendix prompts
#     with a judge model of one's own choosing. This benchmark therefore
#     judges FRAMES questions with the same BrowseComp-Plus grader prompt
#     above (same
#     https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_evaluation/evaluate_run.py
#     provenance), so frames and browsecomp results stay on one scale.
GRADER_TEMPLATE = r"""Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

# Tools that count as a SEARCH for the leaderboard's "Search Calls". The
# locate tools are the retriever: each call asks the corpus for documents.
# list_chunks is the deep read (BrowseComp-Plus calls it get_document) - it
# opens a document the agent already located, so it is reported separately in
# avg_tool_stats but does not inflate the search count.
DEFAULT_SEARCH_TOOLS = ("search_chunks", "grep_chunks", "search_bm25_chunks")

# Official script: calibration error needs at least this many scored
# confidences, otherwise it is reported as 0.0 with a warning.
CALIBRATION_MIN_SAMPLES = 100
DEFAULT_CALIBRATION_BIN_SIZE = 100

# Provider plan/quota circuit breaker. A WALL is durable - once the plan is
# exhausted every remaining question would fail the same way, so the run must
# stop with its completed rows intact. But the gateway reports short TPM/RPM
# bursts with the SAME wording ("已达到 Token Plan 用量上限"), and with N
# questions in flight one burst produces N failures within seconds: aborting an
# 830-question run on that is wrong (it ended runs minutes in while the plan
# still had headroom). So the breaker only aborts when the failures are SPREAD
# OUT; failures clustered inside QUOTA_BURST_WINDOW_SEC are a burst - pause and
# probe again. QUOTA_ABORT_TOTAL bounds the pauses so a genuine wall still ends
# the run instead of looping.
QUOTA_ABORT_AFTER = 3  # consecutive-ish failures before judging
QUOTA_BURST_WINDOW_SEC = 120  # failures closer together than this = burst
QUOTA_BURST_PAUSE_SEC = 90  # sleep before probing again after a burst
QUOTA_ABORT_TOTAL = 12  # hard stop regardless of clustering

# The judge phase runs its own, SMALLER concurrency: judging is a single
# short chat call, so the throughput win of a high concurrency is nil while a
# burst of rate-limit failures costs real verdicts (judge calls that fail with
# a provider error leave the row unjudged, and an unjudged row counts as
# incorrect). Two workers are enough; the answer phase keeps the configured
# concurrency.
JUDGE_CONCURRENCY_CAP = 2

# A judge call that comes back as a provider error is retried before the row is
# recorded as unjudged. The backend decorates provider failures into the ANSWER
# TEXT ("**ERROR**: minimax API error: ..."), so they never surface as a Python
# exception here - retrying has to inspect the text. Without this, one
# rate-limit burst silently produced rows with no verdict at all.
JUDGE_MAX_RETRIES = 3
JUDGE_RETRY_BACKOFF_SEC = (2.0, 5.0, 10.0)
JUDGE_TRANSIENT_MARKERS = (
    "**ERROR**",  # provider failure decorated into the answer text
    "速率限制",
    "用量上限",
    "rate limit",
    "429",
    "529",
    "status 5",
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
)

# Candidate keys per logical field, first hit wins. Keeps dataset-specific
# naming out of the code: add a key here instead of a new loader.
ID_KEYS = ("id", "question_id", "qid")
QUESTION_KEYS = ("question", "query", "prompt")
GOLD_KEYS = ("gold_answer", "answer", "reference_answer", "gold")
REASONING_TYPE_KEYS = ("reasoning_types", "reasoning_type", "types")
EXPECTED_DOC_KEYS = ("expected_doc_ids", "expected_sources")
# The leaderboard scores RECALL against the qrel "evidence" set - every document
# labelled as needed to answer the query - which is a superset of the gold set
# (the documents that also contain the final answer). BrowseComp-Plus ships both
# as expected_doc_ids / evidence_doc_ids, so they are read separately: the
# diagnostic report keeps using expected_doc_ids, leaderboard.json uses
# evidence_doc_ids.
EVIDENCE_DOC_KEYS = ("evidence_doc_ids", "expected_evidence_doc_ids", "qrel_evidence")


class BenchmarkError(RuntimeError):
    pass


class QuotaBreaker:
    """Decides whether Token Plan failures mean "burst" or "wall".

    Every phase outcome is fed to note(); a non-quota outcome clears the
    history. note() returns "" (keep going), "burst" (pause and probe again)
    or "wall" (abort the run). See the QUOTA_* constants for the rationale:
    with concurrency > 1 a single rate-limit burst fails several in-flight
    questions at once, which must not be mistaken for an exhausted plan.
    """

    def __init__(self) -> None:
        self.failed_at: list[float] = []
        self.total = 0

    def note(self, error: str | None) -> str:
        if "Token Plan" not in (error or ""):
            self.failed_at = []
            self.total = 0
            return ""
        now = time.time()
        self.total += 1
        # Keep EVERY failure since the last non-quota row (no age-based
        # dropping: filtering by the window would make the span test below
        # unreachable, because three surviving entries are always closer
        # together than the window).
        self.failed_at.append(now)
        if self.total >= QUOTA_ABORT_TOTAL:
            return "wall"
        if len(self.failed_at) < QUOTA_ABORT_AFTER:
            return ""
        if self.failed_at[-1] - self.failed_at[0] >= QUOTA_BURST_WINDOW_SEC:
            return "wall"
        # Clustered failures: a burst, not a wall. Drop the history so the
        # next probe starts clean.
        self.failed_at = []
        return "burst"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class JsonHttpClient:
    """Minimal RAGFlow HTTP client. One instance per worker thread: the
    underlying requests.Session is not thread safe, hence clone()."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: int,
        max_retries: int,
        backoff_seconds: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self.session = requests.Session()

    def clone(self) -> JsonHttpClient:
        return JsonHttpClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            backoff_seconds=self.backoff_seconds,
        )

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("POST", path, json=body)

    def post_eventstream(self, path: str, body: dict[str, Any]) -> Any:
        stream_body = dict(body)
        stream_body["stream"] = True
        return self._request_eventstream("POST", path, json=stream_body)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, headers=headers, timeout=self.timeout_seconds, **kwargs)
                payload = _decode_response(response)
                if response.status_code >= 400:
                    raise BenchmarkError(f"HTTP {response.status_code} from {url}: {_compact(payload)}")
                if isinstance(payload, dict) and payload.get("code", 0) not in (0, None):
                    raise BenchmarkError(f"RAGFlow code {payload.get('code')} from {url}: {payload.get('message') or _compact(payload)}")
                return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
            except (requests.RequestException, BenchmarkError) as exc:
                last_error = exc
                if attempt >= self.max_retries or not _should_retry(exc):
                    break
                time.sleep(self.backoff_seconds * (2**attempt))

        raise BenchmarkError(str(last_error))

    def _request_eventstream(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "text/event-stream", "Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, headers=headers, timeout=self.timeout_seconds, stream=True, **kwargs)
                if response.status_code >= 400:
                    payload = _decode_response(response)
                    raise BenchmarkError(f"HTTP {response.status_code} from {url}: {_compact(payload)}")
                if "text/event-stream" not in response.headers.get("content-type", ""):
                    payload = _decode_response(response)
                    if isinstance(payload, dict) and payload.get("code", 0) not in (0, None):
                        raise BenchmarkError(f"RAGFlow code {payload.get('code')} from {url}: {payload.get('message') or _compact(payload)}")
                    return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
                return _collect_eventstream_answer(response.iter_lines(decode_unicode=True))
            except (requests.RequestException, BenchmarkError) as exc:
                last_error = exc
                if attempt >= self.max_retries or not _should_retry(exc):
                    break
                time.sleep(self.backoff_seconds * (2**attempt))

        raise BenchmarkError(str(last_error))


def create_session(client: JsonHttpClient, chat_id: str) -> str:
    payload = client.post(
        f"/api/v1/chats/{quote(chat_id, safe='')}/sessions",
        {"name": f"qa-benchmark-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}"},
    )
    session_id = _extract_session_id(payload)
    if not session_id:
        raise BenchmarkError(f"Could not create chat session: {_compact(payload)}")
    return session_id


def post_chat_completion(client: JsonHttpClient, body: dict[str, Any]) -> Any:
    if body.get("stream"):
        return client.post_eventstream("/api/v1/chat/completions", body)
    return client.post("/api/v1/chat/completions", body)


def ask_ragflow(
    client: JsonHttpClient,
    cfg: dict[str, Any],
    question: str,
    *,
    session_id: str | None,
) -> Any:
    chat_cfg = cfg.get("ragflow_chat", {})
    body: dict[str, Any] = {
        "chat_id": cfg["chat_id"],
        "question": question,
        "stream": bool(chat_cfg.get("stream", False)),
    }
    if session_id:
        body["session_id"] = session_id
    if chat_cfg.get("llm_id"):
        body["llm_id"] = chat_cfg["llm_id"]
    for key in ("quote", "refine_multiturn", "temperature", "top_p", "max_tokens"):
        if key in chat_cfg:
            body[key] = chat_cfg[key]

    # agent_mode (e.g. "smart-reasoning") drives the agentic ReAct loop; in that
    # mode the chat API reads `reasoning` as part of the agentic request, so it
    # is pinned to 2 unless the config overrides it.
    body["reasoning"] = int(chat_cfg.get("reasoning", 2))
    if chat_cfg.get("agent_mode"):
        body["agent_mode"] = chat_cfg["agent_mode"]

    return post_chat_completion(client, body)


def ask_judge(client: JsonHttpClient, cfg: dict[str, Any], row: dict[str, Any], llm_id: str | None = None) -> Any:
    body: dict[str, Any] = {
        "chat_id": cfg["judge_chat_id"],
        "question": render_judge_prompt(row),
        "stream": bool(cfg.get("judge_stream", False)),
    }
    # llm_id override: the judge chat is a PLAIN chat (no agent_mode), so it has
    # NO failover chain - it is pinned to the single instance it is bound to,
    # and when that instance's plan is exhausted every judgement fails. Asking
    # for another of the tenant's instances explicitly is what lets judging
    # survive a single-instance quota wall.
    if llm_id:
        body["llm_id"] = llm_id
    return post_chat_completion(client, body)


def judge_llm_ids(cfg: dict[str, Any]) -> list[str | None]:
    """The llm_ids to try when judging, in order.

    `judge_llm_ids` (list) or `judge_llm_id` (single) in the config; an empty
    configuration yields [None] — the chat's own binding, i.e. today's default.
    """
    configured = cfg.get("judge_llm_ids")
    if isinstance(configured, list) and [x for x in configured if str(x).strip()]:
        return [str(x).strip() for x in configured if str(x).strip()]
    single = cfg.get("judge_llm_id")
    if single:
        return [str(single).strip()]
    return [None]


def _is_transient_judge_answer(answer: str) -> bool:
    """True when a judge reply is a provider failure rather than a verdict.

    The backend decorates provider errors into the answer text
    ("**ERROR**: minimax API error: ... 速率限制"), so they arrive here as a
    perfectly valid string. A retry can only help when the failure is the
    transient kind, hence the marker test on top of the ERROR prefix.
    """
    if not answer.lstrip().startswith("**ERROR**"):
        return False
    return any(marker in answer for marker in JUDGE_TRANSIENT_MARKERS[1:])


def ask_judge_with_retry(client: JsonHttpClient, cfg: dict[str, Any], row: dict[str, Any]) -> str:
    """ask_judge with backoff retries, then with the next configured instance.

    Two failure classes, handled in order of cost:
      * a transient burst - retried on the SAME instance with backoff;
      * an exhausted instance - retried on the NEXT llm_id (plain chats have no
        failover chain of their own, so the rotation has to happen here).
    One rate-limit burst used to leave rows with no verdict at all (and an
    unjudged row counts as incorrect).
    """
    ids = judge_llm_ids(cfg)
    answer = ""
    for llm_id in ids:
        answer = extract_answer_text(ask_judge(client.clone(), cfg, row, llm_id))
        for attempt in range(JUDGE_MAX_RETRIES):
            if not _is_transient_judge_answer(answer):
                return answer
            time.sleep(JUDGE_RETRY_BACKOFF_SEC[min(attempt, len(JUDGE_RETRY_BACKOFF_SEC) - 1)])
            answer = extract_answer_text(ask_judge(client.clone(), cfg, row, llm_id))
        # Retries on this instance are exhausted; fall through to the next id
        # (when the remaining failure is a plan wall, not a burst).
    return answer


def render_judge_prompt(row: dict[str, Any]) -> str:
    """Render THE judge prompt. The template is not configurable: it is the
    BrowseComp-Plus scoring script's reference prompt, so letting a config
    rewrite it would silently break comparability with published numbers."""
    return (
        GRADER_TEMPLATE.replace("{question}", str(row.get("question", "")))
        .replace("{correct_answer}", str(row.get("gold_answer", "")))
        .replace(
            "{response}",
            re.sub(r"<think>.*?</think>", "", str(row.get("ragflow_answer", "")), flags=re.DOTALL),
        )
    )


def run_answer_phase(
    *,
    client: JsonHttpClient,
    cfg: dict[str, Any],
    questions: list[dict[str, Any]],
    answers_path: Path,
    concurrency: int = 1,
) -> bool:
    """Returns True when the run was aborted by the quota circuit breaker."""
    completed = {key for key, row in _read_jsonl_by_id(answers_path).items() if not (row.get("ragflow_error") or "").strip()}
    chat_cfg = cfg.get("ragflow_chat", {})
    shared_session_id = None
    concurrency = max(1, int(concurrency or 1))

    # A shared session carries conversation history: with parallel workers the
    # turns would interleave into one another, so above concurrency 1 every
    # question gets its own session no matter what the config says.
    shared_ok = not chat_cfg.get("fresh_session_per_question", True) and concurrency == 1
    if shared_ok:
        shared_session_id = create_session(client.clone(), cfg["chat_id"])
    elif concurrency > 1 and not chat_cfg.get("fresh_session_per_question", True):
        print("[answers] concurrency > 1 forces fresh_session_per_question (a shared session's history would interleave)")

    # seq/total count the PENDING questions, not the batch: a resumed run must
    # not print "52/36" (batch position over pending total).
    pending = [question for question in questions if _run_id(question["question_id"]) not in completed]
    jobs = [(seq, question, _run_id(question["question_id"])) for seq, question in enumerate(pending, start=1)]
    skipped = len(questions) - len(jobs)
    if skipped:
        print(f"[answers] skip {skipped} existing row(s)")
    if not jobs:
        return

    total = len(jobs)
    started = time.time()
    breaker = QuotaBreaker()
    aborted = False
    write_lock = threading.Lock()

    def _answer_one(seq: int, question: dict[str, Any], run_id: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "run_id": run_id,
            "question_id": question["question_id"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "reasoning_types": question["reasoning_types"],
            "ragflow_answer": "",
            "ragflow_error": "",
            "ragflow_session_id": None,
        }

        expected_doc_ids = _unique_doc_ids(question.get("expected_doc_ids") or [])
        if expected_doc_ids:
            row["expected_doc_ids"] = sorted(expected_doc_ids)
        evidence_doc_ids = _unique_doc_ids(question.get("evidence_doc_ids") or [])
        if evidence_doc_ids:
            row["evidence_doc_ids"] = sorted(evidence_doc_ids)

        try:
            session_id = shared_session_id if shared_ok else None
            payload = ask_ragflow(client.clone(), cfg, question["question"], session_id=session_id)
            answer = extract_answer_text(payload)
            row["ragflow_answer"] = answer
            row["ragflow_session_id"] = _extract_session_id(payload)
            row.update(extract_run_stats(payload))
            if not answer:
                row["ragflow_error"] = "empty_ragflow_answer"
            elif answer.lstrip().startswith("**ERROR**"):
                row["ragflow_error"] = answer.strip()
        except Exception as exc:  # noqa: BLE001 - one bad question must not kill the run
            row["ragflow_error"] = str(exc)

        status = row["ragflow_error"] or f"{len(row['ragflow_answer'])} chars"
        print(f"[answers] {seq}/{total} {run_id}: {status}", flush=True)
        return row

    with answers_path.open("a", encoding="utf-8") as handle:

        def _run_one(seq: int, question: dict[str, Any], run_id: str) -> bool:
            """Answer one question, append its row, run the quota breaker.
            Returns True only for a genuine wall (a burst pauses and probes
            again instead of ending the run). Client-side timing: the backend
            logs each question's token/latency summary, but those logs rotate
            away (the launch log is rewritten on every restart), so wall-clock
            is kept here to make a run analysable from its own artefacts
            alone."""
            started_one = time.time()
            row = _answer_one(seq, question, run_id)
            row["answer_elapsed_seconds"] = round(time.time() - started_one, 3)
            with write_lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                verdict = breaker.note(row.get("ragflow_error"))
            if verdict == "burst":
                print(
                    f"[answers] Token Plan failures clustered inside {QUOTA_BURST_WINDOW_SEC}s - rate-limit burst, not an exhausted plan: pausing {QUOTA_BURST_PAUSE_SEC}s and probing again",
                    flush=True,
                )
                time.sleep(QUOTA_BURST_PAUSE_SEC)
            elif verdict == "wall":
                print(
                    f"[answers] provider plan quota exhausted ({breaker.total} Token Plan failures, "
                    f"spread over >= {QUOTA_BURST_WINDOW_SEC}s) - aborting; remaining question(s) left "
                    f"untouched. Re-run the same command once the quota is replenished."
                )
            return verdict == "wall"

        if concurrency <= 1:
            for seq, question, run_id in jobs:
                if _run_one(seq, question, run_id):
                    aborted = True
                    break
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_run_one, seq, question, run_id) for seq, question, run_id in jobs]
                pending = set(futures)
                while pending and not aborted:
                    done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in done:
                        if future.cancelled():
                            continue
                        if future.result():
                            aborted = True
                            # Queued-but-unstarted questions are cancelled;
                            # already-running ones are allowed to finish.
                            for other in pending:
                                other.cancel()
                            break

    print(f"[answers] {total} run(s) in {time.time() - started:.1f}s -> {answers_path}")
    return aborted


def run_judge_phase(
    *,
    client: JsonHttpClient,
    cfg: dict[str, Any],
    answers_path: Path,
    leaderboard_path: Path,
    concurrency: int = 1,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Judge every unanswered answer row and persist the verdicts.

    Judgements live in leaderboard.json's per_query_judgements extension,
    keyed by run key - that file is both the submission output and the resume
    state, so there is no separate judged artefact. Returns (judgements,
    aborted); `aborted` is True when the quota circuit breaker fired.
    """
    if not answers_path.exists():
        raise FileNotFoundError(f"Missing answers file: {answers_path}")

    rows = _dedupe_last(_read_jsonl(answers_path))
    # A judgement whose verdict text is a backend error (e.g. a provider quota
    # wall) is NOT "done": it must be re-judged on the next run, which
    # overwrites the failed record.
    completed = {key for key, record in _read_judgements(leaderboard_path).items() if not _is_backend_error_verdict(record)}

    todo = [row for row in rows if _row_run_key(row) not in completed]
    pending = list(enumerate(todo, start=1))
    skipped = len(rows) - len(pending)
    if skipped:
        print(f"[judge] skip {skipped} existing row(s)")

    judgements = _read_judgements(leaderboard_path)
    if not pending:
        return judgements, False

    def _judge_one(row: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "correct": None,
            "confidence": None,
            "extracted_final_answer": None,
            "verdict": None,
            "judge_error": "",
        }
        if row.get("ragflow_error"):
            record["judge_error"] = "excluded_due_to_ragflow_error"
        else:
            try:
                judge_answer = ask_judge_with_retry(client, cfg, row)
                verdict = parse_grader_verdict(judge_answer)
                record["verdict"] = verdict
                record["correct"] = verdict["correct"]
                record["confidence"] = verdict["confidence"]
                record["extracted_final_answer"] = verdict["extracted_final_answer"]
            except Exception as exc:  # noqa: BLE001
                record["judge_error"] = str(exc)
        return record

    concurrency = max(1, int(concurrency or 1))
    # Judging is one short chat call per row: a high concurrency buys no
    # throughput but multiplies the damage of a rate-limit burst, so the judge
    # phase is capped independently of the answer phase.
    if concurrency > JUDGE_CONCURRENCY_CAP:
        print(f"[judge] capping concurrency to {JUDGE_CONCURRENCY_CAP} (configured: {concurrency})")
        concurrency = JUDGE_CONCURRENCY_CAP
    breaker = QuotaBreaker()
    aborted = False
    write_lock = threading.Lock()

    def _finish_judge(seq: int, row: dict[str, Any]) -> bool:
        """Judge one row, persist its verdict, run the quota breaker. Returns
        True only for a genuine wall (a burst pauses and probes again)."""
        started_one = time.time()
        record = _judge_one(row)
        elapsed = round(time.time() - started_one, 3)
        key = _row_run_key(row)
        record["query_id"] = str(row.get("question_id") or key or "")
        record["run_key"] = key
        record["judge_elapsed_seconds"] = elapsed

        with write_lock:
            judgements[key] = record
            # One write per judged row: an aborted run keeps its completed
            # verdicts on disk, which is exactly what the resume logic reads.
            _write_json(
                leaderboard_path,
                _leaderboard_with_judgements(_dedupe_last(_read_jsonl(answers_path)), judgements, cfg),
            )
            # Same breaker as the answer phase: judge and answer share the
            # provider plan, so a quota wall here would burn the remaining
            # rows. Clustered failures are a burst, not a wall.
            verdict = breaker.note(record.get("judge_error"))

        status = record["judge_error"] or f"correct={record['correct']}"
        print(f"[judge] {seq}/{len(pending)} {key}: {status} ({elapsed}s)", flush=True)
        if verdict == "burst":
            print(
                f"[judge] Token Plan failures clustered inside {QUOTA_BURST_WINDOW_SEC}s - rate-limit burst, not an exhausted plan: pausing {QUOTA_BURST_PAUSE_SEC}s and probing again",
                flush=True,
            )
            time.sleep(QUOTA_BURST_PAUSE_SEC)
        elif verdict == "wall":
            print(
                f"[judge] provider plan quota exhausted ({breaker.total} Token Plan failures, "
                f"spread over >= {QUOTA_BURST_WINDOW_SEC}s) - aborting; remaining row(s) left "
                f"unjudged. Re-run the same command once the quota is replenished."
            )
        return verdict == "wall"

    if concurrency <= 1:
        for seq, row in pending:
            if _finish_judge(seq, row):
                aborted = True
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_finish_judge, seq, row) for seq, row in pending]
            outstanding = set(futures)
            while outstanding and not aborted:
                done, outstanding = concurrent.futures.wait(outstanding, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    if future.cancelled():
                        continue
                    if future.result():
                        aborted = True
                        for other in outstanding:
                            other.cancel()
                        break

    print(f"[judge] {len(judgements)} judgement(s) -> {leaderboard_path}")
    return judgements, aborted


# --------------------------------------------------------------------------
# Judgement persistence (leaderboard.json per_query_judgements)
# --------------------------------------------------------------------------
def _read_judgements(leaderboard_path: Path) -> dict[str, dict[str, Any]]:
    """The per_query_judgements already stored in leaderboard.json, keyed by
    run key. Missing file or missing block -> empty: nothing was judged yet."""
    if not leaderboard_path.exists():
        return {}
    try:
        leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[judge] could not read {leaderboard_path}: {exc}")
        return {}
    records = leaderboard.get("per_query_judgements") if isinstance(leaderboard, dict) else None
    if not isinstance(records, list):
        return {}
    return {str(record.get("run_key")): record for record in records if isinstance(record, dict) and record.get("run_key")}


def _merge_judgements(
    rows: list[dict[str, Any]],
    judgements: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach each row's judge verdict to the answer row (in memory) so the
    leaderboard builder can score it. The persisted verdicts stay in
    leaderboard.json's per_query_judgements block."""
    merged: list[dict[str, Any]] = []
    for row in rows:
        record = judgements.get(_row_run_key(row))
        if record:
            row = dict(row)
            row["judge_correct"] = record.get("correct")
            row["judge_confidence"] = record.get("confidence")
            row["judge_error"] = record.get("judge_error") or ""
            row["judge_extracted_answer"] = record.get("extracted_final_answer")
            row["judge_parsed"] = record.get("verdict")
        merged.append(row)
    return merged


def _leaderboard_with_judgements(
    rows: list[dict[str, Any]],
    judgements: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Merge the judge verdicts into the answer rows and build the leaderboard
    from the merge. Used both incrementally (one write per judged row, so an
    aborted run keeps its completed verdicts for resume) and for the final
    export."""
    return build_leaderboard(_merge_judgements(rows, judgements), cfg)


# --------------------------------------------------------------------------
# Leaderboard export (BrowseComp-Plus submission format)
# --------------------------------------------------------------------------
def build_leaderboard(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    evidence_by_id: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build the JSON the BrowseComp-Plus leaderboard asks for.

    Every formula mirrors scripts_evaluation/evaluate_with_openai.py in
    texttron/BrowseComp-Plus, including its denominators:
      * Accuracy divides by ALL questions of the run. A question the agent
        failed on (error, empty answer, unparseable judgement) counts as
        incorrect, which is why this number is lower than report.json's
        `overall_accuracy` (that one excludes the failed rows).
      * Recall averages only over questions that have evidence documents.
      * Calibration error needs at least 100 scored confidences, otherwise the
        reference script reports 0.0.
    """
    lb_cfg = cfg.get("leaderboard") or {}
    evidence_by_id = evidence_by_id or {}
    search_tools = _search_tools(lb_cfg)
    beta = _calibration_bin_size(lb_cfg)

    total = len(rows)
    per_query_metrics: list[dict[str, Any]] = []
    per_query_usage: list[dict[str, Any]] = []
    confidences: list[float] = []
    correctness: list[bool] = []
    recalls: list[float] = []
    tool_totals: dict[str, float] = defaultdict(float)
    tool_error_totals: dict[str, float] = defaultdict(float)
    tool_error_samples: dict[str, str] = {}
    stats_missing = 0
    judged = 0
    per_query_judgements: list[dict[str, Any]] = []

    for row in rows:
        query_id = str(row.get("question_id") or _row_run_key(row) or "")
        correct_raw = _leaderboard_correct(row)
        # Anything the judge did not affirm is a miss: the reference script
        # counts a result whose judge_result.correct is absent (error, empty
        # answer, unparseable verdict) as not correct.
        correct = bool(correct_raw)
        if correct_raw is not None:
            judged += 1
            confidence = _confidence_percent(row.get("judge_confidence"))
            if confidence is not None:
                confidences.append(confidence)
                correctness.append(correct)

        evidence = _unique_doc_ids(evidence_by_id.get(query_id) or row.get("evidence_doc_ids") or [])
        retrieved = row.get("retrieved_docids")
        recall = None
        if evidence:
            hits = set(_unique_doc_ids(retrieved or [])) & set(evidence)
            recall = len(hits) / len(evidence)
            recalls.append(recall)

        per_query_metrics.append(
            {
                "query_id": query_id,
                "correct": correct,
                "recall": _round_percent(recall),
            }
        )

        # Extension, not part of the submission schema: the full judge verdict
        # per question (replaces the former judged_answers.jsonl). Rows the
        # judge never reached are omitted - _diagnostics.judged counts them.
        if any(row.get(k) is not None for k in ("judge_correct", "judge_error")):
            per_query_judgements.append(
                {
                    "query_id": query_id,
                    "run_key": _row_run_key(row),
                    "correct": correct_raw,
                    "confidence": _confidence_percent(row.get("judge_confidence")),
                    "extracted_final_answer": row.get("judge_extracted_answer"),
                    "verdict": row.get("judge_parsed"),
                    "judge_error": row.get("judge_error") or "",
                }
            )

        counts = row.get("tool_call_counts")
        if isinstance(counts, dict):
            for name, count in counts.items():
                numeric = _as_float(count)
                if numeric is not None:
                    tool_totals[str(name)] += numeric
        elif not row.get("ragflow_error"):
            # A row the backend never reported stats for is not "zero calls":
            # streaming runs and pre-instrumentation backends never see them.
            stats_missing += 1

        # Tool-failure accounting: a row whose tools keep failing is the
        # signature of a retrieval outage, not a hard question - make it
        # visible in the diagnostics instead of buried in answers.jsonl.
        row_errors = row.get("tool_call_errors")
        if isinstance(row_errors, dict):
            for name, count in row_errors.items():
                numeric = _as_float(count)
                if numeric is not None:
                    tool_error_totals[str(name)] += numeric
        row_samples = row.get("tool_error_samples")
        if isinstance(row_samples, dict):
            for name, text in row_samples.items():
                tool_error_samples.setdefault(str(name), str(text))

        per_query_usage.append(_usage_row(query_id, row, search_tools))

    correct_count = sum(1 for entry in per_query_metrics if entry["correct"])
    accuracy_percent = round(correct_count / total * 100.0, 2) if total else 0.0
    # Rows that actually received a verdict (the headline number counts the
    # rest as incorrect, mirroring the reference script).
    scored_rows = [row for row in rows if row.get("judge_correct") is not None]
    judged_total = len(scored_rows)
    judged_ok = sum(1 for row in scored_rows if row.get("judge_correct"))
    recall_percent = round(sum(recalls) / len(recalls) * 100.0, 2) if recalls else None

    avg_tool_stats = {name: count / total for name, count in tool_totals.items()} if total else {}
    search_calls = round(sum(avg_tool_stats.get(name, 0.0) for name in search_tools), 2)

    calibration_error_percent = 0.0
    calibration_note = f"not computed: only {len(confidences)} scored confidence(s), need {CALIBRATION_MIN_SAMPLES}"
    if len(confidences) >= CALIBRATION_MIN_SAMPLES:
        calibration_error_percent = round(calibration_error(confidences, correctness, beta=beta), 2)
        calibration_note = f"{len(confidences)} scored confidence(s), bin size {beta}"
    if len(confidences) < CALIBRATION_MIN_SAMPLES:
        print(f"[leaderboard] calibration error {calibration_note}")

    return {
        "LLM": str(lb_cfg.get("llm") or "change me when submitting"),
        "Retriever": str(lb_cfg.get("retriever") or "change me when submitting"),
        "Accuracy (%)": accuracy_percent,
        "Recall (%)": recall_percent,
        "Search Calls": search_calls,
        "Calibration Error (%)": calibration_error_percent,
        "Link": str(lb_cfg.get("link") or "change me when submitting"),
        "Evaluation Date": str(lb_cfg.get("evaluation_date") or datetime.now().astimezone().date().isoformat()),
        # The reference script's own summary carries per-tool averages; the
        # leaderboard reads "Search Calls" as the retrieval-tool subset of it.
        "avg_tool_stats": dict(sorted(avg_tool_stats.items())),
        # Extension: per-tool failure totals + one root-cause sample per
        # tool, so an outage shows up in the leaderboard diagnostics.
        "tool_error_stats": dict(sorted(tool_error_totals.items())) if tool_error_totals else {},
        "tool_error_samples": dict(sorted(tool_error_samples.items())) if tool_error_samples else {},
        "per_query_metrics": per_query_metrics,
        # Extension, not part of the submission schema: per-question token and
        # wall-clock cost. Kept in its own key so per_query_metrics stays
        # byte-compatible with what the leaderboard parses.
        "per_query_usage": per_query_usage,
        "per_query_judgements": per_query_judgements,
        "_diagnostics": {
            "questions": total,
            "judged": judged,
            "unjudged_counted_incorrect": total - judged,
            # Accuracy over the rows that actually received a verdict. The
            # headline Accuracy (%) follows the reference script and counts an
            # unjudged row as incorrect, so a run hit by provider rate limits
            # reads low for a reason that has nothing to do with the agent -
            # this number separates the two.
            "judged_only_accuracy_percent": round(judged_ok / judged_total * 100.0, 2) if judged_total else None,
            "judged_only_denominator": judged_total,
            "recall_scored": len(recalls),
            "search_tools": list(search_tools),
            "calibration": calibration_note,
            "run_stats_missing": stats_missing,
            "note": " ".join(
                [
                    "Accuracy counts every question of the run; failed or unjudged rows are",
                    "incorrect, so it is lower than report.json's overall_accuracy.",
                ]
                + (
                    [
                        f"Recall and Search Calls are understated: {stats_missing} row(s) were",
                        "answered by a backend that did not report tool_call_counts /",
                        "retrieved_docids (a streaming run, or one built before the fields",
                        "existed), and rows without accounting are scored as 'retrieved nothing'.",
                    ]
                    if stats_missing
                    else []
                )
            ),
        },
    }


def _leaderboard_correct(row: dict[str, Any]) -> bool | None:
    """The binary verdict of a row, or None when the row was never judged.

    `judge_correct` is what the judge phase writes; rows the judge never
    reached (backend error, quota abort) return None and count as incorrect.
    """
    if row.get("judge_correct") is not None:
        return bool(row["judge_correct"])
    if row.get("judge_error") or row.get("ragflow_error"):
        return None
    return None


def _confidence_percent(value: Any) -> float | None:
    """Normalize a judge confidence onto the reference script's 0-100 scale.

    The grader prompt asks for a percentage. Legacy values at or below 1 are
    read as fractions of 1. (A genuine "1%" is indistinguishable from "100%"
    here - no judge emits it.)
    """
    numeric = _as_float(value)
    if numeric is None:
        return None
    if 0.0 <= numeric <= 1.0:
        return numeric * 100.0
    return min(100.0, numeric)


def _search_tools(lb_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Which tool names count as a search call (locate tools). Deep reads
    (list_chunks) are reported in avg_tool_stats but are not searches: they
    open a document the agent already located."""
    configured = lb_cfg.get("search_tools")
    if configured is None:
        return DEFAULT_SEARCH_TOOLS
    return tuple(str(name).strip() for name in configured if str(name).strip())


def _calibration_bin_size(lb_cfg: dict[str, Any]) -> int:
    try:
        size = int(lb_cfg.get("calibration_bin_size") or DEFAULT_CALIBRATION_BIN_SIZE)
    except (TypeError, ValueError):
        size = DEFAULT_CALIBRATION_BIN_SIZE
    return max(1, size)


def _usage_row(query_id: str, row: dict[str, Any], search_tools: tuple[str, ...]) -> dict[str, Any]:
    """Per-question token and time cost, straight from the backend's run
    accounting plus the client-side wall clock the answer phase records."""
    usage = row.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    counts = row.get("tool_call_counts")
    counts = counts if isinstance(counts, dict) else {}
    retrieved = row.get("retrieved_docids")

    prompt_tokens = _as_int(usage.get("prompt_tokens"))
    completion_tokens = _as_int(usage.get("completion_tokens"))
    total_tokens = _as_int(usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "query_id": query_id,
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "llm_calls": _as_int(usage.get("llm_calls")),
        # Server-side: the pipeline's own timing for the turn (agentic run plus
        # its delivery gate). Client-side: everything the benchmark measured,
        # i.e. the above plus transport, queueing and response decoding.
        "server_seconds": _as_float(row.get("server_elapsed_seconds")),
        "client_seconds": _as_float(row.get("answer_elapsed_seconds")),
        "search_calls": sum(int(_as_float(counts.get(name)) or 0) for name in search_tools),
        "tool_calls": {str(name): int(_as_float(count) or 0) for name, count in counts.items()},
        "retrieved_doc_count": len(retrieved) if isinstance(retrieved, list) else None,
        # Delivery-gate audit accounting (agentic runs on a backend that
        # reports it): suspects is the suspect count of EVERY audit round the
        # gate ran, oldest first (a PASS round reports 0, so the list length
        # doubles as the audit round count); passed is the LAST audit's
        # verdict. None when the backend did not report gate_audit.
        "audit_suspects": _gate_audit_suspects(row),
        "audit_passed": _gate_audit_passed(row),
        # Chunk-read depth split (None when the backend did not report it):
        # deep = full chunk content the model read, shallow = snippet windows
        # only. Together they show where a run's context weight came from.
        "deep_read_chunks": _as_int(row.get("deep_read_chunks")),
        "shallow_read_chunks": _as_int(row.get("shallow_read_chunks")),
    }


def _gate_audit_suspects(row: dict[str, Any]) -> list[int] | None:
    """The per-round suspect counts of the delivery gate's answer auditor."""
    audit = row.get("gate_audit")
    if not isinstance(audit, dict):
        return None
    suspects = audit.get("suspects")
    if not isinstance(suspects, list):
        return None
    out = [_as_int(s) for s in suspects]
    return out if all(s is not None for s in out) else None


def _gate_audit_passed(row: dict[str, Any]) -> bool | None:
    """Whether the delivery gate's LAST audit verdict was PASS."""
    audit = row.get("gate_audit")
    if not isinstance(audit, dict) or "passed" not in audit:
        return None
    return bool(audit.get("passed"))


def calibration_error(confidences: list[float], correctness: list[bool], beta: int = DEFAULT_CALIBRATION_BIN_SIZE) -> float:
    """L2 calibration error, ported line for line from the reference script's
    calib_err() (itself from Hendrycks' outlier-exposure tools).

    The loop is `range(len(bins) - 1)`, so the final bin - the one extended to
    hold the remainder - is never scored: with 830 questions and beta=100 the
    last 130 samples are excluded. That is a bug in the reference script, but it
    is reproduced here on purpose - "fixing" it would make the number
    incomparable with the published leaderboard entries.

    One difference is unavoidable: the reference sorts with numpy's argsort,
    which is an UNSTABLE quicksort, so when many answers share a confidence
    (our agent states none, so they are all 100) it permutes the ties
    arbitrarily and the per-bin correctness averages move with it - measured
    divergence on an all-ties input: ~9 percentage points. This port sorts
    stably, which is at least reproducible run to run.
    """
    if not confidences or len(confidences) != len(correctness):
        return 0.0

    confidence = [(_as_float(c) or 0.0) / 100.0 for c in confidences]
    correct = [1.0 if c else 0.0 for c in correctness]
    order = sorted(range(len(confidence)), key=lambda index: confidence[index])
    confidence = [confidence[index] for index in order]
    correct = [correct[index] for index in order]
    total = len(confidence)

    bins = [[i * beta, (i + 1) * beta] for i in range(total // beta)]
    if not bins:
        return 0.0
    bins[-1] = [bins[-1][0], total]

    cerr = 0.0
    for i in range(len(bins) - 1):
        lo, hi = bins[i]
        size = hi - lo
        if size <= 0:
            continue
        bin_confidence = confidence[lo:hi]
        bin_correct = correct[lo:hi]
        difference = abs(sum(bin_confidence) / size - sum(bin_correct) / size)
        cerr += size / total * difference * difference

    return math.sqrt(cerr) * 100.0


# --------------------------------------------------------------------------
# Judge scoring
# --------------------------------------------------------------------------
# Patterns mirror parse_judge_response() in the reference evaluation script
# (scripts_evaluation/evaluate_with_openai.py): a model may or may not bold the
# label, so each field is tried in its bold, colon-outside-bold, and plain
# forms before giving up.
_GRADER_ANSWER_PATTERNS = (
    r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
    r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
    r"extracted_final_answer:\s*(.*?)(?=\n|$)",
)
_GRADER_REASONING_PATTERNS = (
    r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
    r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
    r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
)
_GRADER_CORRECT_PATTERNS = (
    r"\*\*correct:\*\*\s*(yes|no)",
    r"\*\*correct\*\*:\s*(yes|no)",
    r"correct:\s*(yes|no)",
)
_GRADER_CONFIDENCE_PATTERNS = (
    r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
    r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
    r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
)


def _grader_field(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def parse_grader_verdict(text: str) -> dict[str, Any]:
    """Parse the leaderboard judge's verdict out of GRADER_TEMPLATE's answer.

    Returns {"correct": bool, "confidence": float, "extracted_final_answer":
    str|None, "reasoning": str|None}. Raises when `correct` is missing: the
    leaderboard's accuracy is a boolean and a judge reply without one is a
    failed judgement, never an implicit "no".
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)

    raw_correct = _grader_field(cleaned, _GRADER_CORRECT_PATTERNS)
    if raw_correct is None:
        raise ValueError(f"Could not parse judge correctness from: {cleaned[:500]}")

    # The reference script reads the confidence the ANSWER states and defaults
    # to 100 when there is none, so an agent that never states one is scored as
    # maximally confident - the calibration error then equals its miss rate.
    confidence = 100.0
    raw_confidence = _grader_field(cleaned, _GRADER_CONFIDENCE_PATTERNS)
    if raw_confidence is not None:
        confidence = min(100.0, float(raw_confidence))

    return {
        "correct": raw_correct.lower() == "yes",
        "confidence": confidence,
        "extracted_final_answer": _grader_field(cleaned, _GRADER_ANSWER_PATTERNS),
        "reasoning": _grader_field(cleaned, _GRADER_REASONING_PATTERNS),
    }


def _extract_verdict(parsed: dict[str, Any]) -> str | None:
    for key in ("verdict", "result", "judgement", "judgment"):
        value = parsed.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized == "correct":
            return "correct"
        if normalized == "incorrect":
            return "incorrect"
        if normalized.startswith("partial"):
            return "partial"
    return None


def _coerce_accuracy(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "correct", "yes"}:
            return 1.0
        if lowered in {"false", "incorrect", "no"}:
            return 0.0
        try:
            score = float(lowered.rstrip("%"))
        except ValueError:
            return None
        if value.strip().endswith("%"):
            score = score / 100.0
    else:
        return None

    if 1.0 < score <= 100.0:
        score = score / 100.0
    return max(0.0, min(1.0, score))


# --------------------------------------------------------------------------
# Answer / reference extraction
# --------------------------------------------------------------------------
def extract_answer_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("answer", "content", "text", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                nested = extract_answer_text(value)
                if nested:
                    return nested
        if "data" in payload:
            return extract_answer_text(payload["data"])
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            return extract_answer_text(choices[0])
    return ""


def extract_reference(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    reference = payload.get("reference") or payload.get("references")
    if isinstance(reference, dict):
        return reference
    if isinstance(payload.get("data"), dict):
        return extract_reference(payload["data"])
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            reference = extract_reference(choice)
            if reference:
                return reference
    return {}


def extract_run_stats(payload: Any) -> dict[str, Any]:
    """Read the run accounting the backend ships next to the answer.

    Keys, each present only when the backend actually reported it:
      * `tool_call_counts` - how many times each tool ran (Search Calls);
      * `retrieved_docids` - every document the retrieval tools surfaced
        (Recall is measured against this union, not against the citations);
      * `usage`            - {prompt_tokens, completion_tokens, total_tokens,
        llm_calls} for the WHOLE turn, not just the final answer;
      * `server_elapsed_seconds` - the pipeline's own timing for the turn.

    All of them are emitted by the Go backend on non-streaming agentic
    responses only: a streaming run assembles the answer text on the client and
    never sees them, and neither does a backend built before the fields
    existed. Absent keys are left out rather than defaulted to zero, so a row
    can be flagged as "no accounting" instead of being scored as a run that
    made no calls and retrieved nothing.
    """
    if not isinstance(payload, dict):
        return {}

    data = payload.get("data") if isinstance(payload.get("data"), dict) else None
    source = payload if "tool_call_counts" in payload or "retrieved_docids" in payload else (data or payload)

    stats: dict[str, Any] = {}

    counts = source.get("tool_call_counts")
    if isinstance(counts, dict):
        stats["tool_call_counts"] = {str(name): int(_as_float(count) or 0) for name, count in counts.items() if _as_float(count) is not None}

    # Per-tool failure accounting (agentic runs on a backend that reports
    # it): how many calls returned a failure notice, plus one representative
    # root cause per tool - the diagnosis channel for retrieval outages
    # (e.g. the embedding backend draining its balance) and argument bugs.
    errors = source.get("tool_call_errors")
    if isinstance(errors, dict):
        err_counts = {str(name): int(_as_float(count) or 0) for name, count in errors.items() if _as_float(count) is not None}
        if err_counts:
            stats["tool_call_errors"] = err_counts

    samples = source.get("tool_error_samples")
    if isinstance(samples, dict) and samples:
        stats["tool_error_samples"] = {str(name): str(text) for name, text in samples.items()}

    docs = source.get("retrieved_docids")
    if isinstance(docs, list):
        stats["retrieved_docids"] = _unique_doc_ids(docs)

    usage = source.get("usage")
    if isinstance(usage, dict):
        stats["usage"] = {key: _as_int(usage.get(key)) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls")}

    elapsed = _as_float(source.get("elapsed_seconds"))
    if elapsed is not None:
        stats["server_elapsed_seconds"] = round(elapsed, 3)

    # Chunk-read accounting (agentic runs on a backend that reports it): how
    # many chunks the retrieval tools put in front of the model, split by
    # read depth - deep = full chunk content (list_chunks / search_chunks),
    # shallow = snippet windows (grep_chunks / search_bm25_chunks).
    for key in ("deep_read_chunks", "shallow_read_chunks"):
        value = _as_int(source.get(key))
        if value is not None:
            stats[key] = value

    gate_audit = source.get("gate_audit")
    if isinstance(gate_audit, dict):
        suspects = gate_audit.get("suspects")
        if isinstance(suspects, list):
            stats["gate_audit"] = {
                "suspects": [_as_int(s) for s in suspects],
                "passed": bool(gate_audit.get("passed")),
            }

    return stats


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load questions from .jsonl, a JSON list, or a JSON {id: {...}} mapping."""
    if path.suffix.lower() == ".jsonl":
        return _load_questions_jsonl(path)

    raw = _load_json(path)
    if isinstance(raw, list):
        return [_normalize_question(row, index) for index, row in enumerate(raw, start=1)]
    if isinstance(raw, dict):
        if isinstance(raw.get("questions"), list):
            return [_normalize_question(row, index) for index, row in enumerate(raw["questions"], start=1)]
        return [_normalize_question(payload, question_id) for question_id, payload in sorted(raw.items(), key=lambda item: _natural_key(str(item[0])))]
    raise ValueError(f"Unsupported question file format: {path}")


def _load_questions_jsonl(path: Path) -> list[dict[str, Any]]:
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        questions.append(_normalize_question(payload, line_number))
    return questions


def load_evidence_doc_ids(path: Path) -> dict[str, list[str]]:
    """question_id -> evidence document ids, read from the questions file.

    Returns an empty mapping (with a warning) when the file cannot be read: the
    leaderboard export then falls back to whatever the answer rows carry, and a
    run over a moved/renamed dataset still produces its accuracy numbers.
    """
    try:
        questions = load_questions(path)
    except (OSError, TypeError, ValueError) as exc:
        print(f"[leaderboard] could not read evidence documents from {path}: {exc}")
        return {}
    return {str(question["question_id"]): sorted(_unique_doc_ids(question.get("evidence_doc_ids") or [])) for question in questions}


def _normalize_question(payload: Any, fallback_id: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"Question row {fallback_id} must be a JSON object")
    return {
        "question_id": str(_first_value(payload, ID_KEYS) or fallback_id),
        "question": str(_first_value(payload, QUESTION_KEYS) or ""),
        "gold_answer": str(_first_value(payload, GOLD_KEYS) or ""),
        "reasoning_types": _as_string_list(_first_value(payload, REASONING_TYPE_KEYS)),
        "expected_doc_ids": _first_doc_list(payload, EXPECTED_DOC_KEYS),
        "evidence_doc_ids": _first_doc_list(payload, EVIDENCE_DOC_KEYS) or _first_doc_list(payload, EXPECTED_DOC_KEYS),
    }


def _first_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if payload.get(key) not in (None, ""):
            return payload[key]
    return None


def _first_doc_list(payload: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    for key in keys:
        doc_ids = _as_doc_id_list(payload.get(key))
        if doc_ids:
            return doc_ids
    return []


def select_questions(
    questions: list[dict[str, Any]],
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the config's `question_ids` selection block.

    {
      "include": [55, 67],   # pick exactly these ids
      "random": 100,         # ... or sample this many
      "exclude": [55, ...]   # dropped from the result of either strategy
    }

    include and random are alternative strategies: if both are set, include
    wins; if neither is effective (empty include, random <= 0) every question
    is selected. exclude is applied last in all cases; for the random strategy
    the sample is drawn from the pool after exclusion.
    """
    include = [str(qid).strip() for qid in (selection.get("include") or []) if str(qid).strip()]
    try:
        random_count = int(selection.get("random") or 0)
    except (TypeError, ValueError):
        random_count = 0
    excluded = {str(qid).strip() for qid in (selection.get("exclude") or [])}

    if include:
        by_id = {str(question["question_id"]): question for question in questions}
        missing = [qid for qid in include if qid not in by_id]
        if missing:
            raise ValueError(f"question_ids.include contains unknown id(s): {', '.join(missing)}")
        selected = [by_id[qid] for qid in include]
        return [question for question in selected if str(question["question_id"]) not in excluded]

    pool = [question for question in questions if str(question["question_id"]) not in excluded]
    if random_count <= 0:
        return pool
    if random_count >= len(pool):
        return pool
    # Seed with the current epoch second: every run draws a fresh sample by
    # design, reproducibility comes from question_ids.include instead.
    return random.Random(int(time.time())).sample(pool, random_count)


def _run_id(question_id: Any) -> str:
    return str(question_id)


def _row_run_key(row: dict[str, Any]) -> str | None:
    run_id = row.get("run_id")
    if run_id:
        return str(run_id)
    question_id = row.get("question_id")
    return str(question_id) if question_id is not None else None


def _is_backend_error_verdict(row: dict[str, Any]) -> bool:
    """True when the judge never produced a verdict. Two shapes exist: the
    backend's error bubble as the judge answer text (provider quota walls), or
    an unparseable verdict - both carry judge_error and no verdict. Records
    with a real verdict (including legitimate fails) are never touched."""
    if str(row.get("judge_answer") or "").lstrip().startswith("**ERROR**"):
        return True
    return _as_float(row.get("accuracy")) is None and bool(row.get("judge_error"))


def _dedupe_last(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a JSONL file to one row per run_id (the last one wins) - a
    re-judged question appends a fresh row that shadows the earlier failure."""
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _row_run_key(row)
        if key is None:
            deduped[f"__idx__{id(row)}"] = row
            continue
        deduped[key] = row
    return list(deduped.values())


# --------------------------------------------------------------------------
# Config / IO helpers
# --------------------------------------------------------------------------
def _make_client(cfg: dict[str, Any]) -> JsonHttpClient:
    api_key = cfg.get("ragflow_api_key") or os.environ.get("RAGFLOW_API_KEY", "")
    if not api_key:
        raise ValueError("ragflow_api_key is required in the config (or set the RAGFLOW_API_KEY environment variable)")
    return JsonHttpClient(
        base_url=_base_url(cfg.get("backend", {})),
        api_key=api_key,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        backoff_seconds=DEFAULT_BACKOFF_SECONDS,
    )


def _base_url(backend: dict[str, Any]) -> str:
    if backend.get("base_url"):
        return str(backend["base_url"]).rstrip("/")
    scheme = backend.get("scheme", "http")
    host = backend.get("host", "127.0.0.1")
    port = backend.get("port", 80)
    return f"{scheme}://{host}:{port}"


def _dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("dataset") or {}


def _questions_path(cfg: dict[str, Any]) -> str:
    """dataset.questions_path is canonical; flat top-level keys still work so an
    ad-hoc config does not need the dataset wrapper."""
    dataset = _dataset(cfg)
    for value in (
        dataset.get("questions_path"),
        cfg.get("questions_path"),
        cfg.get("frames_questions_path"),
        cfg.get("frames_mapping_path"),
    ):
        if value:
            return str(value)
    raise ValueError("dataset.questions_path is required")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = _row_run_key(row)
        if key is not None:
            keyed[key] = row
    return keyed


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _resolve_output_dir(value: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    resolved = Path(value.replace("<timestamp>", timestamp)).expanduser()
    return resolved if resolved.is_absolute() else (Path.cwd() / resolved)


def _read_id_list(path: Path) -> list[str]:
    """Read a question-id list: ids separated by commas, whitespace or
    newlines; `#` starts a comment. Used by `question_ids.include_file`, so a
    generated batch (e.g. the ids a previous run judged wrong) lives in its own
    reviewable file instead of inline in the JSON config."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    ids: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for part in line.replace(",", " ").split():
            if part.strip():
                ids.append(part.strip())
    return ids


def _decode_response(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}


def _collect_eventstream_answer(lines: Any) -> str:
    answer = ""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("code", 0) not in (0, None):
            raise BenchmarkError(f"RAGFlow code {payload.get('code')}: {payload.get('message') or _compact(payload)}")
        answer = _append_answer_part(answer, payload)
    return answer


def _append_answer_part(answer: str, payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return answer
    data = payload.get("data")
    if isinstance(data, dict):
        if isinstance(data.get("answer"), str):
            return answer + data["answer"]
        if data.get("reference"):
            return answer
    if isinstance(payload.get("answer"), str):
        return answer + payload["answer"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            return answer + delta["content"]
    return answer


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, requests.RequestException):
        return True
    message = str(exc)
    return "HTTP 5" in message or "timed out" in message or "Timeout" in message


def _extract_session_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("session_id", "conversation_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    if isinstance(payload.get("data"), dict):
        return _extract_session_id(payload["data"])
    return None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    return None if numeric is None else int(numeric)


def _round_percent(value: float | None) -> float | None:
    """Leaderboard recall is a percentage rounded to two decimals; None stays
    None (a question with no labelled evidence has no recall, not a zero one)."""
    return None if value is None else round(value * 100.0, 2)


def _as_string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _as_doc_id_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    values = [value] if isinstance(value, (str, int, float)) else value
    if not isinstance(values, list):
        return []
    return _unique_doc_ids(_normalize_benchmark_doc_id(item) for item in values)


def _unique_doc_ids(values: Any) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    for value in values:
        normalized = _normalize_benchmark_doc_id(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        doc_ids.append(normalized)
    return doc_ids


def _normalize_benchmark_doc_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    filename = text.replace("\\", "/").rsplit("/", 1)[-1]
    if filename.lower().endswith(".md"):
        filename = filename[:-3]
    return filename.strip() or None


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _average(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return round(sum(cleaned) / len(cleaned), 6) if cleaned else None


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return (0, f"{int(value):012d}")
    except ValueError:
        return (1, value)


def _compact(payload: Any) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return text[:1000]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run questions against one RAGFlow chat and judge the answers with another (serial execution).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  qa_benchmark.py --config browsecomp_conf.json --ids 11,33 --dry-run\n"
            "  qa_benchmark.py --config frames_conf.json --ids 27,71\n"
            "  qa_benchmark.py --run outputs/qa_benchmark_20260902_120000 --skip-answers\n"
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to the benchmark config JSON")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Questions answered (and judged) in parallel; overrides the config's concurrency (default: 1 = serial)",
    )
    parser.add_argument("--ids", help="Comma-separated question ids; overrides question_ids.include from the config")
    parser.add_argument("--run", help="Reuse an existing output directory instead of creating a new one")
    parser.add_argument("--overwrite", action="store_true", help="Drop existing JSONL files instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without calling any API")
    parser.add_argument("--skip-answers", action="store_true", help="Judge an existing answers.jsonl")
    parser.add_argument("--skip-judge", action="store_true", help="Only collect answers")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _load_json(config_path)
    base_dir = config_path.parent

    output_cfg = cfg.get("output", {})
    if args.run:
        output_dir = Path(args.run).expanduser().resolve()
    else:
        output_dir = _resolve_output_dir(output_cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    answers_path = output_dir / output_cfg.get("answers_jsonl", DEFAULT_ANSWERS_NAME)
    leaderboard_path = output_dir / output_cfg.get("leaderboard_json", DEFAULT_LEADERBOARD_NAME)

    # CLI --ids replaces question_ids.include for this run; everything else in
    # the question_ids block (random is moot, exclude still applies) is kept.
    selection = dict(cfg.get("question_ids") or {})
    # A generated batch (e.g. the ids a previous run judged wrong) belongs in
    # the config as a file reference rather than inline: it stays reviewable
    # and regenerable, and the resume command needs no --ids. Precedence:
    # --ids beats include_file beats the inline include list.
    include_file = str(selection.get("include_file") or "").strip()
    if include_file and not args.ids:
        ids_path = _resolve_path(include_file, base_dir)
        file_ids = _read_id_list(ids_path)
        if not file_ids:
            parser.error(f"question_ids.include_file is missing or empty: {ids_path}")
        selection["include"] = file_ids
        print(f"[ids] {len(file_ids)} id(s) from {ids_path}")
    if args.ids:
        selection["include"] = [part.strip() for part in args.ids.split(",") if part.strip()]

    if answers_path.exists() and not args.overwrite:
        existing = _dedupe_last(_read_jsonl(answers_path))
        answered = sum(1 for row in existing if not (row.get("ragflow_error") or "").strip())
        judged = sum(1 for record in _read_judgements(leaderboard_path).values() if not _is_backend_error_verdict(record))
        print(f"[resume] {answers_path.name}: {len(existing)} row(s) - {answered} answered (skipped), {len(existing) - answered} failed/aborted will be retried; {judged} verdict(s) already stored")

    # Parallelism: the CLI flag wins, then the config's top-level
    # "concurrency", then 1 (strictly serial - the historical behavior).
    try:
        concurrency = int(args.concurrency if args.concurrency is not None else cfg.get("concurrency") or 1)
    except (TypeError, ValueError):
        concurrency = 1
    concurrency = max(1, concurrency)

    selected: list[dict[str, Any]] = []
    if args.skip_answers:
        if not answers_path.exists():
            parser.error(f"--skip-answers needs an existing answers file: {answers_path}")
        print(f"Reusing answers file {answers_path}")
    else:
        questions_path = _resolve_path(_questions_path(cfg), base_dir)
        questions = load_questions(questions_path)
        selected = select_questions(questions, selection)
        print(f"Loaded {len(questions)} questions from {questions_path}")
        dataset = _dataset(cfg)
        if dataset:
            print(f"Dataset: {dataset.get('name') or '(unnamed)'} - {dataset.get('description') or '(no description)'}")
            if dataset.get("corpus_path"):
                print(f"Corpus: {dataset['corpus_path']}")
        print(f"Selected {len(selected)} question(s) (concurrency={concurrency})")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        for question in selected[:10]:
            print(f"- {question['question_id']}: {question['question'][:100]}")
        if len(selected) > 10:
            print(f"... {len(selected) - 10} more")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_answers:
        if args.overwrite and answers_path.exists():
            answers_path.unlink()
        answer_aborted = run_answer_phase(
            client=_make_client(cfg),
            cfg=cfg,
            questions=selected,
            answers_path=answers_path,
            concurrency=concurrency,
        )
        if answer_aborted:
            print(f"Run aborted at the answer phase; {answers_path} holds the completed rows.")
            return 1
        if args.skip_judge:
            print(f"Answers written to {answers_path}")
            return 0

    judgements, judge_aborted = run_judge_phase(
        client=_make_client(cfg),
        cfg=cfg,
        answers_path=answers_path,
        leaderboard_path=leaderboard_path,
        concurrency=concurrency,
    )

    # The leaderboard export scores retrieval recall against each question's
    # EVIDENCE documents, which live in the questions file rather than in the
    # answers - an answers.jsonl from an older run predates the field, and
    # --skip-answers never re-reads the questions at all.
    rows = _merge_judgements(_dedupe_last(_read_jsonl(answers_path)), judgements)
    leaderboard = build_leaderboard(rows, cfg, load_evidence_doc_ids(_resolve_path(_questions_path(cfg), base_dir)))
    _write_json(leaderboard_path, leaderboard)

    if judge_aborted:
        print("Run aborted at the judge phase; re-run the same command once the quota is replenished.")
        return 1
    print(f"Leaderboard submission written to {leaderboard_path}")
    print(_leaderboard_summary(leaderboard))
    return 0


def _leaderboard_summary(leaderboard: dict[str, Any]) -> str:
    """One-line-per-metric digest of the submission, so a run's console output
    is enough to sanity check it without opening the JSON."""
    usage = leaderboard.get("per_query_usage") or []
    input_tokens = sum(row.get("input_tokens") or 0 for row in usage)
    output_tokens = sum(row.get("output_tokens") or 0 for row in usage)
    client_seconds = sum(row.get("client_seconds") or 0.0 for row in usage)
    deep_chunks = sum(row.get("deep_read_chunks") or 0 for row in usage)
    shallow_chunks = sum(row.get("shallow_read_chunks") or 0 for row in usage)
    judgements = leaderboard.get("per_query_judgements") or []
    scored = [j for j in judgements if j.get("correct") is not None]
    judged_ok = sum(1 for j in scored if j.get("correct"))
    unjudged = len(judgements) - len(scored)
    judged_only = f"{judged_ok / len(scored) * 100:.2f}%" if scored else "n/a"
    return "\n".join(
        [
            "leaderboard:",
            f"  LLM              : {leaderboard.get('LLM')}",
            f"  Retriever        : {leaderboard.get('Retriever')}",
            f"  Accuracy (%)     : {leaderboard.get('Accuracy (%)')}",
            f"  Accuracy (judged): {judged_only}  ({judged_ok}/{len(scored)} scored rows; {unjudged} unjudged counted incorrect)" if scored else f"  Accuracy (judged): {judged_only}",
            f"  Recall (%)       : {leaderboard.get('Recall (%)')}",
            f"  Search Calls     : {leaderboard.get('Search Calls')}",
            f"  Calibration (%)  : {leaderboard.get('Calibration Error (%)')}",
            f"  avg_tool_stats   : {leaderboard.get('avg_tool_stats')}",
            f"  tool_errors      : {leaderboard.get('tool_error_stats') or 'none'}",
            *(f"  tool_error[{name}]: {text}" for name, text in (leaderboard.get("tool_error_samples") or {}).items()),
            f"  chunk reads      : deep={deep_chunks} shallow={shallow_chunks}",
            f"  tokens           : input={input_tokens} output={output_tokens}",
            f"  wall clock (s)   : {round(client_seconds, 1)}",
        ]
    )


if __name__ == "__main__":
    sys.exit(main())
