#!/usr/bin/env python3
"""Dataset-agnostic QA benchmark runner for RAGFlow.

How a run works:
  * One question loader - .jsonl, JSON list, or JSON {id: {...}} mapping.
  * An answer phase (a RAGFlow chat answers each question) followed by a judge
    phase (a SEPARATE judge chat scores each answer), both resumable and both
    parallelisable via concurrency.
  * One judge scorer: the 0/2/4 rule book first, tolerant fallbacks after.
  * One result artefact: leaderboard.json (the BrowseComp-Plus submission JSON
    plus per-question judgement/usage extensions) carries everything a run
    produces; answers.jsonl stays a pure record of what the agent returned.

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
  question_ids     {include: [..], include_file: "ids.txt", random: N,
                   exclude: [..]}
                   include and random are alternative strategies - if both are
                   set include wins, if neither is effective every question is
                   selected. exclude is dropped from the result of either
                   strategy (the random sample is drawn after exclusion).
                   random is seeded with the current epoch second, so every run
                   draws a fresh sample; use include for reproducibility.
                   include_file points at a file next to the config holding the
                   batch (ids separated by commas/whitespace/newlines, `#`
                   comments allowed) - the recommended carrier for a re-run
                   batch. Precedence: --ids > include_file > include.
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

Re-running a batch efficiently (resume):
  A batch that was interrupted (machine restart, Ctrl-C, or a bounded
  `--wall-max-waits` give-up) is re-run by repeating the SAME command.
  A provider quota wall does NOT interrupt it: the phase waits it out in
  process (see step 4). Everything needed to continue lives in the config and in
  the run directory - there is no separate resume script and no flag to remember.

  1. Carry the batch in the config, not on the command line. `question_ids`
     takes either `include` (inline ids) or `include_file` (a path relative to
     the config; ids separated by commas, whitespace or newlines, `#` comments).
     A GENERATED batch - e.g. every id a previous run judged wrong - belongs in
     a file: reviewable, diffable, regenerable.
         "question_ids": {"include_file": "browsecompplus_retry_ids.txt"}
     Precedence: --ids beats include_file beats include.

  2. Drop <timestamp> from output.output_dir. A timestamped path starts a NEW
     run on every invocation; a fixed one is the batch's permanent home, and
     each invocation continues it:
         "output": {"output_dir": "outputs/browsecomp_retry_batch"}
     (--run DIR aims a single invocation at another directory; --overwrite discards
     its JSONL and starts clean.)

  3. Skipping finished work IS the resume - no state file, no bookkeeping:
       answers: a question counts as finished when its row carries an answer
                and NO ragflow_error, OR when its error is PERSISTENT (the
                provider refused the input itself - content policy, e.g.
                MiniMax's `input new_sensitive`; a retry reproduces the
                refusal). Rows killed mid-flight with a TRANSIENT error (the
                **ERROR** text a quota wall leaves) are retried.
                `[resume] ... N answered + P persistent-error (skipped),
                M transient failed/aborted will be retried` prints the split.
       judge:   verdicts live in leaderboard.json's per_query_judgements and are
                persisted one row at a time, so judging resumes at the first row
                without one; a verdict that is ITSELF a backend error is re-judged
                rather than counted.
     Above concurrency 1 answer rows land out of order; resume is id-based, so
     that changes nothing.

  4. A quota wall is ridden out IN PROCESS (no scheduler needed). When the
     provider plan is exhausted the phase stops dispatching, waits
     WALL_RETRY_WAIT_SEC (30 min, `--wall-wait`), then re-enters itself and
     retries exactly the work still missing an answer/verdict. That repeats for
     as long as it takes: the wait count is unbounded unless you pass
     `--wall-max-waits N`, and every finished row/verdict stays on disk
     throughout. So a single invocation survives a plan wall:
         python3 scripts/ragflow_benchmark.py --config <conf>
     A scheduler is still the right tool for a MACHINE restart (nothing can
     survive a reboot), which is what the rest of this section covers.

  5. Scheduling it - systemd user timer (no root, survives the terminal):

     The unit pair is installed per user; name it after the batch:
         ~/.config/systemd/user/<batch>.service   Type=oneshot; WorkingDirectory=the
                                                  repo; ExecStart=<bracketed pgrep
                                                  guard> && the python command above;
                                                  stdout/stderr appended to the batch log
         ~/.config/systemd/user/<batch>.timer     OnCalendar=hourly
     Install, inspect and remove:
         systemctl --user daemon-reload
         systemctl --user enable --now <batch>.timer        # start the schedule
         systemctl --user list-timers <batch>.timer         # next/last run
         systemctl --user status <batch>.service            # outcome of the last tick
         journalctl --user -u <batch>.service -n 50         # unit-level log
         systemctl --user stop <batch>.timer                # pause (keeps it installed)
         systemctl --user disable --now <batch>.timer       # remove from the schedule
         tail -f outputs/<batch>.log                        # the batch's own log
     Concrete example installed for the browsecomp retry batch:
         browsecomp-retry-resume.timer -> browsecomp-retry-resume.service
         -> scripts/ragflow_benchmark.py --config scripts/browsecompplus_retry_conf.json
         log: outputs/browsecomp_retry_batch_run.log

     cron equivalent (hosts without systemd):
         0 * * * * cd /path/to/ragflow && pgrep -f "[r]agflow_benchmark.py" >/dev/null || setsid nohup /home/zhichyu/.venv/bin/python3 -u scripts/ragflow_benchmark.py --config scripts/browsecompplus_retry_conf.json >> outputs/browsecomp_retry_batch_run.log 2>&1 &

     Both forms rely on the same guard: the pgrep check refuses to start a second
     run, and the run's own skip logic ignores finished rows and verdicts. BRACKET
     the pattern's first character (`[r]agflow_benchmark.py`, `[b]atch.json`): a
     plain `pgrep -f ragflow_benchmark` ALSO matches the wrapper shell that is
     running the guard itself — its command line contains the pattern, and the
     python path — so the guard always concludes "already running", exits 0, and
     the schedule silently never runs the batch. Measured on the systemd timer:
     every hourly tick exited in 0.3s for a day. A run started by hand is enough
     to prove the schedule works: check `systemctl --user status <batch>.service`
     (or the cron mail) and confirm the batch log grew. Delete or disable the
     schedule once the batch has no remaining
     questions (the `[resume]` line then reports 0 pending).
     questions (the `[resume]` line then reports 0 pending).

  6. Judge an existing batch without re-answering: --skip-answers.

  7. Watch a resumed batch: the per-question `[answers] seq/total` counts the
     PENDING rows (not the batch position), and `[judge]` prints one verdict per
     row plus the skipped count for rows already judged.

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
import subprocess
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
# Raised 1800 -> 2400 on 2026-09-17: the server's smart-reasoning wall is now 30
# minutes, so a 1800s client timeout would cut off questions exactly when the
# server was about to answer them (observed: a question failing at exactly 1800s
# with "Read timed out (read timeout=1800)").
DEFAULT_TIMEOUT_SECONDS = 2400
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
DEFAULT_SEARCH_TOOLS = ("search_chunks", "search_semantic_chunks", "grep_chunks", "search_bm25_chunks")

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

# Riding out an exhausted plan IN PROCESS (2026-09-18): a wall used to END the
# run, which is what forced an external scheduler (systemd timer / cron) to
# re-launch it every hour. The phases now wait WALL_RETRY_WAIT_SEC and retry the
# still-pending work themselves, with NO upper bound by default
# (WALL_MAX_WAITS = 0). The wait is deliberately long: a Token Plan refills on a
# clock, not on a retry, so probing every few minutes only re-arms the burst
# detector and burns quota on calls that cannot succeed yet.
#   --wall-wait SECONDS   interval between retries (default 1800 = 30 min)
#   --wall-max-waits N    bound the waits (default 0 = unlimited); a bounded run
#                         that gives up still exits 1 with its rows on disk.
WALL_RETRY_WAIT_SEC = 1800  # 30 minutes between wall retries
WALL_MAX_WAITS = 0  # 0 = keep waiting until the plan refills
WALL_WAIT_SLICE_SEC = 30  # sleep slice: keeps Ctrl-C and the log responsive
WALL_WAIT_LOG_EVERY_SEC = 300  # progress line while waiting

# The judge phase used to cap its own concurrency below the answer phase's
# (rationale: one short chat call per row, so high concurrency bought little
# while multiplying rate-limit damage). Dropped per user decision 2026-09-30:
# with 700+ rows to judge a capped judge phase became the long pole of the
# whole run, so judging now runs at the SAME concurrency as the answers.

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
    or "wall" (the plan is exhausted: the caller waits and retries). See the
    QUOTA_* constants for the rationale: with concurrency > 1 a single
    rate-limit burst fails several in-flight questions at once, which must not
    be mistaken for an exhausted plan.
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


def _wait_for_plan_refill(*, label: str, wait_sec: float, round_no: int, max_waits: int) -> None:
    """Sleep out a provider plan wall before the next retry round.

    Sliced rather than one long sleep, for two reasons: Ctrl-C stays responsive,
    and the log keeps proving the process is parked on purpose (a 30-minute
    silence is indistinguishable from a hang). The wait is long by design - a
    Token Plan refills on a clock, not on a retry.
    """
    limit = "unlimited" if max_waits <= 0 else f"at most {max_waits} wait(s)"
    print(
        f"[{label}] provider plan exhausted - waiting {int(wait_sec)}s before retry round {round_no} ({limit}); every finished row/verdict stays on disk",
        flush=True,
    )
    remaining = float(wait_sec)
    slept = 0.0
    next_notice = WALL_WAIT_LOG_EVERY_SEC
    while remaining > 0:
        slice_sec = min(WALL_WAIT_SLICE_SEC, remaining)
        time.sleep(slice_sec)
        slept += slice_sec
        remaining -= slice_sec
        if remaining > 0 and slept >= next_notice:
            print(f"[{label}] still waiting: {int(remaining)}s left before retry round {round_no}", flush=True)
            next_notice += WALL_WAIT_LOG_EVERY_SEC


def run_answer_phase(
    *,
    client: JsonHttpClient,
    cfg: dict[str, Any],
    questions: list[dict[str, Any]],
    answers_path: Path,
    concurrency: int = 1,
    wall_wait_sec: float = WALL_RETRY_WAIT_SEC,
    wall_max_waits: int = WALL_MAX_WAITS,
) -> bool:
    """Answer every pending question, riding out an exhausted plan in process.

    A provider wall no longer ends the run. Each round answers what it can; a
    wall parks the phase for wall_wait_sec and then re-enters it, which retries
    exactly the questions still missing an answer (the resume rule: a row counts
    as finished only when it carries an answer and no ragflow_error). The loop
    is unbounded unless wall_max_waits > 0, so no external scheduler is needed
    to ride out a plan wall - only a machine reboot still needs one.

    Returns True only when a bounded wait budget gave up (wall_max_waits > 0).
    """
    round_no = 0
    waits = 0
    while True:
        round_no += 1
        if round_no > 1:
            print(f"[answers] retry round {round_no} (after {waits} plan-wall wait(s))", flush=True)
        aborted = _answer_phase_once(
            client=client,
            cfg=cfg,
            questions=questions,
            answers_path=answers_path,
            concurrency=concurrency,
        )
        if not aborted:
            if waits:
                print(f"[answers] plan recovered after {waits} wait(s); no questions left", flush=True)
            return False
        if wall_max_waits > 0 and waits >= wall_max_waits:
            print(
                f"[answers] giving up after {waits} plan-wall wait(s) (--wall-max-waits {wall_max_waits}); {answers_path} holds every finished row",
                flush=True,
            )
            return True
        waits += 1
        _wait_for_plan_refill(label="answers", wait_sec=wall_wait_sec, round_no=round_no + 1, max_waits=wall_max_waits)


def _answer_phase_once(
    *,
    client: JsonHttpClient,
    cfg: dict[str, Any],
    questions: list[dict[str, Any]],
    answers_path: Path,
    concurrency: int = 1,
) -> bool:
    """One dispatch round. Returns True when the quota breaker saw a wall."""
    stripped = _strip_damaged_rows(answers_path)
    if stripped:
        print(f"[answers] resume: stripped {stripped} damaged row(s) (transient ragflow_error); backup kept beside the file", flush=True)
    # A row counts as done when it is clean OR when its error is PERSISTENT -
    # the provider refused the input itself, and a retry would only reproduce
    # the refusal (see _is_persistent_ragflow_error).
    completed = {key for key, row in _read_jsonl_by_id(answers_path).items() if not (row.get("ragflow_error") or "").strip() or _is_persistent_ragflow_error(row.get("ragflow_error") or "")}
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

        # Whether the run ever SAW the documents that carry the answer. Without this
        # pair a retrieval miss is indistinguishable from a reasoning failure in the
        # archived row, and the two need opposite fixes: on the 16-question sample six
        # of the nine stalled runs had the whole expected set unserved (q283: zero of
        # nine docs across 123 retrievals, while nine audit rounds argued about the
        # wrong candidate). `served` is the run's own retrieval record, `cited` is what
        # the deliverable's own lines name - a doc can be served and still never used.
        if expected_doc_ids:
            # _unique_doc_ids returns a LIST, so the intersection needs a set on
            # both sides; this runs outside the try above, where a TypeError would
            # take the whole row down instead of degrading to a missing field.
            expected_set = set(expected_doc_ids)
            served = {str(x) for x in (row.get("retrieved_docids") or [])}
            answer_now = row.get("ragflow_answer") or ""
            row["gold_doc_served"] = sorted(expected_set & served)
            row["gold_doc_cited"] = sorted(doc for doc in expected_set if doc in answer_now)

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
                    f"spread over >= {QUOTA_BURST_WINDOW_SEC}s) - this round stops here; the question(s) "
                    f"without an answer are retried after the plan-wall wait.",
                    flush=True,
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
    wall_wait_sec: float = WALL_RETRY_WAIT_SEC,
    wall_max_waits: int = WALL_MAX_WAITS,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Judge every row without a verdict, riding out an exhausted plan in process.

    Same retry contract as run_answer_phase: a plan wall parks the phase for
    wall_wait_sec and then re-enters it, which re-judges exactly the rows still
    missing a verdict (a verdict that is ITSELF a backend error does not count).
    Unbounded unless wall_max_waits > 0. Returns (judgements, gave_up), where
    gave_up is True only when a bounded wait budget was exhausted.
    """
    round_no = 0
    waits = 0
    judgements: dict[str, dict[str, Any]] = {}
    while True:
        round_no += 1
        if round_no > 1:
            print(f"[judge] retry round {round_no} (after {waits} plan-wall wait(s))", flush=True)
        judgements, aborted = _judge_phase_once(
            client=client,
            cfg=cfg,
            answers_path=answers_path,
            leaderboard_path=leaderboard_path,
            concurrency=concurrency,
        )
        if not aborted:
            if waits:
                print(f"[judge] plan recovered after {waits} wait(s); no row left unjudged", flush=True)
            return judgements, False
        if wall_max_waits > 0 and waits >= wall_max_waits:
            print(
                f"[judge] giving up after {waits} plan-wall wait(s) (--wall-max-waits {wall_max_waits}); {leaderboard_path} holds every stored verdict",
                flush=True,
            )
            return judgements, True
        waits += 1
        _wait_for_plan_refill(label="judge", wait_sec=wall_wait_sec, round_no=round_no + 1, max_waits=wall_max_waits)


def _judge_phase_once(
    *,
    client: JsonHttpClient,
    cfg: dict[str, Any],
    answers_path: Path,
    leaderboard_path: Path,
    concurrency: int = 1,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """One judging round. Returns (judgements, wall_seen).

    Judgements live in leaderboard.json's per_query_judgements extension,
    keyed by run key - that file is both the submission output and the resume
    state, so there is no separate judged artefact.
    """
    if not answers_path.exists():
        raise FileNotFoundError(f"Missing answers file: {answers_path}")

    rows = _dedupe_last(_read_jsonl(answers_path))
    stripped = _strip_damaged_judgements(leaderboard_path, rows)
    if stripped:
        print(f"[judge] resume: stripped {stripped} judgement(s) standing on damaged answer row(s); backup kept beside the file", flush=True)
    # Judging scope: the answer rows MINUS the rows still carrying a
    # ragflow_error (those are going to be re-run by the answer phase; judging
    # them now would only record "excluded_due_to_ragflow_error" and be thrown
    # away on the next round), MINUS the questions already judged from an
    # error-free row.
    judged_rows = [row for row in rows if not (row.get("ragflow_error") or "").strip()]
    # A judgement whose verdict text is a backend error (e.g. a provider quota
    # wall) is NOT "done": it must be re-judged on the next run, which
    # overwrites the failed record.
    completed = {key for key, record in _read_judgements(leaderboard_path).items() if not _is_backend_error_verdict(record)}

    todo = [row for row in judged_rows if _row_run_key(row) not in completed]
    pending = list(enumerate(todo, start=1))
    skipped = len(judged_rows) - len(pending)
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
    # Judging runs at the SAME parallelism as the answer phase (user decision,
    # 2026-09-30): with hundreds of rows a capped judge phase is the long pole
    # of the whole run.
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
                f"spread over >= {QUOTA_BURST_WINDOW_SEC}s) - this round stops here; the row(s) left "
                f"without a verdict are re-judged after the plan-wall wait.",
                flush=True,
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


def _strip_damaged_judgements(leaderboard_path: Path, rows: list[dict[str, Any]]) -> int:
    """Drop judgements standing on an answer row that carries a ragflow_error.

    The answer phase re-runs damaged rows, so a judgement recorded as
    "excluded_due_to_ragflow_error" (or any verdict stored while the row was
    damaged) is stale state: it must not survive a resume, or the re-run's
    fresh judgement has to fight an old one. A timestamped backup is written
    beside the file before the rewrite; a leaderboard with nothing to strip is
    left untouched. Returns the number of records removed.
    """
    if not leaderboard_path.exists():
        return 0
    try:
        leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[judge] could not read {leaderboard_path}: {exc}")
        return 0
    records = leaderboard.get("per_query_judgements") if isinstance(leaderboard, dict) else None
    if not isinstance(records, list):
        return 0
    damaged_keys: set[str] = set()
    for row in rows:
        if (row.get("ragflow_error") or "").strip():
            key = _row_run_key(row)
            if key is not None:
                damaged_keys.add(key)
    kept = [record for record in records if isinstance(record, dict) and str(record.get("run_key") or "") not in damaged_keys]
    removed = len(records) - len(kept)
    if removed == 0:
        return 0
    backup = leaderboard_path.with_name(f"{leaderboard_path.name}.bak_strip_{time.strftime('%m%d_%H%M%S')}")
    backup.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    leaderboard["per_query_judgements"] = kept
    _write_json(leaderboard_path, leaderboard)
    return removed


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
# --- Structure metrics (the run's state, kept in markdown) -------------------
#
# The deliverable's Candidate Matrix IS the run's state: one block per
# sub-question, each with the answer TYPE it declared (`slot:`), the operation
# it ran (`Op:`), the variables it bound and consumed (`Binds:` / `From:`), the
# numbered constraints it addresses, one line per candidate
# (Searched / Tested / Eliminated / Retained), the value it derived, and — when
# a block ends without a filler — the failure it classified. Those shapes are
# countable, and counting them answers questions the accuracy figure cannot:
# whether the agent committed to an answer type BEFORE searching, whether every
# hop consumed a bound input, whether a failure was classified at all, and
# whether a run stopped early because it had found the document or because it
# gave up.
#
# The counts are reported, never scored. A field the prompt asks for and a run
# does not write is a fact about that run, and the number that matters is the
# PAIR (adoption, accuracy): a 100%-adopted schema tells you the format landed,
# not that the answers are right — measured 2026-09-23, `## Candidate Matrix`
# appeared in 24/24 deliveries while accuracy was 4/24.
STRUCTURE_FIELDS: tuple[str, ...] = (
    "blocks",
    "titles_anchored",
    "titles_use_vars",
    "slot",
    "kind",
    "op",
    "target_binds",
    "binds",
    "binds_unique",
    "from_lines",
    "constraints",
    "satisfied",
    # Replaces `depends_on`: the field is retired (it stated a DERIVED edge in a
    # second currency and no consumer ever read it), and the chain now rides on
    # the block pointer each `From:` entry carries.
    "from_pointers",
    "searched",
    "tested",
    "eliminated",
    "retained",
    "derived",
    "failure",
    "reasoning_chain",
    "answer_label",
)

# DELIVERY_TOTALS are the per-question check counters summed over the run. They
# are DIAGNOSTIC, never gate material: a count that fires on most rows is not a
# defect a pipeline can act on (unrecordedLocateTools was kept out of the gate for
# firing on 69% of 703 rows), so these exist to measure prevalence first.
DELIVERY_TOTALS: tuple[str, ...] = (
    "from_entries",
    "from_pointers",
    "from_entries_without_pointer",
    "from_pointer_mismatch",
    "from_unbound",
    "inlined_candidate",
    "depends_on_written",
    "searched_lines",
    "constraints_defined",
    "constraints_duplicated",
    "blocks_without_constraint",
    "blocks_without_searched",
    "blocks_with_by_name_query",
    "pivot_missing_blocks",
    "single_term_searches",
    "blocks_without_constraint",
    "constraint_number_reused",
    "blocks_parsed",
    "blocks_total",
    "constraint_number_reused",
    "decomposition_detached_blocks",
    "decomposition_orphan_roots",
    "decomposition_from_none_after_first",
    "decomposition_prose_block_refs",
    "leaked_docnames",
    "leaked_values",
    "example_value_shipped",
)

# EARLY_STOP_DOCS is the breadth below which a run counts as having stopped
# early. It is a SYMPTOM threshold, not a quality bar: a run that found the right
# document stops early, and a run that found nothing also stops early, so the
# number is only readable beside accuracy (see the diagnostics note).
EARLY_STOP_DOCS = 10

# _DOC_TEXT_LIMIT caps how much of one corpus document is read for the
# intermediate-node grounding check. The union of a run's retrieved documents
# can run to megabytes per question, and a value's distinctive tokens are almost
# always in the document's opening half; the cap keeps a full 830-question run
# from spending its time in file I/O.
_DOC_TEXT_LIMIT = 400_000

# _BLOCK_HEAD_RE finds the one marker the schema fixes for a block boundary. Its
# whitespace runs are `[^\S\n]`, never `\s`: `\s*` would swallow the newline
# before the heading too, so the slice of a block would begin with blank lines and
# its title would read as empty.
_BLOCK_HEAD_RE = re.compile(r"(?m)^[^\S\n]*#{2,4}[^\S\n]*Sub-question[^\S\n]+(\d+)[^\S\n]*:")
# _FROM_ENTRY_RE reads one `From:` entry: its variable, and the block pointer the
# producer wrote after it (absent when the entry carries none).
_FROM_ENTRY_RE = re.compile(r"(\?[A-Za-z_][A-Za-z0-9_]*)\s*(?:\(block\s*(\d+)\))?")
# _CANDIDATE_LINE_RE finds the lines that name a candidate inside one block.
_CANDIDATE_LINE_RE = re.compile(r"(?m)^\s*-\s*(?:Tested|Eliminated|Retained):\s*(.+)$")
# _CLAIM_LINE_RE finds the lines that assert something ABOUT THE CORPUS in one block:
# each owes a recorded search (see the Searched-line requirement).
_CLAIM_LINE_RE = re.compile(r"^\s*-\s*(?:Tested|Eliminated|Retained|Failure):")
# Emphasis markers are TYPOGRAPHY, not structure: a measured delivery wrote
# `- **Op:** lookup` and `- **From:** none`, and the prefix matches below read every
# block as EMPTY — a whole matrix scored as zero adopted fields. Strip the emphasis
# before matching a prefix, never before judging what the text says.
_DEEMPHASIS_RE = re.compile(r"\*+")
# Only a VARIABLE crosses a block boundary: a title or a constraint that says "block 1"
# or "the above" writes an edge in prose, where nothing resolves it. Measured
# 2026-09-23: 1 occurrence, in the delivery whose two blocks had ZERO edges between them.
_PROSE_BLOCK_REF_RE = re.compile(r"(?i)\bblock\s*\d+\b|the above|previous sub-question|sub-question above")
_DEPENDS_TAIL_RE = re.compile(r"(?i)[—\s-]+\s*depends_on\s*:.*$")
# _PATTERN_RE pulls the query text out of one recorded `Searched:` payload, so a short
# single-term query can be told from a multi-clue paraphrase.
_PATTERN_RE = re.compile(r'[("`]([^"`)]+)')


def _block_headers(text: str) -> list[dict[str, Any]]:
    """Split a deliverable into blocks and read each block's own header lines.

    Blocks are delimited by their `### Sub-question N:` headings; everything
    else is read by line prefix. A block whose header cannot be read keeps its
    missing parts empty instead of being dropped, so the caller can report
    COVERAGE rather than silently scoring "unparsed" as "clean" — the failure
    mode that lets a metric look green over text nothing actually read.
    """
    marks = list(_BLOCK_HEAD_RE.finditer(text))
    blocks: list[dict[str, Any]] = []
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        body = text[mark.start() : end]
        lines = body.splitlines()
        op = ""
        binds = ""
        has_from_line = False
        asserts_corpus = False
        searched: list[str] = []
        constraints: list[str] = []
        constraint_defs: list[tuple[str, str]] = []
        constraint_texts: list[str] = []
        from_entries: list[tuple[str, int | None]] = []
        for line in lines[1:]:
            stripped = _DEEMPHASIS_RE.sub("", line).strip()
            if not op and stripped.startswith("- Op:"):
                op = stripped[len("- Op:") :].strip()
            elif not binds and stripped.startswith("- Binds:"):
                binds = stripped[len("- Binds:") :].strip()
            elif stripped.startswith("- From:"):
                has_from_line = True
                for name, pointer in _FROM_ENTRY_RE.findall(stripped[len("- From:") :]):
                    from_entries.append((name, int(pointer) if pointer else None))
            elif stripped.startswith("- Constraints:"):
                # Never reuse `body` here: it holds the WHOLE block and is read further
                # down for the candidate heads, so shadowing it emptied `candidates` and
                # silently zeroed every metric built on them (by-name queries, pivot
                # misses, inlining) — measured over six runs the moment it was noticed.
                constraint_line = _DEPENDS_TAIL_RE.sub("", stripped)
                constraint_defs = [(cid, re.sub(r"\s+", " ", claim).strip().lower()) for cid, claim in re.findall(r"\b(c\d+)\s*=\s*([^;]*)", constraint_line)]
                constraints = [cid for cid, _ in constraint_defs]
                constraint_texts.append(constraint_line)
            elif stripped.startswith("- Searched:"):
                searched.append(stripped[len("- Searched:") :].strip())
            elif _CLAIM_LINE_RE.match(stripped):
                asserts_corpus = True
        candidates = [head for head in (_line_head(match) for match in _CANDIDATE_LINE_RE.findall(_DEEMPHASIS_RE.sub("", body))) if head]
        blocks.append(
            {
                "number": int(mark.group(1)),
                "title": _DEEMPHASIS_RE.sub("", lines[0]).strip() if lines else "",
                "op": op,
                "binds": binds,
                "has_from_line": has_from_line,
                "from_entries": from_entries,
                "candidates": candidates,
                "searched": searched,
                "constraints": constraints,
                "constraint_defs": constraint_defs,
                "constraint_texts": constraint_texts,
                "asserts_corpus": asserts_corpus,
            }
        )
    return blocks


def _delivery_checks(text: str, question: str, signatures: dict[str, Any] | None = None) -> dict[str, Any]:
    """Counts for the delivery questions that CAN be answered mechanically.

    Each is membership, not phrasing: a pointer either names the block that
    binds the variable or it does not, and a candidate name is either inside the
    block's own title or it is not. That is what makes these countable exactly —
    and they are still counted HERE, not enforced in the gate. Two reasons: a
    gate opinion costs a whole repair turn, and the prevalence of each defect is
    not measured yet (unrecordedLocateTools stayed out of the gate for firing on
    69% of 703 rows). Read `blocks_parsed` beside every count: a count taken over
    an unparsed block is not evidence of compliance.
    """
    blocks = _block_headers(text)
    bindings: dict[str, int] = {}
    for block in blocks:
        name = block["binds"].strip().lower()
        if name.startswith("?") and name not in bindings:
            bindings[name] = block["number"]
    # The decomposition is ONE chain: union the blocks each edge joins, then read whether
    # every block belongs to the component that carries `?answer`. Measured 2026-09-23:
    # 2 of 18 archived deliveries had detached blocks, and those are exactly the two runs
    # whose audit spent every round on item opinions while the shape stayed broken.
    union = {block["number"]: block["number"] for block in blocks}

    def union_root(x: int) -> int:
        while union[x] != x:
            union[x] = union[union[x]]
            x = union[x]
        return x

    pointers = 0
    without_pointer = 0
    mismatch = 0
    unbound = 0
    inlined = 0
    parsed = 0
    searched_lines = 0
    without_searched = 0
    with_by_name = 0
    pivot_missing = 0
    single_term = 0
    without_constraint = 0
    from_none_after_first = 0
    prose_refs = 0
    defined: dict[str, int] = {}
    definitions: dict[str, set[str]] = {}
    question_lower = (question or "").lower()
    for block in blocks:
        if block["op"] and block["binds"] and block["has_from_line"] and re.search(r"slot:\s*\S", block["title"]):
            parsed += 1
        if block["number"] > 1 and not block["from_entries"]:
            # `From` names what a block consumes, and the contract allows `none` in the
            # FIRST block only: a later block that consumes nothing starts a second,
            # disconnected question, or restates one already made. Measured 2026-09-23:
            # 10 such blocks across 18 archived deliveries.
            from_none_after_first += 1
        # Only a VARIABLE crosses a block boundary: a title or a constraint that says
        # "block 1" / "the above" states an edge in prose, where nothing resolves it.
        # Measured 2026-09-23: 1 occurrence, in the delivery whose two blocks had ZERO
        # edges between them.
        prose_refs += len(_PROSE_BLOCK_REF_RE.findall(block["title"]))
        prose_refs += sum(len(_PROSE_BLOCK_REF_RE.findall(line)) for line in block.get("constraint_texts") or [])
        for name, pointer in block["from_entries"]:
            producer = bindings.get(name.lower())
            if producer is not None:
                union[union_root(block["number"])] = union_root(producer)
            if pointer is None:
                without_pointer += 1
            else:
                pointers += 1
                if producer is None:
                    unbound += 1
                elif producer != pointer:
                    mismatch += 1
        for cid, ctext in block.get("constraint_defs") or []:
            defined[cid] = defined.get(cid, 0) + 1
            definitions.setdefault(cid, set()).add(ctext)
        if not block["constraints"]:
            without_constraint += 1
        searched_lines += len(block["searched"])
        if block["asserts_corpus"] and not block["searched"]:
            without_searched += 1
        recorded = " || ".join(block["searched"]).lower()
        keys = [key for candidate in block["candidates"] for key in _candidate_keys(candidate)]
        if any(key in recorded for key in keys):
            with_by_name += 1
        elif keys:
            # A block that tests or retains a candidate owes a query on that
            # candidate's own name (the by-name query strategy in "How to resolve a sub-question");
            # with candidates and no by-name query
            # it verified none of them.
            pivot_missing += 1
        for raw in block["searched"]:
            match = _PATTERN_RE.search(raw)
            if match and 0 < len(re.findall(r"[A-Za-z0-9']+", match.group(1))) <= 3:
                single_term += 1
        for candidate in block["candidates"]:
            lowered = candidate.lower()
            # A name the QUESTION itself supplies is not an inlined finding:
            # "Confirm the two individuals share the name Kevin Anderson" names
            # the question's own anchor, while "first name (Barbara)" names
            # something the search returned.
            if len(lowered) >= 3 and lowered in block["title"].lower() and lowered not in question_lower:
                inlined += 1
    answer_block = bindings.get("?answer")
    detached = 0
    if answer_block is not None:
        main = union_root(answer_block)
        detached = sum(1 for block in blocks if union_root(block["number"]) != main)
    # A root (`From: none`) is lawful: the flat shape's naming block is one, and a
    # parallel plan starts from SEVERAL. What is not lawful is a root whose variable no
    # block ever consumes - that is a second, disconnected question. `?answer` is exempt
    # because a chain-shaped plan legitimately ends on it. Measured 2026-09-23 over 18
    # archived deliveries: 4 lawful roots after the first block against 6 orphans, and
    # two deliveries ran connected graphs with more than one root.
    consumed_names = {name.lower() for block in blocks for name, _ in block["from_entries"]}
    orphan_roots = sum(1 for block in blocks if not block["from_entries"] and block["binds"] and block["binds"].strip().lower() not in consumed_names and block["binds"].strip().lower() != "?answer")
    return {
        "decomposition_detached_blocks": detached,
        "decomposition_orphan_roots": orphan_roots,
        "decomposition_from_none_after_first": from_none_after_first,
        "decomposition_prose_block_refs": prose_refs,
        "from_pointers": pointers,
        "from_entries": pointers + without_pointer,
        "from_entries_without_pointer": without_pointer,
        "from_pointer_mismatch": mismatch,
        "from_unbound": unbound,
        "inlined_candidate": inlined,
        "searched_lines": searched_lines,
        "constraints_defined": sum(defined.values()),
        "constraints_duplicated": sum(1 for count in defined.values() if count > 1),
        # A number defined twice with the SAME text is a duplicate; with DIFFERENT text
        # it is a restarted numbering, and every `satisfied: [...]` plus the whole
        # unestablished-set aggregation is read by number, so they all stop meaning
        # anything. Measured 2026-09-23: a passing delivery had `c1` mean five
        # different claims in its five blocks.
        "constraint_number_reused": sum(1 for texts in definitions.values() if len(texts) > 1),
        "blocks_without_constraint": without_constraint,
        "blocks_without_searched": without_searched,
        "blocks_with_by_name_query": with_by_name,
        "pivot_missing_blocks": pivot_missing,
        "single_term_searches": single_term,
        "blocks_parsed": parsed,
        "blocks_total": len(blocks),
        **_leakage_counts(text, signatures),
    }


def _candidate_keys(head: str) -> list[str]:
    """Keys under which one candidate counts as searched by name.

    A candidate head can be a bare name ("Kevin Anderson (tennis)") or a whole
    clause ("Both mothers named Barbara (c2 established)"), so testing the head
    verbatim against a query misses almost every real by-name search. The head is
    therefore reduced to its own tokens: the whole string when it is short, plus
    every two-word run inside it — long enough to be distinctive, short enough to
    survive the parentheses, qualifiers and outcome a `Searched:` line carries.
    """
    words = re.findall(r"[A-Za-z0-9'’-]+", head.lower())
    keys = [" ".join(words)] if 2 <= len(words) <= 4 else []
    keys += [" ".join(words[index : index + 2]) for index in range(len(words) - 1)]
    return [key for key in keys if len(key) >= 6]


PROMPT_PATH = Path(__file__).resolve().parent.parent / "conf" / "agentic_rag.yaml"
# _EXAMPLE_VALUE_RE pulls the value a prompt EXAMPLE writes into a field — the
# `Tested:`/`Retained:`/`Derived:`/`Evidence:` heads. Whatever a prompt prints as a
# worked example is in the model's context, and a name or a document id printed
# there can come back out as an answer (measured 2026-09-23: a delivery shipped the
# prompt example's own hospital name and cited the document the example printed).
_EXAMPLE_VALUE_RE = re.compile(r"(?:Tested|Retained|Derived|Evidence):\s*([^—\n`/(]{4,80})")
_DOCNAME_RE = re.compile(r"\b\d{3,}\.md\b")
LEAKAGE_SIGNATURES: dict[str, Any] | None = None


def _prompt_signatures(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """The names and document ids the PROMPT itself prints in its examples.

    A canary, not a judgement: it exists so that re-introducing a real entity into
    an example cannot pass unnoticed. De-identified examples yield an empty set,
    which is the healthy reading — report the signature COUNT beside every zero so
    "nothing leaked" can be told from "nothing was looked for".
    """
    global LEAKAGE_SIGNATURES
    if path is None and LEAKAGE_SIGNATURES is not None:
        return LEAKAGE_SIGNATURES
    docnames: set[str] = set()
    values: set[str] = set()
    try:
        raw = Path(path or PROMPT_PATH).read_text(encoding="utf-8")
    except OSError:
        result = {"docnames": docnames, "values": values, "loaded": False}
        LEAKAGE_SIGNATURES = result if path is None else LEAKAGE_SIGNATURES
        return result
    docnames.update(_DOCNAME_RE.findall(raw))
    for match in _EXAMPLE_VALUE_RE.findall(raw):
        value = match.strip().strip('"').strip()
        # Only names can leak: a lowercase phrase ("the institution slot") is schema
        # vocabulary a delivery is SUPPOSED to repeat, and flagging it would drown
        # the canary in false positives.
        if len(value) >= 6 and not value.startswith("<") and len(re.findall(r"[A-Z]", value)) >= 2:
            values.add(value)
    result = {"docnames": docnames, "values": values, "loaded": True}
    if path is None:
        LEAKAGE_SIGNATURES = result
    return result


def _leakage_counts(text: str, signatures: dict[str, Any] | None) -> dict[str, Any]:
    """Whether the deliverable reused anything the PROMPT prints in its examples."""
    signatures = signatures or _prompt_signatures()
    docnames = signatures.get("docnames") or set()
    values = signatures.get("values") or set()
    answer = "\n".join(re.findall(r"(?im)^.*(?:Final|Guessed)\s+Answer.*$", text))
    lowered = text.lower()
    leaked_values = 0
    shipped = 0
    for value in values:
        needle = value.lower()
        count = lowered.count(needle)
        if count:
            leaked_values += count
            if needle in answer.lower():
                shipped = 1
    return {
        "leaked_docnames": sum(1 for docname in docnames if docname.lower() in lowered),
        "leaked_values": leaked_values,
        "example_value_shipped": shipped,
    }


def structure_metrics(answer: str, question: str = "", signatures: dict[str, Any] | None = None) -> dict[str, Any]:
    """Count the structure a deliverable DECLARED, from its own text.

    Every counter is a regex over the markdown the producer shipped, so the
    result is reproducible from `answers.jsonl` alone and comparable across
    runs. Nothing here judges whether the structure is true — that is the answer
    auditor's job — and nothing here reads the corpus except `values`, which is
    returned for the grounding check the caller runs.
    """
    text = re.sub(r"<think>.*?</think>", "", answer or "", flags=re.DOTALL)
    # Field prefixes are matched on a de-emphasised copy (see _DEEMPHASIS_RE): the
    # bolded-header delivery of 2026-09-23 read as zero adopted fields otherwise.
    plain = _DEEMPHASIS_RE.sub("", text)

    def count(pattern: str) -> int:
        return len(re.findall(pattern, plain, flags=re.MULTILINE))

    titles = re.findall(r"^\s*#{2,4}\s*Sub-question\s+\d+\s*:(.*)$", text, flags=re.MULTILINE)
    metrics: dict[str, Any] = {
        "blocks": len(titles),
        "slot": sum(1 for title in titles if re.search(r"slot:\s*[A-Za-z_\[\]]", title)),
        # Sub-question titles must carry their own anchors (proper noun, number
        # or distinctive term) so a reviewer can read them without re-deriving
        # the chain; a bare slot tag ("which hospital") names none.
        "titles_anchored": sum(1 for title in titles if _title_names_anchor(title)),
        # Titles that state the plan with a variable (?person) instead of inlining
        # another block's current candidate: the decomposition stays reproducible.
        "titles_use_vars": sum(1 for title in titles if "?" in title),
        "kind": sum(1 for title in titles if re.search(r"kind:\s*\S", title)),
        "op": count(r"^\s*-\s*Op:\s*[A-Za-z_]"),
        # Blocks binding the reserved answer variable: the answer path's entry
        # points, and the one header fact the label rules now key on.
        "target_binds": count(r"^\s*-\s*Binds:\s*\?answer\b"),
        "binds": count(r"^\s*-\s*Binds:\s*"),
        # Variable names are unique question-wide, so a run that binds the same
        # name twice is a shape defect the header rules name explicitly; the
        # counter is here so the rule's landing is measurable offline.
        "binds_unique": (
            1
            if len({name.lower() for name in re.findall(r"^\s*-\s*Binds:\s*([^\s(]+)", plain, flags=re.MULTILINE)}) == len(re.findall(r"^\s*-\s*Binds:\s*([^\s(]+)", plain, flags=re.MULTILINE))
            else 0
        ),
        "from_lines": count(r"^\s*-\s*From:\s*"),
        "constraints": count(r"^\s*-\s*Constraints:\s*"),
        "satisfied": count(r"satisfied:\s*\["),
        # Retired field: a nonzero count is the OLD contract still being written,
        # so it is counted as a violation rather than as an adoption.
        "depends_on_written": count(r"depends_on\s*:"),
        "searched": count(r"^\s*-\s*Searched:\s*"),
        "tested": count(r"^\s*-\s*Tested:\s*"),
        "eliminated": count(r"^\s*-\s*Eliminated:\s*"),
        "retained": count(r"^\s*-\s*Retained:\s*"),
        "derived": count(r"^\s*-\s*Derived:\s*"),
        "failure": count(r"^\s*-\s*Failure:\s*"),
        "reasoning_chain": 1 if re.search(r"^\s*#{1,4}\s*Reasoning Chain\b", text, flags=re.MULTILINE) else 0,
        "answer_label": 1 if re.search(r"(?i)\b(?:Final|Guessed)\s+Answer\b", text) else 0,
    }
    metrics["failure_types"] = sorted({match.lower() for match in re.findall(r"^\s*-\s*Failure:\s*([A-Za-z_-]+)", text, flags=re.MULTILINE)})
    metrics["values"] = _intermediate_values(text)
    metrics.update(_delivery_checks(text, question, signatures))
    return metrics


def _title_names_anchor(title: str) -> bool:
    """Whether a sub-question title names an anchor of its own.

    The title must state the sub-question in resolved prose a reviewer can check
    without re-deriving the chain, so it needs at least one anchor that is not
    optional wording: a number, a capitalised name inside the sentence, or a
    distinctive term. A bare slot tag ("which hospital", "the person") has none.
    The check is deliberately loose — it counts anchors, it never grades style —
    and the `slot:`/`kind:` suffix is stripped before it runs.
    """
    # The heading (### Sub-question 2:) and the slot/kind suffix are metadata,
    # never anchors — leaving the sub-question NUMBER in would make every title
    # look anchored, since it carries a digit.
    text = re.sub(r"^\s*#{2,4}\s*Sub-question\s+[0-9]+\s*:\s*", "", title)
    # The separator in front of the tag is punctuation, and the producer picks
    # its glyph freely: measured 2026-09-23, titles arrived with `- slot:` 8
    # times and `— slot:` 6 times in a single 3-question run. The earlier
    # hyphen-only strip left the whole `— slot: … — kind: …` tail in the text
    # for em-dash titles, which is a silent glyph dependence in a metric; that
    # it did NOT change `titles_anchored` on those three questions was verified
    # by recomputing both ways, so this is robustness, not a measured fix.
    text = re.sub(r"\s*[-–—]?\s*(?:slot|kind)\s*:.*$", "", text).strip()
    if re.search(r"[0-9]", text):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z''-]+", text)
    if len(words) < 3:
        return False
    return any(word[:1].isupper() for word in words[1:])


def _line_head(line: str) -> str:
    """The candidate name at the head of a matrix line, before its fields."""
    cut = len(line)
    for separator in ("—", " - ", "satisfied:", "(doc:", "doc:", "(source:"):
        index = line.lower().find(separator.lower())
        if 0 <= index < cut:
            cut = index
    return line[:cut].strip().strip('`*" ')


def _intermediate_values(text: str) -> list[str]:
    """The entity values a run committed to mid-chain.

    Retained candidates and the first clause of every Derived line, deduplicated:
    these are the intermediate nodes an accuracy check can look for in the
    corpus, where the final answer alone only says whether the last hop landed.
    """
    values: list[str] = []
    for line in re.findall(r"^\s*-\s*Retained:\s*(.+)$", text, flags=re.MULTILINE):
        candidate = _line_head(line)
        if candidate and not re.fullmatch(r"(?i)none", candidate):
            values.append(candidate)
    for line in re.findall(r"^\s*-\s*Derived:\s*(.+)$", text, flags=re.MULTILINE):
        head = line.split(";")[0].strip()
        if 3 <= len(head) <= 60:
            values.append(head)
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _value_tokens(value: str) -> list[str]:
    """The distinctive tokens of a value: what a corpus has to carry for the
    value to have come from it, ignoring numbers (dates and counts match
    everywhere) and short grammatical filler."""
    return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) >= 4 and not token.isdigit()]


def _doc_text(doc_id: str, corpus_dir: Path, cache: dict[str, str]) -> str:
    """One corpus document's lower-cased text, cached across questions."""
    key = str(doc_id)
    if key in cache:
        return cache[key]
    name = key if key.endswith(".md") else f"{key}.md"
    text = ""
    try:
        with (corpus_dir / name).open("r", encoding="utf-8", errors="replace") as handle:
            text = handle.read(_DOC_TEXT_LIMIT).lower()
    except OSError:
        text = ""
    cache[key] = text
    return text


def _ground_values(
    values: list[str],
    doc_ids: list[str],
    corpus_dir: Path | None,
    cache: dict[str, str],
) -> tuple[int, int]:
    """Count how many committed values occur in the documents the run retrieved.

    Returns (grounded, checked). A value counts as checked only when it has at
    least one distinctive token, and as grounded when ALL of those tokens appear
    somewhere in the union of the retrieved documents — a proxy for "this
    intermediate node came from the corpus the run actually read", which is the
    closest offline reading of the intermediate-node accuracy WebShaper's
    evaluation section asks for.
    """
    if corpus_dir is None or not values or not doc_ids:
        return 0, 0
    union = "".join(_doc_text(doc_id, corpus_dir, cache) for doc_id in doc_ids)
    if not union:
        return 0, 0
    grounded = 0
    checked = 0
    for value in values:
        tokens = _value_tokens(value)
        if not tokens:
            continue
        checked += 1
        if all(token in union for token in tokens):
            grounded += 1
    return grounded, checked


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
        incorrect - the figure is the run's end-to-end success rate, not the
        success rate among questions that produced a usable answer.
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
    # Structure / behaviour accounting (see STRUCTURE_FIELDS above): kept out of
    # per_query_metrics so that key stays byte-compatible with the leaderboard's
    # parser, exactly as per_query_usage is.
    per_query_structure: list[dict[str, Any]] = []
    structure_seen: dict[str, int] = defaultdict(int)
    failure_type_counts: dict[str, int] = defaultdict(int)
    early_stop_hits = 0
    early_stop_rows = 0
    intermediate_grounded = 0
    intermediate_checked = 0
    corpus_path = (cfg.get("dataset") or {}).get("corpus_path")
    corpus_dir = Path(str(corpus_path)) if corpus_path else None
    doc_text_cache: dict[str, str] = {}
    # Delivery checks: totals over the run, plus a per-question count of the rows
    # that tripped each one (the prevalence figure the gate decision needs).
    delivery_totals: dict[str, int] = defaultdict(int)
    inline_rows = 0
    pointer_rows = 0
    retired_rows = 0
    pivot_rows = 0
    unrecorded_rows = 0
    partition_rows = 0
    reuse_rows = 0
    leak_rows = 0
    decomposition_rows = 0
    prompt_signatures = _prompt_signatures()
    prompt_signature_total = len(prompt_signatures["docnames"]) + len(prompt_signatures["values"])

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
        # per question. Rows the judge never reached are omitted -
        # _diagnostics.judged counts them.
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

        # Structure: what the deliverable DECLARED, counted rather than judged.
        structure = structure_metrics(row.get("ragflow_answer") or "", row.get("question") or "", prompt_signatures)
        committed_values = structure.pop("values", [])
        failure_types = structure.pop("failure_types", [])
        per_query_structure.append({"query_id": query_id, "run_key": _row_run_key(row), **structure})
        for field in STRUCTURE_FIELDS:
            if structure.get(field):
                structure_seen[field] += 1
        for failure_type in failure_types:
            failure_type_counts[failure_type] += 1
        # Delivery checks: measured here, never enforced here — see DELIVERY_TOTALS.
        if structure.get("inlined_candidate"):
            inline_rows += 1
        if structure.get("from_entries_without_pointer") or structure.get("from_pointer_mismatch"):
            pointer_rows += 1
        if structure.get("depends_on_written"):
            retired_rows += 1
        if structure.get("pivot_missing_blocks"):
            pivot_rows += 1
        if structure.get("blocks_without_searched"):
            unrecorded_rows += 1
        if structure.get("constraints_duplicated"):
            partition_rows += 1
        if structure.get("constraint_number_reused"):
            reuse_rows += 1
        if structure.get("leaked_docnames") or structure.get("leaked_values") or structure.get("example_value_shipped"):
            leak_rows += 1
        if structure.get("decomposition_detached_blocks") or structure.get("decomposition_orphan_roots") or structure.get("decomposition_prose_block_refs"):
            decomposition_rows += 1
        for key in DELIVERY_TOTALS:
            delivery_totals[key] += int(structure.get(key) or 0)

        # Early stop: a run that concluded inside a handful of retrieved
        # documents. Symptom, not score - see EARLY_STOP_DOCS.
        retrieved_now = _unique_doc_ids(row.get("retrieved_docids") or [])
        if row.get("retrieved_docids") is not None:
            early_stop_rows += 1
            if len(retrieved_now) <= EARLY_STOP_DOCS:
                early_stop_hits += 1

        # Intermediate nodes: did the entities the deliverable committed to
        # mid-chain come from the documents it actually retrieved?
        grounded, checked = _ground_values(committed_values, retrieved_now, corpus_dir, doc_text_cache)
        intermediate_grounded += grounded
        intermediate_checked += checked

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

    structure_adoption = {field: round(structure_seen[field] / total * 100.0, 1) if total else 0.0 for field in STRUCTURE_FIELDS}
    if corpus_dir is None:
        intermediate_note = "not computed: dataset.corpus_path is not configured, so no corpus text can back the check."
    else:
        intermediate_note = (
            "A committed value counts as grounded when every distinctive token of it (>=4 chars) "
            "occurs in the union of the documents the run retrieved: the offline reading of whether the "
            "mid-chain entities the deliverable committed to came from the corpus it read. Values with no "
            "distinctive token are skipped, and a value that is grounded may still be the wrong one."
        )
    early_stop_note = (
        "A run that concludes inside a handful of retrieved documents is a SYMPTOM, not a score: finding "
        "the document and giving up look identical here, so read this beside Accuracy and beside "
        "structure_adoption_percent's 'failure' field."
    )
    delivery_checks = {
        "rows": total,
        "inlined_candidate": {
            "rows": inline_rows,
            "occurrences": delivery_totals["inlined_candidate"],
            "note": "A candidate name written into a block's own title (a value the search returned, not one the "
            "question supplies) freezes a finding into the plan, so the same question decomposes differently "
            "on another run. Membership, not phrasing: the name comes from the block's own Tested/Eliminated/"
            "Retained lines.",
        },
        "from_pointer": {
            "entries": delivery_totals["from_entries"],
            "with_pointer": delivery_totals["from_pointers"],
            "without_pointer": delivery_totals["from_entries_without_pointer"],
            "mismatch": delivery_totals["from_pointer_mismatch"],
            "unbound": delivery_totals["from_unbound"],
            "rows_with_a_defect": pointer_rows,
            "note": "Each `From: ?x (block N)` should name the block that binds ?x — a DERIVED pointer, checkable "
            "against Binds. `without_pointer` counts entries that carry no pointer (the pre-2026-09-23 shape), "
            "`mismatch` counts pointers naming another block, `unbound` counts names no block binds.",
        },
        "depends_on_written": {
            "rows": retired_rows,
            "occurrences": delivery_totals["depends_on_written"],
            "note": "The field is RETIRED (the chain is stated by From); a nonzero count is the old contract still being written, not an adoption.",
        },
        "searched_record": {
            "lines": delivery_totals["searched_lines"],
            "blocks_without_searched": delivery_totals["blocks_without_searched"],
            "rows": unrecorded_rows,
            "note": "A block that asserts anything about the corpus (Tested/Eliminated/Retained/Failure) owes at "
            "least one recorded `Searched:` line. Measured 2026-09-23: a deliverable can retain six candidates "
            "while recording no search at all, and every search-discipline check reads those lines, so an "
            "unrecorded run is unauditable rather than clean.",
        },
        "pivot": {
            "blocks_with_by_name_query": delivery_totals["blocks_with_by_name_query"],
            "pivot_missing_blocks": delivery_totals["pivot_missing_blocks"],
            "rows_with_a_missing_pivot": pivot_rows,
            "single_term_searches": delivery_totals["single_term_searches"],
            "note": 'the by-name query strategy (in "How to resolve a sub-question"), counted from the deliverable\'s own `Searched` patterns against its own '
            "candidate names: a block with candidates and no by-name query verified none of them. "
            "`single_term_searches` is the crude proxy for the clue-anchor query (<=3 words in the pattern). "
            "On the 2026-09-23 runs every measurable FAILURE had zero by-name queries while every pass had at "
            "least one, and q875 retrieved 129 documents without touching its expected family.",
        },
        "decomposition_graph": {
            "rows_with_a_defect": decomposition_rows,
            "detached_blocks": delivery_totals["decomposition_detached_blocks"],
            "orphan_roots": delivery_totals["decomposition_orphan_roots"],
            "roots_after_the_first": delivery_totals["decomposition_from_none_after_first"],
            "prose_block_refs": delivery_totals["decomposition_prose_block_refs"],
            "note": "The decomposition must be ONE connected acyclic graph that carries `?answer`, and nothing more: "
            "sequence, fan-out, join and nesting all combine, and a root (`From: none`) is lawful - the flat shape's "
            "naming block is one and a parallel plan starts from several. The defects are the two that break the "
            "graph: an ORPHAN ROOT whose variable no block consumes (a second, disconnected question) and a block "
            "outside the `?answer` component; plus one that breaks resolution: an edge written in prose (`block 1`, "
            "`the above`) instead of the variable. Measured 2026-09-23 over 18 archived deliveries: 4 lawful roots "
            "after the first block against 6 orphans, 2 rows with detached blocks (5 of their 6 blocks in one), and "
            "1 prose reference - the detached rows being exactly the runs whose REQUIRED audit spent every round on "
            "item opinions while the shape stayed broken. Counted, not enforced: the auditor names these as "
            "`schema integrity` failures only once the prevalence and the false-positive rate are known.",
        },
        "constraint_partition": {
            "defined": delivery_totals["constraints_defined"],
            "duplicated": delivery_totals["constraints_duplicated"],
            "blocks_without_constraint": delivery_totals["blocks_without_constraint"],
            "rows_with_a_duplicate": partition_rows,
            "number_reused": delivery_totals["constraint_number_reused"],
            "rows_with_a_reused_number": reuse_rows,
            "note": "The numbered constraints form a PARTITION: each `c<k>` is defined by exactly one block, and a "
            "number two blocks define leaves the unsatisfied-set aggregation ambiguous, since it works by that "
            "number. Measured 2026-09-23 (with the retired dependency tails excluded, which is what inflation "
            "from the old field looked like): 26 of 186 definitions were duplicates.",
        },
        "prompt_leakage": {
            "signature_total": prompt_signature_total,
            "signatures_loaded": bool(prompt_signatures.get("loaded")),
            "rows": leak_rows,
            "leaked_docnames": delivery_totals["leaked_docnames"],
            "leaked_values": delivery_totals["leaked_values"],
            "example_value_shipped": delivery_totals["example_value_shipped"],
            "note": "A CANARY over the prompt's own examples, not a judgement: the names and document ids a "
            "prompt prints in a worked example sit in the model's context, and one measured delivery "
            "shipped the example's hospital name as its answer while citing the document the example "
            "printed. `signature_total` counts what was looked for — read it beside the zeros, since a "
            "de-identified prompt and an unloaded prompt both report none.",
        },
        "parse_coverage": {
            "blocks_parsed": delivery_totals["blocks_parsed"],
            "blocks_total": delivery_totals["blocks_total"],
            "percent": round(delivery_totals["blocks_parsed"] / delivery_totals["blocks_total"] * 100.0, 1) if delivery_totals["blocks_total"] else None,
            "note": "A block counts as parsed when its title carries `slot:` and it carries `Op:`, `Binds:` and "
            "`From:`. READ THIS BESIDE EVERY COUNT ABOVE: a count taken over an unparsed block is not "
            "evidence of compliance, and a metric that cannot report its own coverage turns 'unreadable' "
            "into 'clean'.",
        },
        "note": "Measured, never enforced: these three checks are membership, not phrasing, so the gate COULD carry "
        "them — but a gate opinion costs a whole repair turn on every run, and prevalence is not measured yet "
        "(unrecordedLocateTools stayed out of the gate for firing on 69% of 703 rows). Collect the prevalence "
        "here first; move a check into the gate only when its false-positive rate is ~zero, its coverage is "
        "high, and it fires rarely.",
    }

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
        # Extension, not part of the submission schema: the declared structure
        # of each deliverable (state in markdown), counted per question.
        "per_query_structure": per_query_structure,
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
            # Structure adoption: how many questions declared each field. Report
            # it with accuracy, never instead of it - a schema can be fully
            # adopted while the answers stay wrong.
            "structure_adoption_percent": structure_adoption,
            "failure_types": dict(sorted(failure_type_counts.items())) if failure_type_counts else {},
            "early_stop": {
                "threshold_docs": EARLY_STOP_DOCS,
                "rows": early_stop_rows,
                "rows_at_or_below": early_stop_hits,
                "percent": round(early_stop_hits / early_stop_rows * 100.0, 1) if early_stop_rows else None,
                "note": early_stop_note,
            },
            "intermediate_nodes": {
                "values_checked": intermediate_checked,
                "values_grounded": intermediate_grounded,
                "percent": round(intermediate_grounded / intermediate_checked * 100.0, 1) if intermediate_checked else None,
                "note": intermediate_note,
            },
            "delivery_checks": delivery_checks,
            "note": " ".join(
                [
                    "Accuracy counts every question of the run; failed and unjudged rows",
                    "count as incorrect.",
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


def _llm_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """One LLM call's cost: its input/output token split, the model that served
    it and when it started relative to the run."""
    prompt = _as_int(turn.get("prompt_tokens"))
    completion = _as_int(turn.get("completion_tokens"))
    total = _as_int(turn.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {
        "seq": _as_int(turn.get("seq")),
        "at_seconds": _as_float(turn.get("at_seconds")),
        "model": str(turn.get("model") or "") or None,
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
    }


def _gate_audit_rejections(row: dict[str, Any]) -> int | None:
    """How many deliverables the gate refused BEFORE an audit ran (ungrounded
    value, list-only name, value-less line, answer-less continuation). A run
    with rejections and no suspects never reached the auditor."""
    audit = row.get("gate_audit")
    if not isinstance(audit, dict):
        return None
    return _as_int(audit.get("rejections"))


_LOCATE_TOOLS = ("grep_chunks", "search_bm25_chunks", "search_semantic_chunks", "search_chunks")
_SEARCHED_LINE_RE = re.compile(r"(?im)Searched:\s*(grep_chunks|search_bm25_chunks|search_semantic_chunks|search_chunks)")


def _unrecorded_locate_tools(row: dict[str, Any]) -> list[str] | None:
    """Locate tools the run called but the Candidate Matrix never credits on a
    `Searched:` line.

    Reported, never scored: measured on 703 scored rows, 69% of runs omit at
    least one (grep_chunks in 362), so it is the norm rather than an anomaly and
    rejecting runs over it would reject two in three. It is recorded because the
    omission makes a call INVISIBLE - on #71 the one call that surfaced the
    answer's own document was an unrecorded search_semantic_chunks query, and
    nothing but the server log showed the pure-vector leg had done the work."""
    counts = row.get("tool_call_counts")
    if not isinstance(counts, dict):
        return None
    listed = {m.lower() for m in _SEARCHED_LINE_RE.findall(str(row.get("ragflow_answer") or ""))}
    missing = [t for t in _LOCATE_TOOLS if (_as_int(counts.get(t)) or 0) > 0 and t not in listed]
    return missing or None


def _gate_audit_verdicts(row: dict[str, Any]) -> list[str] | None:
    """An excerpt of every audit verdict, oldest first, in step with
    `audit_suspects`. The counts alone say a curve moved 5-3-1-0 but never WHAT
    was contested, so a failure had to be explained by reverse-engineering the
    shipped deliverable - which is how the #221 analysis twice inferred the
    wrong cause."""
    audit = row.get("gate_audit")
    if not isinstance(audit, dict):
        return None
    verdicts = audit.get("audit_verdicts")
    return [str(v) for v in verdicts] if isinstance(verdicts, list) else None


def _gate_audit_citation_groundings(row: dict[str, Any]) -> int | None:
    """Deliverables the gate refused because a candidate line's cited support was a
    BIBLIOGRAPHIC ENTRY: a citation names a work and states nothing about it, so it
    cannot tell two siblings apart. The mechanical half of the sibling discipline."""
    audit = row.get("gate_audit")
    if not isinstance(audit, dict):
        return None
    return _as_int(audit.get("citation_groundings"))


def _gate_audit_failures(row: dict[str, Any]) -> int | None:
    """How many audit passes the auditor could not complete (LLM timeout, tool
    outage). A run whose auditor never returned a verdict carries no suspects and
    no rejections; without this count its record read as "no audit was needed",
    and the deliverable's label was the only place that showed it."""
    audit = row.get("gate_audit")
    if not isinstance(audit, dict):
        return None
    return _as_int(audit.get("audit_failures"))


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
        # Per-turn token detail, oldest first (None when the backend did not
        # report it): the same total as above, broken down per LLM call.
        "llm_turns": row.get("llm_turns"),
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
        "audit_rejections": _gate_audit_rejections(row),
        "audit_failures": _gate_audit_failures(row),
        "citation_groundings": _gate_audit_citation_groundings(row),
        "audit_verdicts": _gate_audit_verdicts(row),
        # Locate tools used but never credited in the Candidate Matrix: an
        # omission here hides which leg actually did the work.
        "unrecorded_locate_tools": _unrecorded_locate_tools(row),
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
        # Per-LLM-call split (agentic runs on a backend that reports it): one
        # entry per call, so a question's cost can be attributed to the turns
        # that produced it instead of being read as one number.
        turns = usage.get("llm_turns")
        if isinstance(turns, list) and turns:
            stats["llm_turns"] = [_llm_turn(turn) for turn in turns if isinstance(turn, dict)]

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

    # And WHICH chunks those counts counted. A total says how much context the
    # reading cost; only the ids can answer whether a passage the deliverable
    # needed was ever in front of the model — the difference between a chunk
    # that was read and silently dropped and one that was never read, which is
    # the whole diagnosis for the evidence_in_hand failures. Recorded as raw
    # facts, unfiltered against what the run cited: drawing that difference is
    # the reader's job, and a threshold chosen before the data exists would
    # only bake in a guess.
    for key in ("deep_read_chunk_ids", "shallow_read_chunk_ids"):
        value = _unique_ids(source.get(key))
        if value:
            stats[key] = value

    gate_audit = source.get("gate_audit")
    if isinstance(gate_audit, dict):
        suspects = gate_audit.get("suspects")
        rejections = _as_int(gate_audit.get("rejections"))
        audit_failures = _as_int(gate_audit.get("audit_failures"))
        verdicts = gate_audit.get("audit_verdicts")
        citation_groundings = _as_int(gate_audit.get("citation_groundings"))
        # Keep the record when ANY signal is present: a gate that refused every
        # deliverable before an audit could run reports no suspects at all, and
        # one whose auditor never returned a verdict reports nothing but the
        # failure - dropping either hid exactly that state (q350/q784, and the
        # audit-outage case).
        if isinstance(suspects, list) or rejections is not None or audit_failures is not None or isinstance(verdicts, list) or citation_groundings is not None:
            stats["gate_audit"] = {
                "suspects": [_as_int(s) for s in suspects] if isinstance(suspects, list) else None,
                "passed": bool(gate_audit.get("passed")),
                "rejections": rejections,
                "audit_failures": audit_failures,
                # The mechanical sibling check: deliverables refused because a
                # candidate's cited support was a bibliographic entry.
                "citation_groundings": citation_groundings,
                # An excerpt of each round's verdict, in step with suspects:
                # counts show the curve, these show what was actually contested.
                "audit_verdicts": [str(v) for v in verdicts] if isinstance(verdicts, list) else None,
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


FINGERPRINT_NAME = "_prompt_fingerprint.json"

# Markers of an error that makes a retry pointless: the provider refused the
# INPUT itself (content policy), so the same question fails identically on
# every attempt. Measured case: q744 - five identical "input new_sensitive"
# failures across as many resumes before this distinction existed. Everything
# else (quota walls, connection aborts, timeouts, retry exhaustion with a
# recoverable cause) stays transient and is retried; an unknown error also
# defaults to transient - one wasted retry beats one silently abandoned
# question.
PERSISTENT_ERROR_MARKERS = (
    "new_sensitive",
    "sensitive content",
    "content_filter",
    "content filter",
)


def _is_persistent_ragflow_error(err: str) -> bool:
    text = (err or "").lower()
    return any(marker in text for marker in PERSISTENT_ERROR_MARKERS)


def _record_run_provenance(output_dir: Path) -> str | None:
    """Stamp the run's git commit into _prompt_fingerprint.json, once.

    The archived rows must answer "what code produced them" without log
    archaeology (the #46 regression review needed exactly that, and had to
    reconstruct it from launch scripts). A fingerprint that already carries a
    commit is left untouched: a resumed run must not overwrite the commit of
    the run that produced the rows. Returns the recorded commit, or None when
    git is unavailable or the commit was already recorded."""
    fingerprint = output_dir / FINGERPRINT_NAME
    try:
        existing = json.loads(fingerprint.read_text(encoding="utf-8")) if fingerprint.exists() else {}
    except (OSError, ValueError) as exc:
        print(f"[run] could not read {fingerprint.name}: {exc}; a fresh one will be written")
        existing = {}
    if not isinstance(existing, dict):
        existing = {}
    if existing.get("git_commit"):
        return str(existing["git_commit"])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=9", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        print(f"[run] git commit not recorded: {exc}")
        return None
    existing["git_commit"] = commit
    existing.setdefault("git_commit_recorded_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = fingerprint.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(fingerprint)
    print(f"[run] git commit recorded: {commit} -> {fingerprint.name}", flush=True)
    return commit


def _strip_damaged_rows(path: Path) -> int:
    """Drop rows a resume should re-run, keep the ones it must not.

    A row whose ragflow_error is TRANSIENT (quota wall, connection abort,
    timeout, retry exhaustion with a recoverable cause) is dropped: the
    resume re-runs it and appends a fresh row. A row carrying a PERSISTENT
    error - the provider refused the input itself (content policy, e.g.
    MiniMax's `input new_sensitive`), so every retry fails identically - is
    KEPT and treated as settled: re-running it would burn quota to reproduce
    the same refusal forever (q744: 5+ identical failures across resumes).
    Rows the parser cannot read are treated as damaged and dropped. A
    timestamped backup is written beside the file before the rewrite, and a
    file with nothing to strip is left untouched. Returns the number of rows
    removed."""
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return 0
    kept: list[str] = []
    damaged = 0
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            damaged += 1
            continue
        if not isinstance(row, dict):
            damaged += 1
            continue
        err = (row.get("ragflow_error") or "").strip()
        if err and not _is_persistent_ragflow_error(err):
            damaged += 1
            continue
        kept.append(line)
    if damaged == 0:
        return 0
    backup = path.with_name(f"{path.name}.bak_strip_{time.strftime('%m%d_%H%M%S')}")
    backup.write_text("\n".join(raw_lines) + ("\n" if raw_lines else ""), encoding="utf-8")
    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return damaged


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


def _unique_ids(values: Any) -> list[str]:
    """Deduplicate opaque identifiers, preserving the backend's own order.

    Unlike `_unique_doc_ids` this applies NO document-name normalization: chunk
    ids are opaque tokens, and trimming a ".md" or a path segment off one would
    be reading document semantics into a value that has none. A payload that is
    not a list is treated as absent rather than guessed at.
    """
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    ids: list[str] = []
    for value in values:
        text = str(value).strip() if value is not None else ""
        if not text or text in seen:
            continue
        seen.add(text)
        ids.append(text)
    return ids


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
            "  # re-run a batch: repeat the same command (see the module docstring)\n"
            "  qa_benchmark.py --config browsecompplus_retry_conf.json\n"
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
    parser.add_argument(
        "--wall-wait",
        type=float,
        default=WALL_RETRY_WAIT_SEC,
        help=f"Seconds to sleep when the provider Token Plan is exhausted, before retrying the pending work in process (default {int(WALL_RETRY_WAIT_SEC)} = 30 min)",
    )
    parser.add_argument(
        "--wall-max-waits",
        type=int,
        default=WALL_MAX_WAITS,
        help=f"Give up after this many plan-wall waits instead of waiting indefinitely (default {WALL_MAX_WAITS} = unlimited; a bounded run exits 1 and leaves its rows/verdicts on disk)",
    )
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
        persistent = sum(1 for row in existing if _is_persistent_ragflow_error(row.get("ragflow_error") or ""))
        transient = len(existing) - answered - persistent
        judged = sum(1 for record in _read_judgements(leaderboard_path).values() if not _is_backend_error_verdict(record))
        print(
            f"[resume] {answers_path.name}: {len(existing)} row(s) - {answered} answered + {persistent} persistent-error (skipped), {transient} transient failed/aborted will be retried; {judged} verdict(s) already stored"
        )

    # Parallelism: the CLI flag wins, then the config's top-level
    # "concurrency", then 1 (strictly serial).
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

    # Provenance before any phase: the fingerprint carries the commit of the
    # run that produced the rows, so --overwrite must drop a stale one (its
    # commit no longer describes what is about to be written), while a resume
    # keeps the original.
    fingerprint_path = output_dir / FINGERPRINT_NAME
    if args.overwrite and fingerprint_path.exists():
        fingerprint_path.unlink()
    _record_run_provenance(output_dir)

    if not args.skip_answers:
        if args.overwrite and answers_path.exists():
            answers_path.unlink()
        answer_aborted = run_answer_phase(
            client=_make_client(cfg),
            cfg=cfg,
            questions=selected,
            answers_path=answers_path,
            concurrency=concurrency,
            wall_wait_sec=args.wall_wait,
            wall_max_waits=args.wall_max_waits,
        )
        if answer_aborted:
            print(f"Run gave up waiting for the plan at the answer phase; {answers_path} holds the completed rows.")
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
        wall_wait_sec=args.wall_wait,
        wall_max_waits=args.wall_max_waits,
    )

    # The leaderboard export scores retrieval recall against each question's
    # EVIDENCE documents, which live in the questions file rather than in the
    # answers - an answers.jsonl from an older run predates the field, and
    # --skip-answers never re-reads the questions at all.
    rows = _merge_judgements(_dedupe_last(_read_jsonl(answers_path)), judgements)
    leaderboard = build_leaderboard(rows, cfg, load_evidence_doc_ids(_resolve_path(_questions_path(cfg), base_dir)))
    _write_json(leaderboard_path, leaderboard)

    if judge_aborted:
        print("Run gave up waiting for the plan at the judge phase; re-run the same command to continue.")
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
