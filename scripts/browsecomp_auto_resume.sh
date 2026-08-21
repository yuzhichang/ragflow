#!/usr/bin/env bash
# Auto-resume watcher for the browsecomp-plus full benchmark run.
#
# Every hour:
#   1. skip if a benchmark process is already running;
#   2. exit when the run is complete (marker file .complete);
#   3. otherwise launch a detached resume of the run.
#
# Quota handling is TIME-BASED, not progress-based: while the provider plan is
# exhausted, every attempt aborts within seconds (the benchmark's own circuit
# breaker trips after 3 "Token Plan" failures), so an attempt is a cheap probe.
# The gate therefore only suppresses probes for PROBE_INTERVAL_SEC after an
# attempt that hit the wall AND made no progress; once that window expires it
# probes again (otherwise recovery would never be noticed — the old
# progress-only gate deadlocked on a stale state value forever).
#
# Started with: (setsid nohup bash scripts/browsecomp_auto_resume.sh >> outputs/browsecomp_auto_resume_watch.log 2>&1 &)
# Stop with:    pkill -f browsecomp_auto_resume.sh   (or touch the .complete marker)

set -u
RUN_DIR="/home/zhichyu/github.com/infiniflow/ragflow"
OUT="$RUN_DIR/outputs/browsecomp_20260908_102644"
CONF="$RUN_DIR/scripts/browsecompplus_full_conf.json"
LOG_RESUME="$RUN_DIR/outputs/browsecomp_full_c3_resume_auto.log"
MARKER="$OUT/.complete"
STATE="$OUT/.auto_resume_state"
TOTAL=830
CYCLE_SEC=3600
PROBE_INTERVAL_SEC=7200   # max quiet time after a quota-walled attempt

read_state() {  # prints "last_clean last_epoch" (0 0 when unusable)
    python3 - "$STATE" <<'PY'
import sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("0 0"); raise SystemExit
try:
    raw = open(p, encoding="utf-8").read().split()
    clean = int(raw[0]); epoch = float(raw[1]) if len(raw) > 1 else 0.0
    # a stale value larger than the question count is a line count from an
    # older watcher version: treat it as "no usable state"
    if clean > 830 or clean < 0:
        clean = 0
    print(f"{clean} {epoch}")
except Exception:
    print("0 0")
PY
}

count_clean() {
    # DISTINCT question ids whose latest row is error-free (answers.jsonl
    # appends retry rows; the LAST row per run_id wins, mirroring the
    # benchmark's own resume dedup).
    python3 - "$OUT/answers.jsonl" <<'PY'
import json, sys
latest = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    latest[str(r.get("question_id"))] = (r.get("ragflow_error") or "").strip()
print(sum(1 for e in latest.values() if not e))
PY
}

cd "$RUN_DIR" || exit 1
echo "[watch] started $(date '+%F %T')"

while true; do
    # 1. a benchmark process is already working — let it run
    if pgrep -f "ragflow_benchmark.py" >/dev/null 2>&1; then
        echo "[watch] $(date '+%F %T') benchmark already running, skip"
        sleep "$CYCLE_SEC"
        continue
    fi

    # 2. completion check: every question answered clean AND judged
    if [ -f "$MARKER" ]; then
        echo "[watch] $(date '+%F %T') complete marker present, exiting"
        exit 0
    fi
    clean=$(count_clean)
    judged=$(grep -o '"run_key"' "$OUT/leaderboard.json" 2>/dev/null | wc -l)
    if [ "$clean" -ge "$TOTAL" ] && [ "$judged" -ge "$TOTAL" ]; then
        touch "$MARKER"
        echo "[watch] $(date '+%F %T') run COMPLETE ($clean clean, $judged judged) — exiting"
        exit 0
    fi

    # 3. quota gate: suppress probes only for PROBE_INTERVAL_SEC after an
    #    attempt that hit the wall and made no progress.
    read -r last_clean last_epoch < <(read_state)
    now=$(date +%s)
    since=$(( now - ${last_epoch%.*} ))
    quota_hit=false
    if [ -f "$LOG_RESUME" ] && tail -20 "$LOG_RESUME" | grep -q "provider plan quota exhausted"; then
        quota_hit=true
    fi
    if $quota_hit && [ "$clean" -le "$last_clean" ] && [ "$since" -lt "$PROBE_INTERVAL_SEC" ]; then
        echo "[watch] $(date '+%F %T') quota walled, quiet window ${since}s/${PROBE_INTERVAL_SEC}s ($clean/$TOTAL clean) — skip this cycle"
        sleep "$CYCLE_SEC"
        continue
    fi

    # 4. launch detached resume (a probe costs seconds when still walled)
    echo "$clean $now" > "$STATE"
    echo "[watch] $(date '+%F %T') launching resume ($clean/$TOTAL clean, $((TOTAL-clean)) to (re)run)"
    (setsid nohup /home/zhichyu/.venv/bin/python3 -u "$RUN_DIR/scripts/ragflow_benchmark.py" \
        --config "$CONF" --run "$OUT" >> "$LOG_RESUME" 2>&1 &)
    sleep 600   # give the resumed batch time to make progress before re-evaluating
done
