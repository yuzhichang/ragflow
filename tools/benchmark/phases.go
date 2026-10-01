package main

// The two run phases (port of QuotaBreaker, run_answer_phase /
// _answer_phase_once, run_judge_phase / _judge_phase_once and the resume
// mechanics: strip damaged rows / damaged judgements, in-process parking and
// the plan-wall wait).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Provider plan/quota circuit breaker. A WALL is durable - once the plan is
// exhausted every remaining question would fail the same way, so the run must
// stop with its completed rows intact. But the gateway reports short TPM/RPM
// bursts with the SAME wording ("已达到 Token Plan 用量上限"), and with N
// questions in flight one burst produces N failures within seconds: aborting
// an 830-question run on that is wrong. So the breaker only aborts when the
// failures are SPREAD OUT; failures clustered inside QUOTA_BURST_WINDOW_SEC
// are a burst - pause and probe again. QUOTA_ABORT_TOTAL bounds the pauses so
// a genuine wall still ends the run instead of looping.
const (
	quotaAbortAfter     = 3
	quotaBurstWindowSec = 120
	quotaBurstPauseSec  = 90
	quotaAbortTotal     = 12
	wallRetryWaitSec    = 1800
	wallMaxWaits        = 0
	wallWaitSliceSec    = 30
	wallWaitLogEverySec = 300
	errorRetryWaitSec   = 1800
)

type quotaBreaker struct {
	mu       sync.Mutex
	failedAt []time.Time
	total    int
}

// note ports QuotaBreaker.note: every phase outcome is fed to it; a non-quota
// outcome clears the history. Returns "" (keep going), "burst" (pause and
// probe again) or "wall" (the plan is exhausted: the caller waits and
// retries).
func (b *quotaBreaker) note(errText string) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !strings.Contains(errText, "Token Plan") {
		b.failedAt = nil
		b.total = 0
		return ""
	}
	now := time.Now()
	b.total++
	// Keep EVERY failure since the last non-quota row (no age-based dropping:
	// filtering by the window would make the span test below unreachable).
	b.failedAt = append(b.failedAt, now)
	if b.total >= quotaAbortTotal {
		return "wall"
	}
	if len(b.failedAt) < quotaAbortAfter {
		return ""
	}
	if b.failedAt[len(b.failedAt)-1].Sub(b.failedAt[0]) >= quotaBurstWindowSec*time.Second {
		return "wall"
	}
	// Clustered failures: a burst, not a wall. Drop the history so the next
	// probe starts clean.
	b.failedAt = nil
	return "burst"
}

// waitForPlanRefill ports _wait_for_plan_refill: sleep out a provider plan
// wall before the next retry round, sliced so the log keeps proving the
// process is parked on purpose (a 30-minute silence is indistinguishable from
// a hang).
func waitForPlanRefill(label string, wait time.Duration, roundNo, maxWaits int) {
	limit := "unlimited"
	if maxWaits > 0 {
		limit = fmt.Sprintf("at most %d wait(s)", maxWaits)
	}
	fmt.Printf("[%s] provider plan exhausted - waiting %ds before retry round %d (%s); every finished row/verdict stays on disk\n",
		label, int(wait.Seconds()), roundNo, limit)
	remaining := wait
	slept := time.Duration(0)
	nextNotice := wallWaitLogEverySec * time.Second
	for remaining > 0 {
		slice := wallWaitSliceSec * time.Second
		if slice > remaining {
			slice = remaining
		}
		time.Sleep(slice)
		slept += slice
		remaining -= slice
		if remaining > 0 && slept >= nextNotice {
			fmt.Printf("[%s] still waiting: %ds left before retry round %d\n", label, int(remaining.Seconds()), roundNo)
			nextNotice += wallWaitLogEverySec * time.Second
		}
	}
}

// stripDamagedRows ports _strip_damaged_rows: drop rows a resume should
// re-run, keep the ones it must not. A row whose ragflow_error is TRANSIENT
// is dropped (the resume re-runs it and appends a fresh row). A row carrying
// a PERSISTENT error - the provider refused the input itself (content
// policy), so every retry fails identically - is KEPT and treated as settled
// (q744: 5+ identical failures across resumes). Unparseable rows are dropped.
// A timestamped backup is written beside the file before the rewrite; a file
// with nothing to strip is left untouched. Returns the number of rows removed.
func stripDamagedRows(path string) int {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	rawLines := strings.Split(string(raw), "\n")
	var kept []string
	damaged := 0
	for _, line := range rawLines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var row map[string]any
		if err := json.Unmarshal([]byte(line), &row); err != nil || row == nil {
			damaged++
			continue
		}
		errText := strings.TrimSpace(strAny(row, "ragflow_error", ""))
		if errText != "" && !isPersistentRagflowError(errText) {
			damaged++
			continue
		}
		kept = append(kept, line)
	}
	if damaged == 0 {
		return 0
	}
	backup := path + fmt.Sprintf(".bak_strip_%s", time.Now().Format("0102_150405"))
	os.WriteFile(backup, []byte(strings.Join(rawLines, "\n")+"\n"), 0o644)
	os.WriteFile(path, []byte(strings.Join(kept, "\n")+"\n"), 0o644)
	return damaged
}

// stripDamagedJudgements ports _strip_damaged_judgements: drop judgements
// standing on an answer row that carries a ragflow_error. The answer phase
// re-runs damaged rows, so a judgement recorded while the row was damaged is
// stale state. A timestamped backup is written beside the file first.
func stripDamagedJudgements(leaderboardPath string, rows []map[string]any) int {
	if !fileExists(leaderboardPath) {
		return 0
	}
	data, err := os.ReadFile(leaderboardPath)
	if err != nil {
		fmt.Printf("[judge] could not read %s: %v\n", leaderboardPath, err)
		return 0
	}
	var leaderboard map[string]any
	if err := json.Unmarshal(data, &leaderboard); err != nil {
		fmt.Printf("[judge] could not read %s: %v\n", leaderboardPath, err)
		return 0
	}
	records, _ := leaderboard["per_query_judgements"].([]any)
	if records == nil {
		return 0
	}
	damagedKeys := map[string]bool{}
	for _, row := range rows {
		if errText := strings.TrimSpace(strAny(row, "ragflow_error", "")); errText != "" {
			if key := rowRunKey(row); key != "" {
				damagedKeys[key] = true
			}
		}
	}
	var kept []any
	for _, recordAny := range records {
		record, ok := recordAny.(map[string]any)
		if !ok || damagedKeys[strAny(record, "run_key", "")] {
			continue
		}
		kept = append(kept, record)
	}
	removed := len(records) - len(kept)
	if removed == 0 {
		return 0
	}
	backup := leaderboardPath + fmt.Sprintf(".bak_strip_%s", time.Now().Format("0102_150405"))
	os.WriteFile(backup, data, 0o644)
	leaderboard["per_query_judgements"] = kept
	writeJSON(leaderboardPath, leaderboard)
	return removed
}

// runAnswerPhase ports run_answer_phase: answer every pending question,
// riding out an exhausted plan in process. Returns true only when a bounded
// wait budget gave up.
func runAnswerPhase(c *httpClient, cfg map[string]any, questions []*question, answersPath string, concurrency int, wallWait time.Duration, wallMaxWaits int) bool {
	roundNo, waits := 0, 0
	for {
		roundNo++
		if roundNo > 1 {
			fmt.Printf("[answers] retry round %d (after %d plan-wall wait(s))\n", roundNo, waits)
		}
		aborted := answerPhaseOnce(c, cfg, questions, answersPath, concurrency)
		if !aborted {
			if waits > 0 {
				fmt.Println("[answers] plan recovered after", waits, "wait(s); no questions left")
			}
			return false
		}
		if wallMaxWaits > 0 && waits >= wallMaxWaits {
			fmt.Printf("[answers] giving up after %d plan-wall wait(s) (--wall-max-waits %d); %s holds every finished row\n",
				waits, wallMaxWaits, answersPath)
			return true
		}
		waits++
		waitForPlanRefill("answers", wallWait, roundNo+1, wallMaxWaits)
	}
}

// answerPhaseOnce ports _answer_phase_once: one dispatch round. Returns true
// when the quota breaker saw a wall.
func answerPhaseOnce(c *httpClient, cfg map[string]any, questions []*question, answersPath string, concurrency int) bool {
	if stripped := stripDamagedRows(answersPath); stripped > 0 {
		fmt.Printf("[answers] resume: stripped %d damaged row(s) (transient ragflow_error); backup kept beside the file\n", stripped)
	}
	// A row counts as done when it is clean OR when its error is PERSISTENT -
	// the provider refused the input itself, and a retry would only reproduce
	// the refusal.
	completed := map[string]bool{}
	for key, row := range readJSONLByID(answersPath) {
		errText := strings.TrimSpace(strAny(row, "ragflow_error", ""))
		if errText == "" || isPersistentRagflowError(errText) {
			completed[key] = true
		}
	}

	chatCfg, _ := cfg["ragflow_chat"].(map[string]any)
	freshPerQuestion := true
	if v, ok := chatCfg["fresh_session_per_question"].(bool); ok {
		freshPerQuestion = v
	}
	sharedOK := !freshPerQuestion && concurrency == 1
	sharedSessionID := ""
	if sharedOK {
		id, err := createSession(c, strAny(cfg, "chat_id", ""))
		if err != nil {
			fmt.Printf("[answers] %v\n", err)
			return false
		}
		sharedSessionID = id
	} else if concurrency > 1 && !freshPerQuestion {
		fmt.Println("[answers] concurrency > 1 forces fresh_session_per_question (a shared session's history would interleave)")
	}

	// seq/total count the PENDING questions, not the batch: a resumed run must
	// not print "52/36" (batch position over pending total).
	type job struct {
		seq int
		q   *question
		rid string
	}
	var jobs []job
	for _, q := range questions {
		if !completed[q.questionID] {
			jobs = append(jobs, job{len(jobs) + 1, q, q.questionID})
		}
	}
	if skipped := len(questions) - len(jobs); skipped > 0 {
		fmt.Printf("[answers] skip %d existing row(s)\n", skipped)
	}
	if len(jobs) == 0 {
		return false
	}
	total := len(jobs)
	started := time.Now()
	breaker := &quotaBreaker{}
	var writeMu sync.Mutex

	answerOne := func(j job) map[string]any {
		row := map[string]any{
			"run_id":             j.rid,
			"question_id":        j.q.questionID,
			"question":           j.q.question,
			"gold_answer":        j.q.goldAnswer,
			"reasoning_types":    toAnySlice(j.q.reasoningTypes),
			"ragflow_answer":     "",
			"ragflow_error":      "",
			"ragflow_session_id": nil,
		}
		if len(j.q.expectedDocs) > 0 {
			row["expected_doc_ids"] = toAnySlice(uniqueDocIDs(toAnySlice(j.q.expectedDocs)))
		}
		if len(j.q.evidenceDocs) > 0 {
			row["evidence_doc_ids"] = toAnySlice(uniqueDocIDs(toAnySlice(j.q.evidenceDocs)))
		}
		payload, err := askRagflow(c, cfg, j.q.question, sharedSessionID)
		if err == nil {
			answer := extractAnswerText(payload)
			row["ragflow_answer"] = answer
			if sid := extractSessionID(payload); sid != "" {
				row["ragflow_session_id"] = sid
			}
			for k, v := range extractRunStats(payload) {
				row[k] = v
			}
			if answer == "" {
				row["ragflow_error"] = "empty_ragflow_answer"
			} else if strings.HasPrefix(strings.TrimLeft(answer, " \t\n"), "**ERROR**") {
				row["ragflow_error"] = strings.TrimSpace(answer)
			}
		} else {
			row["ragflow_error"] = err.Error()
		}
		// Whether the run ever SAW the documents that carry the answer:
		// `served` is the run's own retrieval record, `cited` is what the
		// deliverable's own lines name - a doc can be served and still never
		// used. Without this pair a retrieval miss is indistinguishable from
		// a reasoning failure, and the two need opposite fixes.
		if len(j.q.expectedDocs) > 0 {
			expected := map[string]bool{}
			for _, d := range j.q.expectedDocs {
				expected[d] = true
			}
			servedSet := map[string]bool{}
			for _, d := range asDocIDList(row["retrieved_docids"]) {
				servedSet[d] = true
			}
			var served, cited []string
			for d := range expected {
				if servedSet[d] {
					served = append(served, d)
				}
				if strings.Contains(strAny(row, "ragflow_answer", ""), d) {
					cited = append(cited, d)
				}
			}
			sortStrings(served)
			sortStrings(cited)
			row["gold_doc_served"] = toAnySlice(served)
			row["gold_doc_cited"] = toAnySlice(cited)
		}
		status := strAny(row, "ragflow_error", "")
		if status == "" {
			status = fmt.Sprintf("%d chars", len(strAny(row, "ragflow_answer", "")))
		}
		fmt.Printf("[answers] %d/%d %s: %s\n", j.seq, total, j.rid, status)
		return row
	}

	handle, err := os.OpenFile(answersPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Printf("[answers] could not open %s: %v\n", answersPath, err)
		return false
	}
	defer handle.Close()

	runOne := func(j job) bool {
		// A TRANSIENT failure does not produce a row: the question parks for
		// ERROR_RETRY_WAIT_SEC and retries in process, for as long as it
		// takes - the provider being down is a property of the clock, not of
		// the question. A PERSISTENT failure (content policy) is terminal and
		// recorded.
		startedOne := time.Now()
		attempt := 0
		var row map[string]any
		for {
			row = answerOne(j)
			errText := strings.TrimSpace(strAny(row, "ragflow_error", ""))
			if errText == "" || isPersistentRagflowError(errText) {
				break
			}
			attempt++
			fmt.Printf("[answers] %d/%d %s: retryable error (attempt %d) - %.160s; parking %ds, no row recorded\n",
				j.seq, total, j.rid, attempt, errText, errorRetryWaitSec)
			time.Sleep(errorRetryWaitSec * time.Second)
		}
		row["answer_elapsed_seconds"] = round3(time.Since(startedOne).Seconds())
		line, _ := json.Marshal(row)
		writeMu.Lock()
		handle.Write(append(line, '\n'))
		handle.Sync()
		verdict := breaker.note(strAny(row, "ragflow_error", ""))
		writeMu.Unlock()
		switch verdict {
		case "burst":
			fmt.Printf("[answers] Token Plan failures clustered inside %ds - rate-limit burst, not an exhausted plan: pausing %ds and probing again\n",
				quotaBurstWindowSec, quotaBurstPauseSec)
			time.Sleep(quotaBurstPauseSec * time.Second)
		case "wall":
			fmt.Printf("[answers] provider plan quota exhausted (%d Token Plan failures, spread over >= %ds) - this round stops here; the question(s) without an answer are retried after the plan-wall wait.\n",
				breaker.total, quotaBurstWindowSec)
		}
		return verdict == "wall"
	}

	aborted := false
	if concurrency <= 1 {
		for _, j := range jobs {
			if runOne(j) {
				aborted = true
				break
			}
		}
	} else {
		jobsCh := make(chan job)
		var wg sync.WaitGroup
		var mu sync.Mutex
		for w := 0; w < concurrency; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range jobsCh {
					if runOne(j) {
						mu.Lock()
						aborted = true
						mu.Unlock()
					}
				}
			}()
		}
		for _, j := range jobs {
			mu.Lock()
			stop := aborted
			mu.Unlock()
			if stop {
				break // queued-but-unstarted questions are not dispatched
			}
			jobsCh <- j
		}
		close(jobsCh)
		wg.Wait()
	}

	fmt.Printf("[answers] %d run(s) in %.1fs -> %s\n", total, time.Since(started).Seconds(), answersPath)
	return aborted
}

// runJudgePhase ports run_judge_phase: judge every row without a verdict,
// riding out an exhausted plan in process. Returns (judgements, gaveUp).
func runJudgePhase(c *httpClient, cfg map[string]any, answersPath, leaderboardPath string, concurrency int, wallWait time.Duration, wallMaxWaits int) (map[string]map[string]any, bool) {
	roundNo, waits := 0, 0
	for {
		roundNo++
		if roundNo > 1 {
			fmt.Printf("[judge] retry round %d (after %d plan-wall wait(s))\n", roundNo, waits)
		}
		judgements, aborted := judgePhaseOnce(c, cfg, answersPath, leaderboardPath, concurrency)
		if !aborted {
			if waits > 0 {
				fmt.Println("[judge] plan recovered after", waits, "wait(s); no row left unjudged")
			}
			return judgements, false
		}
		if wallMaxWaits > 0 && waits >= wallMaxWaits {
			fmt.Printf("[judge] giving up after %d plan-wall wait(s) (--wall-max-waits %d); %s holds every stored verdict\n",
				waits, wallMaxWaits, leaderboardPath)
			return judgements, true
		}
		waits++
		waitForPlanRefill("judge", wallWait, roundNo+1, wallMaxWaits)
	}
}

// judgePhaseOnce ports _judge_phase_once: one judging round.
func judgePhaseOnce(c *httpClient, cfg map[string]any, answersPath, leaderboardPath string, concurrency int) (map[string]map[string]any, bool) {
	if !fileExists(answersPath) {
		fmt.Printf("[judge] missing answers file: %s\n", answersPath)
		return map[string]map[string]any{}, false
	}
	rows := dedupeLast(readJSONL(answersPath))
	if stripped := stripDamagedJudgements(leaderboardPath, rows); stripped > 0 {
		fmt.Printf("[judge] resume: stripped %d judgement(s) standing on damaged answer row(s); backup kept beside the file\n", stripped)
	}
	// Judging scope: the answer rows MINUS the rows still carrying a
	// ragflow_error (those are going to be re-run by the answer phase), MINUS
	// the questions already judged from an error-free row.
	var judgedRows []map[string]any
	for _, row := range rows {
		if strings.TrimSpace(strAny(row, "ragflow_error", "")) == "" {
			judgedRows = append(judgedRows, row)
		}
	}
	completed := map[string]bool{}
	for key, record := range readJudgements(leaderboardPath) {
		if !isBackendErrorVerdict(record) {
			completed[key] = true
		}
	}
	type jjob struct {
		seq int
		row map[string]any
	}
	var todo []jjob
	for _, row := range judgedRows {
		if !completed[rowRunKey(row)] {
			todo = append(todo, jjob{len(todo) + 1, row})
		}
	}
	if skipped := len(judgedRows) - len(todo); skipped > 0 {
		fmt.Printf("[judge] skip %d existing row(s)\n", skipped)
	}
	judgements := readJudgements(leaderboardPath)
	if len(todo) == 0 {
		return judgements, false
	}

	breaker := &quotaBreaker{}
	var writeMu sync.Mutex
	aborted := false

	finishJudge := func(j jjob) bool {
		// A TRANSIENT judge error does not persist a verdict: the row parks
		// for ERROR_RETRY_WAIT_SEC and re-judges in process. A PERSISTENT
		// error is recorded - it is re-judged after a resume.
		startedOne := time.Now()
		key := rowRunKey(j.row)
		attempt := 0
		var record map[string]any
		for {
			record = judgeOne(c, cfg, j.row)
			jerr := strings.TrimSpace(strAny(record, "judge_error", ""))
			if jerr == "" || isPersistentRagflowError(jerr) || jerr == "excluded_due_to_ragflow_error" {
				break
			}
			attempt++
			fmt.Printf("[judge] %d/%d %s: retryable judge error (attempt %d) - %.160s; parking %ds, no verdict recorded\n",
				j.seq, len(todo), key, attempt, jerr, errorRetryWaitSec)
			time.Sleep(errorRetryWaitSec * time.Second)
		}
		elapsed := round3(time.Since(startedOne).Seconds())
		record["query_id"] = strAny(j.row, "question_id", key)
		record["run_key"] = key
		record["judge_elapsed_seconds"] = elapsed

		writeMu.Lock()
		judgements[key] = record
		// One write per judged row: an aborted run keeps its completed
		// verdicts on disk, which is exactly what the resume logic reads.
		writeJSON(leaderboardPath, leaderboardWithJudgements(dedupeLast(readJSONL(answersPath)), judgements, cfg))
		verdict := breaker.note(strAny(record, "judge_error", ""))
		writeMu.Unlock()

		status := strAny(record, "judge_error", "")
		if status == "" {
			status = fmt.Sprintf("correct=%v", record["correct"])
		}
		fmt.Printf("[judge] %d/%d %s: %s (%.3fs)\n", j.seq, len(todo), key, status, elapsed)
		switch verdict {
		case "burst":
			fmt.Printf("[judge] Token Plan failures clustered inside %ds - rate-limit burst, not an exhausted plan: pausing %ds and probing again\n",
				quotaBurstWindowSec, quotaBurstPauseSec)
			time.Sleep(quotaBurstPauseSec * time.Second)
		case "wall":
			fmt.Printf("[judge] provider plan quota exhausted (%d Token Plan failures, spread over >= %ds) - this round stops here; the row(s) left without a verdict are re-judged after the plan-wall wait.\n",
				breaker.total, quotaBurstWindowSec)
		}
		return verdict == "wall"
	}

	if concurrency <= 1 {
		for _, j := range todo {
			if finishJudge(j) {
				aborted = true
				break
			}
		}
	} else {
		jobsCh := make(chan jjob)
		var wg sync.WaitGroup
		var mu sync.Mutex
		for w := 0; w < concurrency; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for j := range jobsCh {
					if finishJudge(j) {
						mu.Lock()
						aborted = true
						mu.Unlock()
					}
				}
			}()
		}
		for _, j := range todo {
			mu.Lock()
			stop := aborted
			mu.Unlock()
			if stop {
				break
			}
			jobsCh <- j
		}
		close(jobsCh)
		wg.Wait()
	}

	fmt.Printf("[judge] %d judgement(s) -> %s\n", len(judgements), leaderboardPath)
	return judgements, aborted
}

// readJSONLByID ports _read_jsonl_by_id.
func readJSONLByID(path string) map[string]map[string]any {
	out := map[string]map[string]any{}
	for _, row := range readJSONL(path) {
		if key := rowRunKey(row); key != "" {
			out[key] = row
		}
	}
	return out
}

// countSettledJudgements ports the judged count of the [resume] line.
func countSettledJudgements(records map[string]map[string]any) int {
	n := 0
	for _, record := range records {
		if !isBackendErrorVerdict(record) {
			n++
		}
	}
	return n
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}

var _ = filepath.Join // keep import if unused paths change
