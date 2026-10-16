package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func boolPtr(b bool) *bool { return &b }

// ---------------------------------------------------------------- pure functions

func TestSamplePassDeterministicAndDisjoint(t *testing.T) {
	n := 23
	order := samplePass(42, 1, n)
	if len(order) != n {
		t.Fatalf("want %d indices, got %d", n, len(order))
	}
	seen := map[int]bool{}
	for _, v := range order {
		if v < 0 || v >= n || seen[v] {
			t.Fatalf("not a permutation: %v", order)
		}
		seen[v] = true
	}
	if got := samplePass(42, 1, n); !equalInts(got, order) {
		t.Fatalf("same (seed,pass) must reproduce the order")
	}
	if equalInts(samplePass(42, 2, n), order) {
		t.Fatalf("a different pass must draw a different order")
	}
}

func TestPassBatchesCoverWithoutOverlap(t *testing.T) {
	batches := passBatches(7, 3, 25, 10)
	if len(batches) != 3 || len(batches[0]) != 10 || len(batches[2]) != 5 {
		t.Fatalf("unexpected batch shapes: %v", lens(batches))
	}
	all := map[int]int{}
	for _, b := range batches {
		for _, v := range b {
			all[v]++
		}
	}
	if len(all) != 25 {
		t.Fatalf("batches must cover the whole set exactly once, got %d distinct", len(all))
	}
}

func TestNormalizeStrategyText(t *testing.T) {
	cases := map[string]string{
		"1. Search clues in parallel.":         "Search clues in parallel",
		"- \"Assign one subagent per clue\"":   "Assign one subagent per clue",
		"Group subagents by clue specificity.": "Group subagents by clue specificity",
	}
	for in, want := range cases {
		if got := normalizeStrategyText(in, 30); got != want {
			t.Errorf("normalize(%q) = %q, want %q", in, got, want)
		}
	}
	long := strings.Repeat("word ", 40)
	if got := countWords(normalizeStrategyText(long, 30)); got != 30 {
		t.Errorf("expected 30-word truncation, got %d words", got)
	}
}

func TestStrategyFileRoundTrip(t *testing.T) {
	set := newStrategySet()
	set.Items = []strategyItem{
		{ID: "s0001", Text: "Search clues in parallel", TotalScore: 3},
		{ID: "s0002", Text: "Group subagents by clue specificity", TotalScore: 1},
	}
	set.rank(10)
	version, texts := parseStrategyFile(set.renderFile("pass=1 batch=0"))
	if version != set.Version || len(texts) != 2 || texts[0] != "Search clues in parallel" {
		t.Fatalf("round-trip failed: v=%d texts=%v", version, texts)
	}
}

func TestClassifyFailures(t *testing.T) {
	cases := []struct {
		err  error
		want failClass
	}{
		{errorsNew("minimax API error: 已达到 Token Plan 用量上限"), failQuota},
		{errorsNew("rate limit exceeded"), failQuota},
		{&httpStatusError{Status: 429}, failQuota},
		{errorsNew("connection reset by peer"), failNetwork},
		{&httpStatusError{Status: 502}, failNetwork},
		{context.DeadlineExceeded, failTimeout},
		{errorsNew("request timed out"), failTimeout},
		{&httpStatusError{Status: 400}, failPermanent},
		{errorsNew("json: cannot unmarshal"), failPermanent},
	}
	for _, c := range cases {
		if got := classify(c.err); got != c.want {
			t.Errorf("classify(%v) = %s, want %s", c.err, got, c.want)
		}
	}
}

func TestParseGraderVerdict(t *testing.T) {
	v, err := parseGraderVerdict("extracted_final_answer: Paris\ncorrect: yes\nconfidence: 90%")
	if err != nil || !v.Correct || v.Confidence != 90 {
		t.Fatalf("yes verdict = %+v err=%v", v, err)
	}
	if _, err := parseGraderVerdict("no verdict here"); err == nil {
		t.Fatal("a reply without `correct:` must be an error, never an implicit no")
	}
}

func TestParseReflectionStrict(t *testing.T) {
	active := map[string]bool{"s0001": true}
	good := `{"ratings":[{"strategy_id":"s0001","score":2},{"strategy_id":"s9999","score":1},{"strategy_id":"s0001","score":1}],
	          "lessons":[{"text":"Parallelize clue search","score":1},{"text":"","score":2},{"text":"no score"}]}`
	lessons, ratings, err := parseReflection(good, active, 2)
	if err != nil {
		t.Fatalf("parseReflection: %v", err)
	}
	if len(ratings) != 1 || ratings[0].Score != 2 {
		t.Fatalf("unknown/duplicate ratings must be dropped, got %+v", ratings)
	}
	if len(lessons) != 1 || lessons[0].Text != "Parallelize clue search" {
		t.Fatalf("empty/missing-score lessons must be dropped, got %+v", lessons)
	}
}

func TestBuildTrainsetIgnoresUnjudged(t *testing.T) {
	dir := t.TempDir()
	qp := filepath.Join(dir, "questions.jsonl")
	var qb []byte
	for _, id := range []string{"1", "2", "3"} {
		rec, _ := json.Marshal(map[string]any{"id": id, "question": "q" + id, "gold_answer": "a" + id})
		qb = append(qb, append(rec, '\n')...)
	}
	if err := os.WriteFile(qp, qb, 0o644); err != nil {
		t.Fatal(err)
	}
	mkRun := func(name string, rows []map[string]any) {
		rd := filepath.Join(dir, name)
		_ = os.MkdirAll(rd, 0o755)
		b, _ := json.Marshal(map[string]any{"per_query_judgements": rows})
		_ = os.WriteFile(filepath.Join(rd, "leaderboard.json"), b, 0o644)
	}
	mkRun("browsecomp_runA", []map[string]any{
		{"query_id": "1", "correct": false},
		{"query_id": "2", "correct": false},
		{"query_id": "3", "correct": true},
		{"query_id": "1", "correct": nil},
	})
	mkRun("frames_excluded", []map[string]any{{"query_id": "1", "correct": false}})

	out := filepath.Join(dir, "train", "top.jsonl")
	meta, err := buildTrainset(dir, qp, 2, out)
	if err != nil {
		t.Fatalf("buildTrainset: %v", err)
	}
	if meta.NRuns != 1 {
		t.Fatalf("strict filter must keep only browsecomp/c2 runs, got %d", meta.NRuns)
	}
	recs, meta2, err := loadTrainset(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 2 || recs[0].ID != "1" || recs[0].Stats.Wrong != 1 || recs[0].Stats.Seen != 1 {
		t.Fatalf("unjudged rows must not inflate counts: %+v", recs[0].Stats)
	}
	if meta2.SHA256 != meta.SHA256 {
		t.Fatal("sha mismatch build/load")
	}
	_ = os.WriteFile(out, []byte("{\"id\":\"x\"}\n"), 0o644)
	if _, _, err := loadTrainset(out); err == nil {
		t.Fatal("a changed trainset must fail the sha256 check")
	}
}

// ---------------------------------------------------------------- review regression

func TestMergeConsumesOldIdentity(t *testing.T) {
	s := &strategySet{Items: []strategyItem{{ID: "s0001", Text: "Search clues", TotalScore: 5}}}
	n, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{{
		ID: "s0001", Text: "Search clues", SourceIDs: []string{"s0001"},
		Terse: boolPtr(true), CorpusSpecific: boolPtr(false),
	}}}, nil, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if len(n.Items) != 1 || n.Items[0].ID != "s0001" {
		t.Fatalf("unchanged strategy was cloned: %+v", n.Items)
	}
}

func TestRejectionRemovesCandidate(t *testing.T) {
	s := &strategySet{Items: []strategyItem{{ID: "s0001", Text: "Bad"}, {ID: "s0002", Text: "Good"}}}
	n, _, err := applyMerge(s, llmMergeOutput{Rejected: []llmRejection{{SourceID: "s0001", Reason: "bad"}}}, nil, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if n.hasID("s0001") {
		t.Fatal("explicitly rejected source remains in the pool")
	}
}

func TestMissingSourcesRejected(t *testing.T) {
	ls := []lesson{{TentativeID: "t1", Text: "Verify clues", Score: 2}}
	if _, _, err := applyMerge(newStrategySet(), llmMergeOutput{}, ls, 10, 30); err == nil {
		t.Fatal("missing source mapping accepted")
	}
}

func TestLosingChallengerKeepsOthers(t *testing.T) {
	s := &strategySet{Items: []strategyItem{{ID: "s0001", Text: "A", TotalScore: 1}, {ID: "s0002", Text: "B", TotalScore: 9}}}
	ls := []lesson{{TentativeID: "t1", Text: "C", Score: 5}}
	n, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{{
		Text: "C", SourceIDs: []string{"t1"}, Contradicts: []string{"s0001", "s0002"},
		Terse: boolPtr(true), CorpusSpecific: boolPtr(false),
	}}}, ls, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if !n.hasID("s0001") {
		t.Fatal("A removed even though the challenger ultimately lost")
	}
}

func TestPlateauConvergesAfterCommit(t *testing.T) {
	o, err := openOutputDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.release()
	e := &executor{cfg: defaultConfig(), out: o}
	cp := &checkpoint{}
	cp.Eval.LastCommitted = &evalResult{EvalID: "old", Accuracy: 50, Scored: 10, Coverage: 1}
	r := evalResult{EvalID: "new", Accuracy: 50, Scored: 10, Coverage: 1}
	if _, err := e.commitEval(cp, newStrategySet(), r); err != nil {
		t.Fatal(err)
	}
	if !e.converged(cp, r) {
		t.Fatal("a plateau must converge in the production call order")
	}
}

func TestNewPassStartsNewEvaluation(t *testing.T) {
	o, err := openOutputDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.release()
	e := &executor{out: o}
	cp := &checkpoint{Pass: 2, Phase: phaseTrain}
	cp.Eval.EvalID = "old"
	cp.Eval.NextIndex = 1
	r, err := e.evaluatePass(context.Background(), cp, newStrategySet(), []trainsetRecord{{ID: "q1"}})
	if err != nil {
		t.Fatal(err)
	}
	if r.EvalID == "old" {
		t.Fatal("new pass reused the previous eval id")
	}
}

func TestCanceledEvalStaysPending(t *testing.T) {
	o, err := openOutputDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.release()
	e := &executor{out: o}
	cp := &checkpoint{Pass: 1}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := e.evaluatePass(ctx, cp, newStrategySet(), []trainsetRecord{{ID: "q1"}}); err == nil {
		t.Fatal("a canceled incomplete evaluation returned success")
	}
}

func TestSessionHistoryRealShape(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"code":0,"data":[{"id":"s1","message":[{"role":"user","content":"q"},{"role":"assistant","content":"answer"}]}]}`)
	}))
	defer srv.Close()
	c := &httpClient{base: srv.URL, hc: srv.Client()}
	h, ok := c.sessionHistory(context.Background(), "c1", "s1")
	st, ans := matchTurn(h, "q")
	if !ok || st != reconcileDone || ans != "answer" {
		t.Fatalf("stored turn misread: ok=%v state=%v hist=%+v", ok, st, h)
	}
}

func TestOrdinaryAnswerIsNotFailure(t *testing.T) {
	for _, a := range []string{"Geoffrey Chaucer", "born in 1940s", "the value is 429", "eof appears here"} {
		if answerLooksLikeError(a) {
			t.Fatalf("ordinary answer %q classified as a provider failure", a)
		}
	}
	if !answerLooksLikeError("**ERROR**: minimax API error: 用量上限") {
		t.Fatal("a decorated provider error must be detected")
	}
}

func TestResumeRejectsChangedTrainset(t *testing.T) {
	cfg := defaultConfig()
	cp := &checkpoint{Seed: cfg.Seed}
	if err := validateResume(cp, cfg, trainsetMeta{SHA256: "different"}); err == nil {
		t.Fatal("changed trainset accepted on resume")
	}
}

func TestFailedMergeDiscardsBatchScores(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/sessions") {
			fmt.Fprint(w, `{"code":0,"data":{"id":"session"}}`)
			return
		}
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		ans := "wrong"
		switch b["chat_id"] {
		case "judge":
			ans = "correct: no"
		case "merge":
			ans = "invalid merge"
		}
		if q, _ := b["question"].(string); strings.HasPrefix(q, "Your search did not find") {
			// A lesson (not just a rating) is needed to force the merge call.
			ans = `{"ratings":[{"strategy_id":"s0001","score":-2}],"lessons":[{"text":"Split clues across subagents","score":1}]}`
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"code": 0, "data": map[string]any{"answer": ans}})
	}))
	defer srv.Close()

	cfg := defaultConfig()
	cfg.OutputDir = t.TempDir()
	cfg.ChatID, cfg.JudgeChatID, cfg.MergeChatID = "answer", "judge", "merge"
	cfg.MaxPasses, cfg.BatchSize = 1, 1

	o, err := openOutputDir(cfg.OutputDir)
	if err != nil {
		t.Fatal(err)
	}
	cp := &checkpoint{Phase: phaseTrain, Pass: 1, Seed: cfg.Seed, Items: []strategyItem{{ID: "s0001", Text: "Search clues", TotalScore: 5}}}
	if err := o.writeCheckpoint(cp); err != nil {
		t.Fatal(err)
	}
	o.release()

	e := &executor{cfg: cfg, client: &httpClient{base: srv.URL, hc: srv.Client()}, quiet: true}
	if err := e.runTraining(context.Background(), []trainsetRecord{{ID: "q1", Question: "q", GoldAnswer: "a"}}, trainsetMeta{}); err != nil {
		t.Fatal(err)
	}
	got, _, err := loadCheckpoint(cfg.OutputDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Items) == 0 || got.Items[0].TotalScore != 5 {
		t.Fatalf("a failed merge committed the batch's scores: %+v version=%d", got.Items, got.Version)
	}
}

func TestLogWriteFailureStopsCommit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		fmt.Fprint(w, "bad request")
	}))
	defer srv.Close()
	cfg := defaultConfig()
	cfg.OutputDir = t.TempDir()
	cfg.MaxPasses, cfg.BatchSize = 1, 1
	if err := os.Mkdir(filepath.Join(cfg.OutputDir, "state.jsonl"), 0o755); err != nil {
		t.Fatal(err)
	}
	e := &executor{cfg: cfg, client: &httpClient{base: srv.URL, hc: srv.Client()}, quiet: true}
	if err := e.runTraining(context.Background(), []trainsetRecord{{ID: "q1"}}, trainsetMeta{}); err == nil {
		t.Fatal("a failed mandatory audit write was accepted and the cursor advanced")
	}
}

// TestBuildTrainsetRanksByWrongRate pins the rate-first ordering: by absolute wrong
// count q2 (wrong=2) would lead, but its rate (0.5) is below q1's (1.0).
func TestBuildTrainsetRanksByWrongRate(t *testing.T) {
	dir := t.TempDir()
	qp := filepath.Join(dir, "questions.jsonl")
	var qb []byte
	for _, id := range []string{"1", "2", "3"} {
		rec, _ := json.Marshal(map[string]any{"id": id, "question": "q" + id, "gold_answer": "a" + id})
		qb = append(qb, append(rec, '\n')...)
	}
	if err := os.WriteFile(qp, qb, 0o644); err != nil {
		t.Fatal(err)
	}
	rows := []map[string]any{}
	add := func(id string, wrong, ok int) {
		for i := 0; i < wrong; i++ {
			rows = append(rows, map[string]any{"query_id": id, "correct": false})
		}
		for i := 0; i < ok; i++ {
			rows = append(rows, map[string]any{"query_id": id, "correct": true})
		}
	}
	add("1", 1, 0) // rate 1.00
	add("2", 2, 2) // rate 0.50
	add("3", 1, 3) // rate 0.25
	rd := filepath.Join(dir, "browsecomp_runA")
	_ = os.MkdirAll(rd, 0o755)
	b, _ := json.Marshal(map[string]any{"per_query_judgements": rows})
	_ = os.WriteFile(filepath.Join(rd, "leaderboard.json"), b, 0o644)

	out := filepath.Join(dir, "top.jsonl")
	if _, err := buildTrainset(dir, qp, 10, out); err != nil {
		t.Fatal(err)
	}
	recs, _, err := loadTrainset(out)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for _, r := range recs {
		got = append(got, r.ID)
	}
	if strings.Join(got, ",") != "1,2,3" {
		t.Fatalf("want rate-first order 1,2,3, got %v", got)
	}
}

// ---------------------------------------------------------------- round-2 regression

// An item may only claim an existing id that is one of its own sources.
func TestOutputIdentityMustBelongToSources(t *testing.T) {
	s := &strategySet{Items: []strategyItem{{ID: "s0001", Text: "Original", TotalScore: 5}}}
	ls := []lesson{{TentativeID: "t1", Text: "Different method", Score: 2}}
	if _, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{{
		ID: "s0001", Text: "Different method", SourceIDs: []string{"t1"},
		Terse: boolPtr(true), CorpusSpecific: boolPtr(false),
	}}}, ls, 10, 30); err == nil {
		t.Fatal("unconsumed old identity overwritten by an unrelated source")
	}
}

// A conflict loser must not be resurrected by its own rewrite item.
func TestConflictLoserNotResurrected(t *testing.T) {
	s := &strategySet{Items: []strategyItem{{ID: "s0001", Text: "Parallel", TotalScore: 5}, {ID: "s0002", Text: "Sequential", TotalScore: 1}}}
	a := llmItem{ID: "s0001", Text: "Parallel", SourceIDs: []string{"s0001"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false), Contradicts: []string{"s0002"}}
	b := llmItem{ID: "s0002", Text: "Sequential", SourceIDs: []string{"s0002"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false)}
	n, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{a, b}}, nil, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if n.hasID("s0002") {
		t.Fatal("the conflict loser survived via its rewrite exemption")
	}
}

// A mapped (predicted) score must survive a rewrite of the same strategy.
func TestDuplicateScoreSurvivesRewrite(t *testing.T) {
	s := &strategySet{Items: []strategyItem{{ID: "s0001", Text: "Verify", TotalScore: 5, RatingCount: 2}}}
	ls := []lesson{{TentativeID: "t1", Text: "Verify", Score: 2}}
	d := llmItem{Text: "Verify", SourceIDs: []string{"t1"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false), DuplicateOf: "s0001"}
	keep := llmItem{ID: "s0001", Text: "Verify", SourceIDs: []string{"s0001"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false)}
	n, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{keep, d}}, ls, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if got := n.byID("s0001"); got == nil || got.TotalScore != 7 {
		t.Fatalf("mapped +2 disappeared: %+v", n.Items)
	}
}

// An all-rejected update keeps the previous pool (and version).
func TestEmptyUpdatePreservesPool(t *testing.T) {
	s := &strategySet{Version: 5, Items: []strategyItem{{ID: "s0001", Text: "Verify", TotalScore: 5}}}
	n, _, err := applyMerge(s, llmMergeOutput{Rejected: []llmRejection{{SourceID: "s0001", Reason: "drop"}}}, nil, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if len(n.Items) == 0 || n.Version != 5 {
		t.Fatalf("all-rejected update discarded the previous pool: items=%d version=%d", len(n.Items), n.Version)
	}
}

// nextID must never reuse a retired id.
func TestNextIDMonotonic(t *testing.T) {
	s := newStrategySet()
	id1 := s.allocID()
	_ = s.allocID() // id2, later "deleted"
	s.Items = []strategyItem{{ID: id1}}
	if id3 := s.allocID(); id3 != "s0003" {
		t.Fatalf("id reused after deletion: %s (want s0003)", id3)
	}
}

// A hard kill leaves a partial trailing line; appending must not corrupt the next record.
func TestJSONLPartialTailIsTruncated(t *testing.T) {
	o, err := openOutputDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.release()
	if err := os.WriteFile(o.path("x.jsonl"), []byte("{\"a\":1}\n{\"b\":2"), artifactPerm); err != nil {
		t.Fatal(err)
	}
	if err := o.appendJSONL("x.jsonl", map[string]any{"c": 3}); err != nil {
		t.Fatal(err)
	}
	rows, err := readJSONLStrict(o.path("x.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0]["a"] == nil || rows[1]["c"] == nil {
		t.Fatalf("partial tail not handled: %+v", rows)
	}
}

// The config snapshot must never contain the API key.
func TestConfigSnapshotRedacted(t *testing.T) {
	o, err := openOutputDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.release()
	cfg := defaultConfig()
	cfg.RAGFlowAPIKey = "SECRET-KEY-DO-NOT-LEAK"
	if err := writeRunIdentity(o, cfg, "ids", "fp"); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(o.path("_config_snapshot.json"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "SECRET-KEY-DO-NOT-LEAK") {
		t.Fatal("config snapshot leaked the API key")
	}
}

// ---------------------------------------------------------------- round-3 regression

// duplicate_of must not consume its own mapping target.
func TestDuplicateMappingKeepsTarget(t *testing.T) {
	s := &strategySet{NextID: 2, Items: []strategyItem{{ID: "s0001", Text: "Verify", TotalScore: 5}, {ID: "s0002", Text: "Search", TotalScore: 1}}}
	ls := []lesson{{TentativeID: "t1", QID: "q1", Text: "Verify", Score: 2}}
	d := llmItem{ID: "s0001", Text: "Verify", SourceIDs: []string{"s0001", "t1"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false), DuplicateOf: "s0001"}
	n, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{d, {
		ID: "s0002", Text: "Search", SourceIDs: []string{"s0002"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false)}}}, ls, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if n.byID("s0001") == nil {
		t.Fatal("mapped target was consumed and deleted")
	}
}

// A losing merged group retires every source it consumed.
func TestLosingGroupRetiresAllSources(t *testing.T) {
	s := &strategySet{NextID: 3, Items: []strategyItem{
		{ID: "s0001", Text: "A", TotalScore: 2}, {ID: "s0002", Text: "B", TotalScore: 1}, {ID: "s0003", Text: "C", TotalScore: 9}}}
	a := llmItem{ID: "s0001", Text: "AB", SourceIDs: []string{"s0001", "s0002"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false), Contradicts: []string{"s0003"}}
	n, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{a, {
		ID: "s0003", Text: "C", SourceIDs: []string{"s0003"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false)}}}, nil, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if n.hasID("s0002") {
		t.Fatal("a source of the rejected merged group still survives")
	}
}

// Recovery must drop an uncommitted trailing half-line BEFORE reading.
func TestEvaluationRecoversPartialTail(t *testing.T) {
	o, err := openOutputDir(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer o.release()
	if err := os.WriteFile(o.path("eval_state.jsonl"), []byte(`{"eval_id":"job","qid":`), artifactPerm); err != nil {
		t.Fatal(err)
	}
	cp := &checkpoint{Phase: phaseEvaluate, Pass: 1}
	cp.Eval.EvalID = "job"
	e := &executor{out: o}
	if _, err := e.evaluatePass(context.Background(), cp, newStrategySet(), []trainsetRecord{{ID: "q1"}}); err != nil {
		t.Fatalf("uncommitted partial tail blocked recovery: %v", err)
	}
}

// Two lessons from the same question mapping to the same strategy count once.
func TestSameQuestionPredictionsCountOnce(t *testing.T) {
	s := &strategySet{NextID: 1, Items: []strategyItem{{ID: "s0001", Text: "Verify", TotalScore: 5}}}
	ls := []lesson{{TentativeID: "t1", QID: "q1", Text: "Verify", Score: 2}, {TentativeID: "t2", QID: "q1", Text: "Verify", Score: 2}}
	a := llmItem{Text: "Verify", SourceIDs: []string{"t1"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false), DuplicateOf: "s0001"}
	b := llmItem{Text: "Verify", SourceIDs: []string{"t2"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false), DuplicateOf: "s0001"}
	keep := llmItem{ID: "s0001", Text: "Verify", SourceIDs: []string{"s0001"}, Terse: boolPtr(true), CorpusSpecific: boolPtr(false)}
	n, _, err := applyMerge(s, llmMergeOutput{Items: []llmItem{keep, a, b}}, ls, 10, 30)
	if err != nil {
		t.Fatal(err)
	}
	if got := n.byID("s0001"); got == nil || got.TotalScore != 7 {
		t.Fatalf("same-question predictions counted twice: %+v", n.Items)
	}
}

// An all-rejected merge must discard this batch's accumulated ratings, not commit them.
func TestEmptyFallbackDiscardsBatchScores(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/chat/completions" {
			fmt.Fprint(w, `{"code":0,"data":{"id":"session"}}`)
			return
		}
		var b map[string]any
		_ = json.NewDecoder(r.Body).Decode(&b)
		ans := "wrong"
		switch b["chat_id"] {
		case "judge":
			ans = "correct: no"
		case "answer":
			if q, _ := b["question"].(string); len(q) > 0 && q[0] == 'Y' {
				ans = `{"ratings":[{"strategy_id":"s0001","score":-2}],"lessons":[{"text":"Verify","score":1}]}`
			}
		case "merge":
			ans = `{"items":[],"rejected":[{"source_id":"s0001","reason":"drop"},{"source_id":"t1.0.0.0","reason":"drop"}]}`
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"code": 0, "data": map[string]any{"answer": ans}})
	}))
	defer srv.Close()
	cfg := defaultConfig()
	cfg.OutputDir = t.TempDir()
	cfg.ChatID, cfg.JudgeChatID, cfg.MergeChatID = "answer", "judge", "merge"
	cfg.MaxPasses, cfg.BatchSize = 1, 1
	o, err := openOutputDir(cfg.OutputDir)
	if err != nil {
		t.Fatal(err)
	}
	cp := &checkpoint{Phase: phaseTrain, Pass: 1, Seed: cfg.Seed, NextID: 1, Items: []strategyItem{{ID: "s0001", Text: "Verify", TotalScore: 5}}}
	if err := o.writeCheckpoint(cp); err != nil {
		t.Fatal(err)
	}
	o.release()
	e := &executor{cfg: cfg, client: &httpClient{base: srv.URL, hc: srv.Client()}, quiet: true}
	if err := e.runTraining(context.Background(), []trainsetRecord{{ID: "q1", Question: "q", GoldAnswer: "a"}}, trainsetMeta{}); err != nil {
		t.Fatal(err)
	}
	got, _, err := loadCheckpoint(cfg.OutputDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Items) == 0 || got.Items[0].TotalScore != 5 {
		t.Fatalf("fallback committed this batch's scores: %+v version=%d", got.Items, got.Version)
	}
}

// ---------------------------------------------------------------- round-4 regression

// A merge that only changes scores (structure unchanged) IS applied; withBase must not
// discard this batch's ratings.
func TestSuccessfulMappingPreservesUsedScore(t *testing.T) {
	base := &strategySet{NextID: 1, Items: []strategyItem{{ID: "s0001", Text: "Verify clues", TotalScore: 5}}}
	base.rank(10)
	next := cloneSet(base)
	next.addScore("s0001", -2)
	ls := []lesson{{TentativeID: "t1", QID: "q1", Text: "Verify clues", Score: 2}}
	d := llmItem{ID: "s0001", Text: "Verify clues", SourceIDs: []string{"s0001", "t1"}, DuplicateOf: "s0001", Terse: boolPtr(true), CorpusSpecific: boolPtr(false)}
	got, _, err := applyMerge(next, llmMergeOutput{Items: []llmItem{d}}, ls, 10, 30,
		withBase(base), withUsedPairs(map[string]bool{"q1|s0001": true}))
	if err != nil {
		t.Fatal(err)
	}
	if g := got.byID("s0001"); g == nil || g.TotalScore != 3 || got.Version != 1 {
		t.Fatalf("successful mapping discarded the used rating: score=%v version=%d", g, got.Version)
	}
}

// dedupePool survivors must be pairwise distinct, independent of input order.
func TestNearDuplicateSurvivorsArePairwiseDistinct(t *testing.T) {
	a := "a b c d e f g h i j"
	b := "a b c d e f g h i k"
	c := "a b c d e f g h k l"
	items := []strategyItem{{ID: "s0001", Text: a, TotalScore: 1}, {ID: "s0002", Text: c, TotalScore: 2}, {ID: "s0003", Text: b, TotalScore: 3}}
	got, _ := dedupePool(items)
	for i := range got {
		for j := 0; j < i; j++ {
			if nearDuplicate(got[i].Text, got[j].Text) {
				t.Fatalf("near duplicates survived: %+v", got)
			}
		}
	}
}

// An equal-score duplicate is resolved by the stable id, not by input order.
func TestEqualScoreDuplicateUsesStableID(t *testing.T) {
	got, _ := dedupePool([]strategyItem{{ID: "s0002", Text: "Verify clues", TotalScore: 5}, {ID: "s0001", Text: "Verify clues", TotalScore: 5}})
	if len(got) != 1 || got[0].ID != "s0001" {
		t.Fatalf("tie resolved by iteration order: %+v", got)
	}
}

// Model-side temporary failures wait a FLAT 10 min (sliding-window allowance); transport
// hiccups keep the fast exponential backoff.
func TestRetryDelayByClass(t *testing.T) {
	p := retryPolicy{initial: time.Second, max: 60 * time.Second, quota: 600 * time.Second}
	if got := p.delayFor(failQuota, 0); got != 600*time.Second {
		t.Fatalf("quota wait = %s, want 10m", got)
	}
	if got := p.delayFor(failQuota, 7); got != 600*time.Second {
		t.Fatalf("quota wait must stay flat, got %s", got)
	}
	if got := p.delayFor(failNetwork, 0); got != time.Second {
		t.Fatalf("network initial = %s, want 1s", got)
	}
	if got := p.delayFor(failNetwork, 9); got != 60*time.Second {
		t.Fatalf("network cap = %s, want 60s", got)
	}
	if got := (retryPolicy{}).delayFor(failQuota, 0); got != 10*time.Minute {
		t.Fatalf("default quota wait = %s, want 10m", got)
	}
	// The exact real-world provider string must classify as quota.
	const highDemand = "**ERROR**: codexagent: turn failed (kind=internalServerError): We’re currently experiencing high demand, which may cause temporary errors."
	if !answerLooksLikeError(highDemand) {
		t.Fatal("the real high-demand error text must be detected")
	}
	if classify(errorsNew(highDemand)) != failQuota {
		t.Fatal("high demand must classify as quota")
	}
}

// TestProbeRetryClass pins the preflight failure policy: a provider-errored probe reached
// Codex and must be waited out, while "no nonce rollout" is degradation and must abort.
func TestProbeRetryClass(t *testing.T) {
	const highDemand = "**ERROR**: codexagent: turn failed (kind=internalServerError): We’re currently experiencing high demand, which may cause temporary errors."
	// A provider error answer means the request reached Codex -> retry on the quota interval.
	if c, ok := probeRetryClass(&probeProviderError{answer: highDemand}); !ok || c != failQuota {
		t.Fatalf("provider error: got (%v,%v), want (quota,true)", c, ok)
	}
	// No rollout at all -> engine degraded -> abort.
	if _, ok := probeRetryClass(errNoProbeRollout); ok {
		t.Fatal("errNoProbeRollout must abort, not retry")
	}
	// Transport-side transient errors from ask() -> retry.
	if c, ok := probeRetryClass(&httpStatusError{Status: 429}); !ok || c != failQuota {
		t.Fatalf("429: got (%v,%v), want (quota,true)", c, ok)
	}
	if c, ok := probeRetryClass(&httpStatusError{Status: 503}); !ok || c != failNetwork {
		t.Fatalf("503: got (%v,%v), want (network,true)", c, ok)
	}
	// Permanent 4xx -> abort.
	if _, ok := probeRetryClass(&httpStatusError{Status: 400}); ok {
		t.Fatal("400 must abort, not retry")
	}
}

// ---------------------------------------------------------------- helpers

func errorsNew(s string) error { return fmt.Errorf("%s", s) }

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func lens(b [][]int) []int {
	var out []int
	for _, x := range b {
		out = append(out, len(x))
	}
	return out
}
