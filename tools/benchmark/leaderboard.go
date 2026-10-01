package main

// Judge scoring, verdict parsing and the leaderboard builder (port of
// parse_grader_verdict, _judge_one, _merge_judgements, _usage_row,
// calibration_error, build_leaderboard and _leaderboard_summary).

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Grader verdict parsing (port of parse_grader_verdict)
// ---------------------------------------------------------------------------

// Patterns mirror parse_judge_response() in the reference evaluation script:
// a model may or may not bold the label, so each field is tried in its bold,
// colon-outside-bold, and plain forms before giving up. Go's RE2 has no
// lookahead, so the "up to end of line" semantics become (?m) + `.*`.
var (
	graderAnswerPatterns = []string{
		`(?im)\*\*extracted_final_answer:\*\*[ \t]*(.*)`,
		`(?im)\*\*extracted_final_answer\*\*:[ \t]*(.*)`,
		`(?im)extracted_final_answer:[ \t]*(.*)`,
	}
	graderCorrectPatterns = []string{
		`(?i)\*\*correct:\*\*\s*(yes|no)`,
		`(?i)\*\*correct\*\*:\s*(yes|no)`,
		`(?i)correct:\s*(yes|no)`,
	}
	graderConfidencePatterns = []string{
		`(?i)\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?`,
		`(?i)\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?`,
		`(?i)confidence:\s*(\d+(?:\.\d+)?)\s*%?`,
	}
)

var graderReasoningLabel = regexp.MustCompile(`(?is)\*\*reasoning:\*\*|\*\*reasoning\*\*:|reasoning:`)
var graderReasoningStop = regexp.MustCompile(`(?is)\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:`)

// graderField ports _grader_field for the line-scoped fields.
func graderField(text string, patterns []string) string {
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		if m := re.FindStringSubmatch(text); m != nil {
			return strings.TrimSpace(m[1])
		}
	}
	return ""
}

// graderReasoning ports the reasoning field: from the reasoning label to the
// next correct: line (or the end of the text).
func graderReasoning(text string) string {
	loc := graderReasoningLabel.FindStringIndex(text)
	if loc == nil {
		return ""
	}
	rest := text[loc[1]:]
	if stop := graderReasoningStop.FindStringIndex(rest); stop != nil {
		rest = rest[:stop[0]]
	}
	return strings.TrimSpace(rest)
}

// parseGraderVerdict ports parse_grader_verdict. Raises (returns an error)
// when `correct` is missing: the leaderboard's accuracy is a boolean and a
// judge reply without one is a failed judgement, never an implicit "no".
func parseGraderVerdict(text string) (map[string]any, error) {
	cleaned := thinkRE.ReplaceAllString(text, "")
	rawCorrect := graderField(cleaned, graderCorrectPatterns)
	if rawCorrect == "" {
		snippet := cleaned
		if len(snippet) > 500 {
			snippet = snippet[:500]
		}
		return nil, fmt.Errorf("could not parse judge correctness from: %s", snippet)
	}
	// The reference script reads the confidence the ANSWER states and defaults
	// to 100 when there is none, so an agent that never states one is scored
	// as maximally confident - the calibration error then equals its miss rate.
	confidence := 100.0
	if raw := graderField(cleaned, graderConfidencePatterns); raw != "" {
		if f := asFloat(raw); f != nil && *f < 100.0 {
			confidence = *f
		}
	}
	return map[string]any{
		"correct":                strings.EqualFold(rawCorrect, "yes"),
		"confidence":             confidence,
		"extracted_final_answer": nilIfEmpty(graderField(cleaned, graderAnswerPatterns)),
		"reasoning":              nilIfEmpty(graderReasoning(cleaned)),
	}, nil
}

func nilIfEmpty(s string) any {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return s
}

// judgeOne ports _judge_one: run the judge for one answer row and shape the
// judgement record.
func judgeOne(c *httpClient, cfg map[string]any, row map[string]any) map[string]any {
	record := map[string]any{
		"correct":                nil,
		"confidence":             nil,
		"extracted_final_answer": nil,
		"verdict":                nil,
		"judge_error":            "",
	}
	if strings.TrimSpace(strAny(row, "ragflow_error", "")) != "" {
		record["judge_error"] = "excluded_due_to_ragflow_error"
	} else {
		judgeAnswer := askJudgeWithRetry(c, cfg, row)
		verdict, err := parseGraderVerdict(judgeAnswer)
		if err != nil {
			record["judge_error"] = err.Error()
		} else {
			record["verdict"] = verdict
			record["correct"] = verdict["correct"]
			record["confidence"] = verdict["confidence"]
			record["extracted_final_answer"] = verdict["extracted_final_answer"]
		}
	}
	return record
}

// isBackendErrorVerdict ports _is_backend_error_verdict: True when the judge
// never produced a verdict (backend error bubble or unparseable verdict).
func isBackendErrorVerdict(record map[string]any) bool {
	if strings.HasPrefix(strings.TrimLeft(strAny(record, "judge_answer", ""), " \t\n"), "**ERROR**") {
		return true
	}
	return asFloat(record["accuracy"]) == nil && strings.TrimSpace(strAny(record, "judge_error", "")) != ""
}

// readJudgements ports _read_judgements: the per_query_judgements already
// stored in leaderboard.json, keyed by run key.
func readJudgements(leaderboardPath string) map[string]map[string]any {
	if !fileExists(leaderboardPath) {
		return map[string]map[string]any{}
	}
	data, err := os.ReadFile(leaderboardPath)
	if err != nil {
		fmt.Printf("[judge] could not read %s: %v\n", leaderboardPath, err)
		return map[string]map[string]any{}
	}
	var leaderboard map[string]any
	if err := json.Unmarshal(data, &leaderboard); err != nil {
		fmt.Printf("[judge] could not read %s: %v\n", leaderboardPath, err)
		return map[string]map[string]any{}
	}
	records, _ := leaderboard["per_query_judgements"].([]any)
	out := map[string]map[string]any{}
	for _, recordAny := range records {
		record, ok := recordAny.(map[string]any)
		if !ok {
			continue
		}
		if key := strAny(record, "run_key", ""); key != "" {
			out[key] = record
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Merge + leaderboard build
// ---------------------------------------------------------------------------

// mergeJudgements ports _merge_judgements: attach each row's judge verdict to
// the answer row (in memory) so the leaderboard builder can score it.
func mergeJudgements(rows []map[string]any, judgements map[string]map[string]any) []map[string]any {
	var merged []map[string]any
	for _, row := range rows {
		out := map[string]any{}
		for k, v := range row {
			out[k] = v
		}
		if record, ok := judgements[rowRunKey(row)]; ok {
			out["judge_correct"] = record["correct"]
			out["judge_confidence"] = record["confidence"]
			out["judge_error"] = strAny(record, "judge_error", "")
			out["judge_extracted_answer"] = record["extracted_final_answer"]
			out["judge_parsed"] = record["verdict"]
		}
		merged = append(merged, out)
	}
	return merged
}

// leaderboardWithJudgements ports _leaderboard_with_judgements.
func leaderboardWithJudgements(rows []map[string]any, judgements map[string]map[string]any, cfg map[string]any) map[string]any {
	return buildLeaderboard(mergeJudgements(rows, judgements), cfg, nil)
}

// leaderboardCorrect ports _leaderboard_correct: the binary verdict of a row,
// or nil when the row was never judged (and counts as incorrect).
func leaderboardCorrect(row map[string]any) *bool {
	if v, ok := row["judge_correct"]; ok && v != nil {
		b := boolOf(v)
		return &b
	}
	return nil
}

// confidencePercent ports _confidence_percent: normalize a judge confidence
// onto the reference script's 0-100 scale. Legacy values at or below 1 are
// read as fractions of 1.
func confidencePercent(v any) *float64 {
	numeric := asFloat(v)
	if numeric == nil {
		return nil
	}
	out := *numeric
	if out >= 0 && out <= 1 {
		out *= 100
	}
	if out > 100 {
		out = 100
	}
	return &out
}

func searchToolsOf(lbCfg map[string]any) []string {
	if list, ok := lbCfg["search_tools"].([]any); ok {
		var tools []string
		for _, name := range list {
			if s := strings.TrimSpace(fmt.Sprintf("%v", name)); s != "" {
				tools = append(tools, s)
			}
		}
		if len(tools) > 0 {
			return tools
		}
	}
	return []string{"search_chunks", "search_semantic_chunks", "grep_chunks", "search_bm25_chunks"}
}

// calibrationMinSamples: the reference script needs at least this many
// scored confidences, otherwise the calibration error is reported as 0.0.
const calibrationMinSamples = 100

func calibrationBinSizeOf(lbCfg map[string]any) int {
	return asIntDefault(lbCfg["calibration_bin_size"], 100)
}

// calibrationError ports calibration_error, line for line from the reference
// script's calib_err(). The loop is `range(len(bins) - 1)`, so the final bin
// is never scored - a bug in the reference script, reproduced on purpose to
// stay comparable with the published leaderboard entries. One difference is
// unavoidable and deliberate: this port sorts STABLY, where the reference's
// numpy argsort permutes confidence ties arbitrarily.
func calibrationError(confidences []float64, correctness []bool, beta int) float64 {
	if len(confidences) == 0 || len(confidences) != len(correctness) {
		return 0
	}
	type pair struct {
		conf float64
		ok   float64
	}
	pairs := make([]pair, len(confidences))
	for i, c := range confidences {
		pairs[i] = pair{c / 100.0, boolToFloat(correctness[i])}
	}
	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].conf < pairs[j].conf })
	total := len(pairs)
	if beta <= 0 {
		beta = 100
	}
	binCount := total / beta
	if binCount == 0 {
		return 0
	}
	cerr := 0.0
	for i := 0; i < binCount-1; i++ {
		lo, hi := i*beta, (i+1)*beta
		size := hi - lo
		if size <= 0 {
			continue
		}
		sumConf, sumOK := 0.0, 0.0
		for _, p := range pairs[lo:hi] {
			sumConf += p.conf
			sumOK += p.ok
		}
		difference := math.Abs(sumConf/float64(size) - sumOK/float64(size))
		cerr += float64(size) / float64(total) * difference * difference
	}
	return math.Sqrt(cerr) * 100
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// buildLeaderboard ports build_leaderboard: the JSON the BrowseComp-Plus
// leaderboard asks for. Every formula mirrors the reference script, including
// its denominators: Accuracy divides by ALL questions of the run (a question
// the agent failed on counts as incorrect); Recall averages only over
// questions that have evidence documents; calibration needs >= 100 scored
// confidences.
func buildLeaderboard(rows []map[string]any, cfg map[string]any, evidenceByID map[string][]string) map[string]any {
	lbCfg, _ := cfg["leaderboard"].(map[string]any)
	if lbCfg == nil {
		lbCfg = map[string]any{}
	}
	searchTools := searchToolsOf(lbCfg)
	beta := calibrationBinSizeOf(lbCfg)

	total := len(rows)
	var perQueryMetrics []map[string]any
	var perQueryUsage []map[string]any
	var perQueryStructure []map[string]any
	var confidences []float64
	var correctness []bool
	var recalls []float64
	toolTotals := map[string]float64{}
	toolErrorTotals := map[string]float64{}
	toolErrorSamples := map[string]string{}
	statsMissing := 0
	judgedCount := 0
	var perQueryJudgements []map[string]any
	structureSeen := map[string]int{}
	failureTypeCounts := map[string]int{}
	earlyStopHits, earlyStopRows := 0, 0
	intermediateGrounded, intermediateChecked := 0, 0
	var corpusDir string
	if dataset, _ := cfg["dataset"].(map[string]any); dataset != nil {
		corpusDir = strAny(dataset, "corpus_path", "")
	}
	docTextCache := map[string]string{}
	deliveryTotals := map[string]int{}
	inlineRows, pointerRows, retiredRows, pivotRows := 0, 0, 0, 0
	unrecordedRows, partitionRows, reuseRows, leakRows, decompositionRows := 0, 0, 0, 0, 0
	promptSigs := promptSignatures("")
	signatureTotal := len(promptSigs["docnames"].(map[string]bool)) + len(promptSigs["values"].(map[string]bool))

	for _, row := range rows {
		queryID := strAny(row, "question_id", rowRunKey(row))
		correctRaw := leaderboardCorrect(row)
		// Anything the judge did not affirm is a miss: the reference script
		// counts a result whose judge_result.correct is absent (error, empty
		// answer, unparseable verdict) as not correct.
		correct := correctRaw != nil && *correctRaw
		if correctRaw != nil {
			judgedCount++
			if confidence := confidencePercent(row["judge_confidence"]); confidence != nil {
				confidences = append(confidences, *confidence)
				correctness = append(correctness, correct)
			}
		}

		evidence := uniqueDocIDs(toAnySlice(firstNonEmptySlice(evidenceByID[queryID], asDocIDList(row["evidence_doc_ids"]))))
		recall := (*float64)(nil)
		if len(evidence) > 0 {
			retrievedSet := map[string]bool{}
			for _, d := range asDocIDList(row["retrieved_docids"]) {
				retrievedSet[d] = true
			}
			hits := 0
			for _, d := range evidence {
				if retrievedSet[d] {
					hits++
				}
			}
			r := float64(hits) / float64(len(evidence))
			recall = &r
			recalls = append(recalls, r)
		}
		perQueryMetrics = append(perQueryMetrics, map[string]any{
			"query_id": queryID,
			"correct":  correct,
			"recall":   roundPercentNil(recall),
		})

		if row["judge_correct"] != nil || row["judge_error"] != nil {
			perQueryJudgements = append(perQueryJudgements, map[string]any{
				"query_id":               queryID,
				"run_key":                rowRunKey(row),
				"correct":                correctRaw,
				"confidence":             confidencePercent(row["judge_confidence"]),
				"extracted_final_answer": row["judge_extracted_answer"],
				"verdict":                row["judge_parsed"],
				"judge_error":            strAny(row, "judge_error", ""),
			})
		}

		if counts, ok := row["tool_call_counts"].(map[string]any); ok {
			for name, count := range counts {
				if numeric := asFloat(count); numeric != nil {
					toolTotals[name] += *numeric
				}
			}
		} else if strings.TrimSpace(strAny(row, "ragflow_error", "")) == "" {
			// A row the backend never reported stats for is not "zero calls":
			// streaming runs and pre-instrumentation backends never see them.
			statsMissing++
		}
		if rowErrors, ok := row["tool_call_errors"].(map[string]any); ok {
			for name, count := range rowErrors {
				if numeric := asFloat(count); numeric != nil {
					toolErrorTotals[name] += *numeric
				}
			}
		}
		if rowSamples, ok := row["tool_error_samples"].(map[string]any); ok {
			for name, text := range rowSamples {
				if _, exists := toolErrorSamples[name]; !exists {
					toolErrorSamples[name] = fmt.Sprintf("%v", text)
				}
			}
		}

		perQueryUsage = append(perQueryUsage, usageRow(queryID, row, searchTools))

		// Structure: what the deliverable DECLARED, counted rather than judged.
		structure := structureMetrics(strAny(row, "ragflow_answer", ""), strAny(row, "question", ""), promptSigs)
		committedValues, _ := structure["values"].([]string)
		failureTypes, _ := structure["failure_types"].([]string)
		delete(structure, "values")
		delete(structure, "failure_types")
		entry := map[string]any{"query_id": queryID, "run_key": rowRunKey(row)}
		for k, v := range structure {
			entry[k] = v
		}
		perQueryStructure = append(perQueryStructure, entry)
		for _, field := range structureFields {
			if truthy(structure[field]) {
				structureSeen[field]++
			}
		}
		for _, ft := range failureTypes {
			failureTypeCounts[ft]++
		}
		if truthy(structure["inlined_candidate"]) {
			inlineRows++
		}
		if truthy(structure["from_entries_without_pointer"]) || truthy(structure["from_pointer_mismatch"]) {
			pointerRows++
		}
		if truthy(structure["depends_on_written"]) {
			retiredRows++
		}
		if truthy(structure["pivot_missing_blocks"]) {
			pivotRows++
		}
		if truthy(structure["blocks_without_searched"]) {
			unrecordedRows++
		}
		if truthy(structure["constraints_duplicated"]) {
			partitionRows++
		}
		if truthy(structure["constraint_number_reused"]) {
			reuseRows++
		}
		if truthy(structure["leaked_docnames"]) || truthy(structure["leaked_values"]) || truthy(structure["example_value_shipped"]) {
			leakRows++
		}
		if truthy(structure["decomposition_detached_blocks"]) || truthy(structure["decomposition_orphan_roots"]) || truthy(structure["decomposition_prose_block_refs"]) {
			decompositionRows++
		}
		for _, key := range deliveryTotalsFields {
			deliveryTotals[key] += asIntDefault(structure[key], 0)
		}

		// Early stop: a run that concluded inside a handful of retrieved
		// documents. Symptom, not score.
		if row["retrieved_docids"] != nil {
			earlyStopRows++
			if len(uniqueDocIDs(row["retrieved_docids"])) <= earlyStopDocs {
				earlyStopHits++
			}
		}

		// Intermediate nodes: did the entities the deliverable committed to
		// mid-chain come from the documents it actually retrieved?
		retrievedNow := uniqueDocIDs(row["retrieved_docids"])
		grounded, checked := groundValues(committedValues, retrievedNow, corpusDir, docTextCache)
		intermediateGrounded += grounded
		intermediateChecked += checked
	}

	correctCount := 0
	for _, entry := range perQueryMetrics {
		if boolOf(entry["correct"]) {
			correctCount++
		}
	}
	accuracyPercent := 0.0
	if total > 0 {
		accuracyPercent = round2(float64(correctCount) / float64(total) * 100)
	}
	judgedTotal, judgedOK := 0, 0
	for _, row := range rows {
		if row["judge_correct"] != nil {
			judgedTotal++
			if boolOf(row["judge_correct"]) {
				judgedOK++
			}
		}
	}
	recallPercent := any(nil)
	if len(recalls) > 0 {
		sum := 0.0
		for _, r := range recalls {
			sum += r
		}
		recallPercent = round2(sum / float64(len(recalls)) * 100)
	}

	avgToolStats := map[string]float64{}
	if total > 0 {
		for name, count := range toolTotals {
			avgToolStats[name] = count / float64(total)
		}
	}
	searchCalls := 0.0
	for _, name := range searchTools {
		searchCalls += avgToolStats[name]
	}

	calibrationErrorPercent := 0.0
	calibrationNote := fmt.Sprintf("not computed: only %d scored confidence(s), need %d", len(confidences), calibrationMinSamples)
	if len(confidences) >= calibrationMinSamples {
		calibrationErrorPercent = round2(calibrationError(confidences, correctness, beta))
		calibrationNote = fmt.Sprintf("%d scored confidence(s), bin size %d", len(confidences), beta)
	} else {
		fmt.Printf("[leaderboard] calibration error %s\n", calibrationNote)
	}

	structureAdoption := map[string]float64{}
	for _, field := range structureFields {
		if total > 0 {
			structureAdoption[field] = round1(float64(structureSeen[field]) / float64(total) * 100)
		}
	}
	intermediateNote := "not computed: dataset.corpus_path is not configured, so no corpus text can back the check."
	if corpusDir != "" {
		intermediateNote = "A committed value counts as grounded when every distinctive token of it (>=4 chars) occurs in the union of the documents the run retrieved: the offline reading of whether the mid-chain entities the deliverable committed to came from the corpus it read. Values with no distinctive token are skipped, and a value that is grounded may still be the wrong one."
	}

	return map[string]any{
		"LLM":                   orDefault(strAny(lbCfg, "llm", ""), "change me when submitting"),
		"Retriever":             orDefault(strAny(lbCfg, "retriever", ""), "change me when submitting"),
		"Accuracy (%)":          accuracyPercent,
		"Recall (%)":            recallPercent,
		"Search Calls":          round2(searchCalls),
		"Calibration Error (%)": calibrationErrorPercent,
		"Link":                  orDefault(strAny(lbCfg, "link", ""), "change me when submitting"),
		"Evaluation Date":       orDefault(strAny(lbCfg, "evaluation_date", ""), timeNowDate()),
		"avg_tool_stats":        sortedCopy(avgToolStats),
		"tool_error_stats":      sortedCopyF(toolErrorTotals),
		"tool_error_samples":    sortedCopyS(toolErrorSamples),
		"per_query_metrics":     perQueryMetrics,
		"per_query_usage":       perQueryUsage,
		"per_query_judgements":  perQueryJudgements,
		"per_query_structure":   perQueryStructure,
		"_diagnostics":          diagnosticsBlock(total, judgedCount, judgedOK, len(recalls), searchTools, calibrationNote, statsMissing, structureAdoption, failureTypeCounts, earlyStopHits, earlyStopRows, intermediateGrounded, intermediateChecked, intermediateNote, deliveryTotals, inlineRows, pointerRows, retiredRows, pivotRows, unrecordedRows, partitionRows, reuseRows, leakRows, decompositionRows, signatureTotal, promptSigs),
	}
}

func sortedCopy(m map[string]float64) map[string]float64  { return m }
func sortedCopyF(m map[string]float64) map[string]float64 { return m }
func sortedCopyS(m map[string]string) map[string]string   { return m }

func orDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func truthy(v any) bool {
	switch x := v.(type) {
	case int:
		return x != 0
	case float64:
		return x != 0
	case bool:
		return x
	case string:
		return x != ""
	case []string:
		return len(x) > 0
	case nil:
		return false
	default:
		return false
	}
}

func timeNowDate() string { return timeNow().Format("2006-01-02") }

func roundPercentNil(v *float64) any {
	if v == nil {
		return nil
	}
	return round2(*v * 100)
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }
func round3(v float64) float64 { return math.Round(v*1000) / 1000 }

// leaderboardSummary ports _leaderboard_summary: one-line-per-metric digest
// of the submission.
func leaderboardSummary(leaderboard map[string]any) string {
	usage, _ := leaderboard["per_query_usage"].([]map[string]any)
	inputTokens, outputTokens := 0, 0
	clientSeconds := 0.0
	deepChunks, shallowChunks := 0, 0
	for _, row := range usage {
		inputTokens += asIntDefault(row["input_tokens"], 0)
		outputTokens += asIntDefault(row["output_tokens"], 0)
		if f := asFloat(row["client_seconds"]); f != nil {
			clientSeconds += *f
		}
		deepChunks += asIntDefault(row["deep_read_chunks"], 0)
		shallowChunks += asIntDefault(row["shallow_read_chunks"], 0)
	}
	judgements, _ := leaderboard["per_query_judgements"].([]map[string]any)
	scored, judgedOK := 0, 0
	for _, j := range judgements {
		if j["correct"] != nil {
			scored++
			if boolOf(j["correct"]) {
				judgedOK++
			}
		}
	}
	unjudged := len(judgements) - scored
	judgedOnly := "n/a"
	if scored > 0 {
		judgedOnly = fmt.Sprintf("%.2f%%", float64(judgedOK)/float64(scored)*100)
	}
	lines := []string{
		"leaderboard:",
		fmt.Sprintf("  LLM              : %v", leaderboard["LLM"]),
		fmt.Sprintf("  Retriever        : %v", leaderboard["Retriever"]),
		fmt.Sprintf("  Accuracy (%%)     : %v", leaderboard["Accuracy (%)"]),
	}
	if scored > 0 {
		lines = append(lines, fmt.Sprintf("  Accuracy (judged): %s  (%d/%d scored rows; %d unjudged counted incorrect)", judgedOnly, judgedOK, scored, unjudged))
	} else {
		lines = append(lines, "  Accuracy (judged): "+judgedOnly)
	}
	lines = append(lines,
		fmt.Sprintf("  Recall (%%)       : %v", leaderboard["Recall (%)"]),
		fmt.Sprintf("  Search Calls     : %v", leaderboard["Search Calls"]),
		fmt.Sprintf("  Calibration (%%)  : %v", leaderboard["Calibration Error (%)"]),
		fmt.Sprintf("  avg_tool_stats   : %v", leaderboard["avg_tool_stats"]),
		fmt.Sprintf("  tool_errors      : %v", orNonNil(leaderboard["tool_error_stats"], "none")),
		fmt.Sprintf("  chunk reads      : deep=%d shallow=%d", deepChunks, shallowChunks),
		fmt.Sprintf("  tokens           : input=%d output=%d", inputTokens, outputTokens),
		fmt.Sprintf("  wall clock (s)   : %.1f", clientSeconds),
	)
	if samples, ok := leaderboard["tool_error_samples"].(map[string]string); ok {
		for name, text := range samples {
			lines = append(lines, fmt.Sprintf("  tool_error[%s]: %s", name, text))
		}
	}
	return strings.Join(lines, "\n")
}

func orNonNil(v any, def string) string {
	if v == nil {
		return def
	}
	return fmt.Sprintf("%v", v)
}
