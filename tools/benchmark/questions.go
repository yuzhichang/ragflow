package main

// Question loading, normalisation and selection (port of load_questions,
// _normalize_question, select_questions, load_evidence_doc_ids and the id
// helpers).

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Candidate keys per logical field, first hit wins. Keeps dataset-specific
// naming out of the code: add a key here instead of a new loader.
var (
	idKeys            = []string{"id", "question_id", "qid"}
	questionKeys      = []string{"question", "query", "prompt"}
	goldKeys          = []string{"gold_answer", "answer", "reference_answer", "gold"}
	reasoningTypeKeys = []string{"reasoning_types", "reasoning_type", "types"}
	expectedDocKeys   = []string{"expected_doc_ids", "expected_sources"}
	// The leaderboard scores RECALL against the qrel "evidence" set - every
	// document labelled as needed to answer the query - which is a superset of
	// the gold set. BrowseComp-Plus ships both as expected_doc_ids /
	// evidence_doc_ids, so they are read separately: the diagnostic report
	// keeps using expected_doc_ids, leaderboard.json uses evidence_doc_ids.
	evidenceDocKeys = []string{"evidence_doc_ids", "expected_evidence_doc_ids", "qrel_evidence"}
)

type question struct {
	questionID     string
	question       string
	goldAnswer     string
	reasoningTypes []string
	expectedDocs   []string
	evidenceDocs   []string
}

// loadQuestions ports load_questions: .jsonl, JSON list, or {id: {...}} map.
func loadQuestions(path string) ([]*question, error) {
	if strings.EqualFold(filepath.Ext(path), ".jsonl") {
		return loadQuestionsJSONL(path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("invalid JSON in %s: %w", path, err)
	}
	switch payload := raw.(type) {
	case []any:
		out := make([]*question, 0, len(payload))
		for i, row := range payload {
			m, ok := row.(map[string]any)
			if !ok {
				return nil, fmt.Errorf("question row %d must be a JSON object", i+1)
			}
			out = append(out, normalizeQuestion(m, i+1))
		}
		return out, nil
	case map[string]any:
		if list, ok := payload["questions"].([]any); ok {
			out := make([]*question, 0, len(list))
			for i, row := range list {
				m, ok := row.(map[string]any)
				if !ok {
					return nil, fmt.Errorf("question row %d must be a JSON object", i+1)
				}
				out = append(out, normalizeQuestion(m, i+1))
			}
			return out, nil
		}
		ids := make([]string, 0, len(payload))
		for id := range payload {
			ids = append(ids, id)
		}
		sortIDs(ids)
		out := make([]*question, 0, len(ids))
		for _, id := range ids {
			m, ok := payload[id].(map[string]any)
			if !ok {
				return nil, fmt.Errorf("question row %s must be a JSON object", id)
			}
			out = append(out, normalizeQuestion(m, id))
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported question file format: %s", path)
	}
}

func loadQuestionsJSONL(path string) ([]*question, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []*question
	for i, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var payload map[string]any
		if err := json.Unmarshal([]byte(line), &payload); err != nil {
			return nil, fmt.Errorf("invalid JSONL at %s:%d: %w", path, i+1, err)
		}
		out = append(out, normalizeQuestion(payload, i+1))
	}
	return out, nil
}

// loadEvidenceDocIDs ports load_evidence_doc_ids: question_id -> evidence
// document ids, read from the questions file. Returns an empty mapping (with
// a warning) when the file cannot be read: the leaderboard export then falls
// back to whatever the answer rows carry.
func loadEvidenceDocIDs(path string) map[string][]string {
	questions, err := loadQuestions(path)
	if err != nil {
		fmt.Printf("[leaderboard] could not read evidence documents from %s: %v\n", path, err)
		return map[string][]string{}
	}
	out := make(map[string][]string, len(questions))
	for _, q := range questions {
		out[q.questionID] = uniqueDocIDs(toAnySlice(q.evidenceDocs))
	}
	return out
}

func firstValue(payload map[string]any, keys []string) any {
	for _, key := range keys {
		if v, ok := payload[key]; ok && v != nil && v != "" {
			return v
		}
	}
	return nil
}

func firstDocList(payload map[string]any, keys []string) []string {
	for _, key := range keys {
		if docs := asDocIDList(payload[key]); len(docs) > 0 {
			return docs
		}
	}
	return nil
}

func normalizeQuestion(payload map[string]any, fallbackID any) *question {
	return &question{
		questionID:     strAny(payload, "question_id", firstNonEmpty(idValue(payload, idKeys), fmt.Sprintf("%v", fallbackID))),
		question:       idValue(payload, questionKeys),
		goldAnswer:     idValue(payload, goldKeys),
		reasoningTypes: asStringList(firstValue(payload, reasoningTypeKeys)),
		expectedDocs:   firstDocList(payload, expectedDocKeys),
		evidenceDocs:   firstNonEmptySlice(firstDocList(payload, evidenceDocKeys), firstDocList(payload, expectedDocKeys)),
	}
}

func idValue(payload map[string]any, keys []string) string {
	if v := firstValue(payload, keys); v != nil {
		return fmt.Sprintf("%v", v)
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func firstNonEmptySlice(slices ...[]string) []string {
	for _, s := range slices {
		if len(s) > 0 {
			return s
		}
	}
	return nil
}

// selectQuestions ports select_questions: apply the config's question_ids
// block. include and random are alternative strategies (include wins); exclude
// is applied last in all cases; for the random strategy the sample is drawn
// from the pool after exclusion. The random seed is the current epoch second:
// every run draws a fresh sample by design, reproducibility comes from
// question_ids.include instead.
func selectQuestions(questions []*question, selection map[string]any) ([]*question, error) {
	var include []string
	if list, ok := selection["include"].([]any); ok {
		for _, id := range list {
			if s := strings.TrimSpace(fmt.Sprintf("%v", id)); s != "" {
				include = append(include, s)
			}
		}
	}
	randomCount := asIntDefault(selection["random"], 0)
	excluded := map[string]bool{}
	if list, ok := selection["exclude"].([]any); ok {
		for _, id := range list {
			excluded[strings.TrimSpace(fmt.Sprintf("%v", id))] = true
		}
	}

	if len(include) > 0 {
		byID := make(map[string]*question, len(questions))
		for _, q := range questions {
			byID[q.questionID] = q
		}
		var missing []string
		for _, qid := range include {
			if _, ok := byID[qid]; !ok {
				missing = append(missing, qid)
			}
		}
		if len(missing) > 0 {
			return nil, fmt.Errorf("question_ids.include contains unknown id(s): %s", strings.Join(missing, ", "))
		}
		var selected []*question
		for _, qid := range include {
			if !excluded[qid] {
				selected = append(selected, byID[qid])
			}
		}
		return selected, nil
	}

	var pool []*question
	for _, q := range questions {
		if !excluded[q.questionID] {
			pool = append(pool, q)
		}
	}
	if randomCount <= 0 || randomCount >= len(pool) {
		return pool, nil
	}
	rng := rand.New(rand.NewSource(time.Now().Unix()))
	rng.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	return pool[:randomCount], nil
}

func questionsPathOf(cfg map[string]any) string {
	dataset, _ := cfg["dataset"].(map[string]any)
	for _, value := range []any{
		dataset["questions_path"], cfg["questions_path"],
		cfg["frames_questions_path"], cfg["frames_mapping_path"],
	} {
		if s, ok := value.(string); ok && s != "" {
			return s
		}
	}
	return ""
}

// readIDList ports _read_id_list: ids separated by commas, whitespace or
// newlines; `#` starts a comment. Used by question_ids.include_file, so a
// generated batch lives in its own reviewable file instead of inline in the
// JSON config.
func readIDList(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var ids []string
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.SplitN(line, "#", 2)[0]
		for _, part := range strings.Fields(strings.ReplaceAll(line, ",", " ")) {
			if part = strings.TrimSpace(part); part != "" {
				ids = append(ids, part)
			}
		}
	}
	return ids
}

// ---------------------------------------------------------------------------
// Row identity / dedup helpers (port of _run_id / _row_run_key / _dedupe_last)
// ---------------------------------------------------------------------------

func rowRunKey(row map[string]any) string {
	if runID := strings.TrimSpace(strAny(row, "run_id", "")); runID != "" {
		return runID
	}
	return strings.TrimSpace(strAny(row, "question_id", ""))
}

func dedupeLast(rows []map[string]any) []map[string]any {
	// Collapse a JSONL file to one row per run_id (the last one wins) - a
	// re-run question appends a fresh row that shadows the earlier failure.
	// First-seen order is preserved; keyless rows are kept verbatim.
	index := map[string]int{}
	var out []map[string]any
	for _, row := range rows {
		key := rowRunKey(row)
		if key == "" {
			out = append(out, row)
			continue
		}
		if i, ok := index[key]; ok {
			out[i] = row
			continue
		}
		index[key] = len(out)
		out = append(out, row)
	}
	return out
}

func parseIntOrDefault(s string, def int) int {
	if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
		return n
	}
	return def
}
