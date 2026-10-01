package main

// Delivery-gate accounting helpers (port of _gate_audit_* and
// _unrecorded_locate_tools).

import (
	"regexp"
	"strings"
)

var locateTools = []string{"grep_chunks", "search_bm25_chunks", "search_semantic_chunks", "search_chunks"}

var searchedLineRE = regexp.MustCompile(`(?im)Searched:\s*(grep_chunks|search_bm25_chunks|search_semantic_chunks|search_chunks)`)

func gateAuditSuspects(row map[string]any) any {
	audit, _ := row["gate_audit"].(map[string]any)
	if audit == nil {
		return nil
	}
	list, ok := audit["suspects"].([]any)
	if !ok {
		return nil
	}
	out := make([]any, 0, len(list))
	for _, s := range list {
		out = append(out, s)
	}
	return out
}

func gateAuditPassed(row map[string]any) any {
	audit, _ := row["gate_audit"].(map[string]any)
	if audit == nil {
		return nil
	}
	if _, ok := audit["passed"]; !ok {
		return nil
	}
	return boolOf(audit["passed"])
}

func gateAuditRejections(row map[string]any) any {
	audit, _ := row["gate_audit"].(map[string]any)
	if audit == nil {
		return nil
	}
	return audit["rejections"]
}

func gateAuditFailures(row map[string]any) any {
	audit, _ := row["gate_audit"].(map[string]any)
	if audit == nil {
		return nil
	}
	return audit["audit_failures"]
}

func gateAuditCitationGroundings(row map[string]any) any {
	audit, _ := row["gate_audit"].(map[string]any)
	if audit == nil {
		return nil
	}
	return audit["citation_groundings"]
}

func gateAuditVerdicts(row map[string]any) any {
	audit, _ := row["gate_audit"].(map[string]any)
	if audit == nil {
		return nil
	}
	verdicts, ok := audit["audit_verdicts"].([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, v := range verdicts {
		out = append(out, strings.TrimSpace(fmtAny(v)))
	}
	return out
}

// unrecordedLocateTools ports _unrecorded_locate_tools: locate tools the run
// called but the Candidate Matrix never credits on a `Searched:` line.
// Reported, never scored: measured on 703 scored rows, 69% of runs omit at
// least one, so it is the norm rather than an anomaly.
func unrecordedLocateTools(row map[string]any) any {
	counts, ok := row["tool_call_counts"].(map[string]any)
	if !ok {
		return nil
	}
	listed := map[string]bool{}
	for _, m := range searchedLineRE.FindAllStringSubmatch(strAny(row, "ragflow_answer", ""), -1) {
		listed[strings.ToLower(m[1])] = true
	}
	var missing []string
	for _, tool := range locateTools {
		if asIntDefault(counts[tool], 0) > 0 && !listed[tool] {
			missing = append(missing, tool)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	return missing
}

// usageRow ports _usage_row: per-question token and time cost, straight from
// the backend's run accounting plus the client-side wall clock.
func usageRow(queryID string, row map[string]any, searchTools []string) map[string]any {
	usage, _ := row["usage"].(map[string]any)
	if usage == nil {
		usage = map[string]any{}
	}
	counts, _ := row["tool_call_counts"].(map[string]any)
	if counts == nil {
		counts = map[string]any{}
	}
	promptTokens := asInt(usage["prompt_tokens"])
	completionTokens := asInt(usage["completion_tokens"])
	totalTokens := asInt(usage["total_tokens"])
	if totalTokens == nil && promptTokens != nil && completionTokens != nil {
		t := *promptTokens + *completionTokens
		totalTokens = &t
	}
	searchCalls := 0
	for _, name := range searchTools {
		searchCalls += asIntDefault(counts[name], 0)
	}
	toolCalls := map[string]int{}
	for name, count := range counts {
		toolCalls[name] = asIntDefault(count, 0)
	}
	var retrievedCount any
	if retrieved, ok := row["retrieved_docids"].([]any); ok {
		retrievedCount = len(retrieved)
	}
	return map[string]any{
		"query_id":                queryID,
		"input_tokens":            promptTokens,
		"output_tokens":           completionTokens,
		"total_tokens":            totalTokens,
		"llm_calls":               asInt(usage["llm_calls"]),
		"llm_turns":               row["llm_turns"],
		"server_seconds":          asFloat(row["server_elapsed_seconds"]),
		"client_seconds":          asFloat(row["answer_elapsed_seconds"]),
		"search_calls":            searchCalls,
		"tool_calls":              toolCalls,
		"retrieved_doc_count":     retrievedCount,
		"audit_suspects":          gateAuditSuspects(row),
		"audit_passed":            gateAuditPassed(row),
		"audit_rejections":        gateAuditRejections(row),
		"audit_failures":          gateAuditFailures(row),
		"citation_groundings":     gateAuditCitationGroundings(row),
		"audit_verdicts":          gateAuditVerdicts(row),
		"unrecorded_locate_tools": unrecordedLocateTools(row),
		"deep_read_chunks":        asInt(row["deep_read_chunks"]),
		"shallow_read_chunks":     asInt(row["shallow_read_chunks"]),
	}
}
