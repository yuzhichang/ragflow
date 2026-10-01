package main

// Run-stats extraction (port of extract_run_stats and _llm_turn): read the
// run accounting the backend ships next to the answer. All of it is emitted
// by the Go backend on non-streaming agentic responses only; absent keys are
// left out rather than defaulted to zero, so a row can be flagged as "no
// accounting" instead of being scored as a run that made no calls.

import (
	"fmt"
	"io"
	"sort"
)

func ioReadAll(r io.Reader) ([]byte, error) {
	return io.ReadAll(r)
}

func llmTurn(turn map[string]any) map[string]any {
	prompt := asInt(turn["prompt_tokens"])
	completion := asInt(turn["completion_tokens"])
	total := asInt(turn["total_tokens"])
	if total == nil && prompt != nil && completion != nil {
		t := *prompt + *completion
		total = &t
	}
	model := ""
	if v, ok := turn["model"].(string); ok {
		model = v
	}
	return map[string]any{
		"seq":           asInt(turn["seq"]),
		"at_seconds":    asFloat(turn["at_seconds"]),
		"model":         nilIfEmpty(model),
		"input_tokens":  prompt,
		"output_tokens": completion,
		"total_tokens":  total,
	}
}

// extractRunStats ports extract_run_stats.
func extractRunStats(payload any) map[string]any {
	source, ok := payload.(map[string]any)
	if !ok {
		return nil
	}
	if _, hasDirect := source["tool_call_counts"]; !hasDirect {
		if _, hasDocs := source["retrieved_docids"]; !hasDocs {
			if data, ok := payload.(map[string]any)["data"].(map[string]any); ok {
				source = data
			}
		}
	}
	stats := map[string]any{}
	if counts, ok := source["tool_call_counts"].(map[string]any); ok {
		out := map[string]int{}
		for name, count := range counts {
			if n := asInt(count); n != nil {
				out[name] = *n
			}
		}
		stats["tool_call_counts"] = out
	}
	if errors, ok := source["tool_call_errors"].(map[string]any); ok {
		errCounts := map[string]int{}
		for name, count := range errors {
			if n := asInt(count); n != nil {
				errCounts[name] = *n
			}
		}
		if len(errCounts) > 0 {
			stats["tool_call_errors"] = errCounts
		}
	}
	if samples, ok := source["tool_error_samples"].(map[string]any); ok && len(samples) > 0 {
		out := map[string]string{}
		for name, text := range samples {
			out[name] = fmtAny(text)
		}
		stats["tool_error_samples"] = out
	}
	if docs, ok := source["retrieved_docids"].([]any); ok {
		stats["retrieved_docids"] = uniqueDocIDs(docs)
	}
	if usage, ok := source["usage"].(map[string]any); ok {
		out := map[string]any{}
		for _, key := range []string{"prompt_tokens", "completion_tokens", "total_tokens", "llm_calls"} {
			out[key] = asInt(usage[key])
		}
		stats["usage"] = out
		if turns, ok := usage["llm_turns"].([]any); ok && len(turns) > 0 {
			var ported []map[string]any
			for _, turnAny := range turns {
				if turn, ok := turnAny.(map[string]any); ok {
					ported = append(ported, llmTurn(turn))
				}
			}
			if len(ported) > 0 {
				stats["llm_turns"] = ported
			}
		}
	}
	if elapsed := asFloat(source["elapsed_seconds"]); elapsed != nil {
		stats["server_elapsed_seconds"] = round3(*elapsed)
	}
	for _, key := range []string{"deep_read_chunks", "shallow_read_chunks"} {
		if value := asInt(source[key]); value != nil {
			stats[key] = *value
		}
	}
	for _, key := range []string{"deep_read_chunk_ids", "shallow_read_chunk_ids"} {
		if value := uniqueIDs(source[key]); len(value) > 0 {
			stats[key] = value
		}
	}
	if gateAudit, ok := source["gate_audit"].(map[string]any); ok {
		suspects, hasSuspects := gateAudit["suspects"].([]any)
		rejections := asInt(gateAudit["rejections"])
		auditFailures := asInt(gateAudit["audit_failures"])
		verdicts, hasVerdicts := gateAudit["audit_verdicts"].([]any)
		citationGroundings := asInt(gateAudit["citation_groundings"])
		// Keep the record when ANY signal is present: a gate that refused
		// every deliverable before an audit could run reports no suspects at
		// all, and one whose auditor never returned a verdict reports nothing
		// but the failure - dropping either hid exactly that state
		// (q350/q784, and the audit-outage case).
		if hasSuspects || rejections != nil || auditFailures != nil || hasVerdicts || citationGroundings != nil {
			var verdictList []string
			if hasVerdicts {
				for _, v := range verdicts {
					verdictList = append(verdictList, fmtAny(v))
				}
			}
			stats["gate_audit"] = map[string]any{
				"suspects":            suspectList(suspects, hasSuspects),
				"passed":              boolOf(gateAudit["passed"]),
				"rejections":          rejections,
				"audit_failures":      auditFailures,
				"citation_groundings": citationGroundings,
				"audit_verdicts":      verdictListIf(hasVerdicts, verdictList),
			}
		}
	}
	return stats
}

func suspectList(list []any, ok bool) any {
	if !ok {
		return nil
	}
	out := make([]any, 0, len(list))
	for _, s := range list {
		out = append(out, asInt(s))
	}
	return out
}

func verdictListIf(ok bool, list []string) any {
	if !ok {
		return nil
	}
	return list
}

var _ = sort.Strings
var _ = fmt.Sprintf
