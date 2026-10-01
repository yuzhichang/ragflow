package main

// The _diagnostics block of leaderboard.json (port of the tail of
// build_leaderboard): adoption percentages, early-stop symptom, intermediate
// grounding, and the delivery checks with their measured-never-enforced
// notes.

import (
	"fmt"
	"sort"
)

func diagnosticsBlock(total, judged, judgedOK, recallScored int, searchTools []string,
	calibrationNote string, statsMissing int, structureAdoption map[string]float64,
	failureTypeCounts map[string]int, earlyStopHits, earlyStopRows, intermediateGrounded,
	intermediateChecked int, intermediateNote string, deliveryTotals map[string]int,
	inlineRows, pointerRows, retiredRows, pivotRows, unrecordedRows, partitionRows,
	reuseRows, leakRows, decompositionRows int, signatureTotal int, promptSigs map[string]any,
) map[string]any {
	note := "Accuracy counts every question of the run; failed and unjudged rows count as incorrect."
	if statsMissing > 0 {
		note = fmt.Sprintf("%s Recall and Search Calls are understated: %d row(s) were answered by a backend that did not report tool_call_counts / retrieved_docids, and rows without accounting are scored as 'retrieved nothing'.", note, statsMissing)
	}
	failureTypes := map[string]int{}
	for k, v := range failureTypeCounts {
		failureTypes[k] = v
	}
	earlyStopPercent := any(nil)
	if earlyStopRows > 0 {
		earlyStopPercent = round1(float64(earlyStopHits) / float64(earlyStopRows) * 100)
	}
	intermediatePercent := any(nil)
	if intermediateChecked > 0 {
		intermediatePercent = round1(float64(intermediateGrounded) / float64(intermediateChecked) * 100)
	}
	judgedOnlyAccuracy := any(nil)
	if judged > 0 {
		judgedOnlyAccuracy = round2(float64(judgedOK) / float64(judged) * 100)
	}
	return map[string]any{
		"questions": total, "judged": judged,
		"unjudged_counted_incorrect":   total - judged,
		"judged_only_accuracy_percent": judgedOnlyAccuracy,
		"judged_only_denominator":      judged,
		"recall_scored":                recallScored,
		"search_tools":                 searchTools,
		"calibration":                  calibrationNote,
		"run_stats_missing":            statsMissing,
		"structure_adoption_percent":   structureAdoption,
		"failure_types":                failureTypes,
		"early_stop": map[string]any{
			"threshold_docs": earlyStopDocs, "rows": earlyStopRows,
			"rows_at_or_below": earlyStopHits, "percent": earlyStopPercent,
			"note": "A run that concludes inside a handful of retrieved documents is a SYMPTOM, not a score: finding the document and giving up look identical here, so read this beside Accuracy.",
		},
		"intermediate_nodes": map[string]any{
			"values_checked": intermediateChecked, "values_grounded": intermediateGrounded,
			"percent": intermediatePercent, "note": intermediateNote,
		},
		"delivery_checks": deliveryChecksBlock(total, deliveryTotals, inlineRows, pointerRows,
			retiredRows, pivotRows, unrecordedRows, partitionRows, reuseRows, leakRows,
			decompositionRows, signatureTotal, promptSigs),
		"note": note,
	}
}

var _ = sort.Strings
