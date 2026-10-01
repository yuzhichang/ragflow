package main

// deliveryChecksBlock: the delivery_checks subtree of _diagnostics, with the
// measured-never-enforced notes carried over verbatim.

func deliveryChecksBlock(total int, deliveryTotals map[string]int, inlineRows, pointerRows,
	retiredRows, pivotRows, unrecordedRows, partitionRows, reuseRows, leakRows,
	decompositionRows, signatureTotal int, promptSigs map[string]any) map[string]any {
	percent := any(nil)
	if deliveryTotals["blocks_total"] > 0 {
		percent = round1(float64(deliveryTotals["blocks_parsed"]) / float64(deliveryTotals["blocks_total"]) * 100)
	}
	return map[string]any{
		"rows": total,
		"inlined_candidate": map[string]any{
			"rows": inlineRows, "occurrences": deliveryTotals["inlined_candidate"],
			"note": "A candidate name written into a block's own title freezes a finding into the plan, so the same question decomposes differently on another run. Membership, not phrasing: the name comes from the block's own Tested/Eliminated/Retained lines.",
		},
		"from_pointer": map[string]any{
			"entries": deliveryTotals["from_entries"], "with_pointer": deliveryTotals["from_pointers"],
			"without_pointer": deliveryTotals["from_entries_without_pointer"],
			"mismatch":        deliveryTotals["from_pointer_mismatch"], "unbound": deliveryTotals["from_unbound"],
			"rows_with_a_defect": pointerRows,
			"note":               "Each `From: ?x (block N)` should name the block that binds ?x - a DERIVED pointer, checkable against Binds. `without_pointer` counts entries that carry no pointer, `mismatch` counts pointers naming another block, `unbound` counts names no block binds.",
		},
		"depends_on_written": map[string]any{
			"rows": retiredRows, "occurrences": deliveryTotals["depends_on_written"],
			"note": "The field is RETIRED (the chain is stated by From); a nonzero count is the old contract still being written, not an adoption.",
		},
		"searched_record": map[string]any{
			"lines": deliveryTotals["searched_lines"], "blocks_without_searched": deliveryTotals["blocks_without_searched"],
			"rows": unrecordedRows,
			"note": "A block that asserts anything about the corpus owes at least one recorded `Searched:` line; an unrecorded run is unauditable rather than clean.",
		},
		"pivot": map[string]any{
			"blocks_with_by_name_query": deliveryTotals["blocks_with_by_name_query"],
			"pivot_missing_blocks":      deliveryTotals["pivot_missing_blocks"],
			"rows_with_a_missing_pivot": pivotRows,
			"single_term_searches":      deliveryTotals["single_term_searches"],
			"note":                      "The by-name query strategy, counted from the deliverable's own `Searched` patterns against its own candidate names: a block with candidates and no by-name query verified none of them. `single_term_searches` is the crude proxy for the clue-anchor query (<=3 words in the pattern).",
		},
		"decomposition_graph": map[string]any{
			"rows_with_a_defect":    decompositionRows,
			"detached_blocks":       deliveryTotals["decomposition_detached_blocks"],
			"orphan_roots":          deliveryTotals["decomposition_orphan_roots"],
			"roots_after_the_first": deliveryTotals["decomposition_from_none_after_first"],
			"prose_block_refs":      deliveryTotals["decomposition_prose_block_refs"],
			"note":                  "The decomposition must be ONE connected acyclic graph that carries `?answer`. The defects are the two that break the graph: an ORPHAN ROOT whose variable no block consumes and a block outside the `?answer` component; plus one that breaks resolution: an edge written in prose instead of the variable. Counted, not enforced.",
		},
		"constraint_partition": map[string]any{
			"defined": deliveryTotals["constraints_defined"], "duplicated": deliveryTotals["constraints_duplicated"],
			"blocks_without_constraint": deliveryTotals["blocks_without_constraint"],
			"rows_with_a_duplicate":     partitionRows,
			"number_reused":             deliveryTotals["constraint_number_reused"],
			"rows_with_a_reused_number": reuseRows,
			"note":                      "The numbered constraints form a PARTITION: each `c<k>` is defined by exactly one block. Measured 2026-09-23: 26 of 186 definitions were duplicates.",
		},
		"prompt_leakage": map[string]any{
			"signature_total": signatureTotal, "signatures_loaded": boolOf(promptSigs["loaded"]),
			"rows": leakRows, "leaked_docnames": deliveryTotals["leaked_docnames"],
			"leaked_values":         deliveryTotals["leaked_values"],
			"example_value_shipped": deliveryTotals["example_value_shipped"],
			"note":                  "A CANARY over the prompt's own examples, not a judgement. `signature_total` counts what was looked for - read it beside the zeros.",
		},
		"parse_coverage": map[string]any{
			"blocks_parsed": deliveryTotals["blocks_parsed"], "blocks_total": deliveryTotals["blocks_total"],
			"percent": percent,
			"note":    "A block counts as parsed when its title carries `slot:` and it carries `Op:`, `Binds:` and `From:`. READ THIS BESIDE EVERY COUNT ABOVE: a count taken over an unparsed block is not evidence of compliance.",
		},
		"note": "Measured, never enforced: these checks are membership, not phrasing, so the gate COULD carry them - but a gate opinion costs a whole repair turn on every run, and prevalence is not measured yet. Collect the prevalence here first.",
	}
}
