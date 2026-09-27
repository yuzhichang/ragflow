// Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing limitations under
// the License.

package agentic_rag

import (
	"strings"
	"testing"
)

// TestAuditShipsOnAdvisory pins the pipeline's own grading of audit findings:
// an auditor that leaves a completeness finding untagged still gets overruled,
// while a BLOCKING finding keeps the refusal. #36: the auditor graded one
// advisory finding across thirteen rounds, and the one class it did grade
// handed q875 its first win - so the grading cannot be the auditor's call.
func TestAuditShipsOnAdvisory(t *testing.T) {
	advisory := "## Candidate Matrix\n### Sub-question 1: the school\n- Retained: X\n  - audit: suspect: constraint hits not consumed - 95776.md, 35876.md\nAudit Result: FAIL (1 suspects)"
	blocking := "## Candidate Matrix\n### Sub-question 1: the school\n- Retained: X\n  - audit: suspect: elimination ground contradicted by cited chunk - the c5 evidence does not state it\nAudit Result: FAIL (1 suspects)"
	mixed := "## Candidate Matrix\n### Sub-question 1: the school\n- Retained: X\n  - audit: suspect: derived line is missing (block 1)\n- Derived: y\n  - audit: suspect: the supported clues never derive the answer\nAudit Result: FAIL (2 suspects)"
	passed := "## Candidate Matrix\n- Retained: X\n  - audit: pass\nAudit Result: PASS"

	if !auditShipsOnAdvisory(advisory) {
		t.Fatalf("a completeness finding must ship: blocking=%v", auditBlockingOpinions(advisory))
	}
	if auditShipsOnAdvisory(blocking) {
		t.Fatalf("a contradicted ground must NOT ship")
	}
	if auditShipsOnAdvisory(mixed) {
		t.Fatalf("one blocking finding among advisory ones must NOT ship: blocking=%v", auditBlockingOpinions(mixed))
	}
	if auditShipsOnAdvisory(passed) {
		t.Fatalf("an already-passing verdict is not an advisory waiver")
	}
	if n := len(auditOpinions(advisory)); n != 1 {
		t.Fatalf("auditOpinions = %d, want 1", n)
	}
	if got := strings.Join(auditBlockingOpinions(mixed), " | "); !strings.Contains(got, "never derive the answer") {
		t.Fatalf("the blocking opinion must survive, got %q", got)
	}
}
