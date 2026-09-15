/*
 *  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package agentic_rag

import (
	"testing"
)

// q784's shape: the wrong name was one entry among many in a cast list, and the
// question asks for a birth name — which a cast list cannot state.
const castListHaystack = `<chunk chunk_id="c1" doc_id="24653">` +
	`<match_snippet>...du directeurRobert VattierGisèle GrayAndré BervilJacques DynamJacques Meyran` +
	`Félix PaquetAndrée ServilangeJean DaurandPhilippe JanvierJoe BreitbardMarcel Perès` +
	`Pierre Albert BrasseurHenri HenneryHarry MaxJean-Pierre LorrainRené PascalÉdouard Rousseau` +
	`Jacques AngelvinJean SylvainMercédès BrarePaul Deman...</match_snippet></chunk>`

// TestShouldDemoteFinalAnswer pins the label-governance condition after the
// Rejections arm was added: a gate that refused deliverables pre-audit (empty
// Suspects) must still demote, or an unaudited `Final Answer` ships.
func TestShouldDemoteFinalAnswer(t *testing.T) {
	cases := []struct {
		name  string
		audit *GateAuditRecord
		want  bool
	}{
		{"no record", nil, false},
		{"audited and passed", &GateAuditRecord{Suspects: []int{0}, Passed: true}, false},
		{"audited, suspects left", &GateAuditRecord{Suspects: []int{3}, Passed: false}, true},
		{"never audited, but refused", &GateAuditRecord{Rejections: 4, Passed: false}, true},
		{"clean run", &GateAuditRecord{Passed: false}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := shouldDemoteFinalAnswer(tc.audit); got != tc.want {
				t.Fatalf("shouldDemoteFinalAnswer = %v, want %v", got, tc.want)
			}
		})
	}
	rec := &GateAuditRecord{}
	countGateRejection(rec)
	countGateRejection(rec)
	countGateRejection(nil) // must not panic
	if rec.Rejections != 2 {
		t.Fatalf("Rejections = %d, want 2", rec.Rejections)
	}
}

// TestShouldDemoteFinalAnswerOnAuditFailure: an audit that never produced a
// verdict must not read as "audited". The gate exits on an audit error BEFORE
// recording a verdict, so without the AuditFailures count the record was empty
// on a never-audited deliverable and a `Final Answer` shipped as-is.
func TestShouldDemoteFinalAnswerOnAuditFailure(t *testing.T) {
	if !shouldDemoteFinalAnswer(&GateAuditRecord{AuditFailures: 2}) {
		t.Error("an aborted audit (no verdict) must demote Final to Guessed")
	}
	if shouldDemoteFinalAnswer(&GateAuditRecord{AuditFailures: 0, Passed: true}) {
		t.Error("a PASSed audit must keep the Final label")
	}
	// An empty record means no gate was armed for this run (explicit toolset or
	// audit_max_pass = 0): the label governance deliberately stays out of it.
	if shouldDemoteFinalAnswer(&GateAuditRecord{}) {
		t.Error("a run with no gate at all must not be demoted by this rule")
	}
	// The counter has to be reachable from the gate's error path.
	rec := &GateAuditRecord{}
	countGateAuditFailure(rec)
	countGateAuditFailure(nil)
	if rec.AuditFailures != 1 {
		t.Errorf("AuditFailures = %d, want 1", rec.AuditFailures)
	}
}
