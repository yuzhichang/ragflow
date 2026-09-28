//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package agentic_rag

import (
	"strings"
	"testing"
)

// chainPlan is a four-block chain: school -> attendee -> twin -> answer, each
// consuming the previous variable - the shape the Survivors line must respect.
const chainPlan = `### Sub-question 1: ?school is the school with 3 founders whose name is the birthplace of one of them — slot: name — kind: institution
- Op: lookup
- Binds: ?school
- From: none
- Constraints: c1 = ?school has 3 founders

### Sub-question 2: ?attendee is an individual who attended ?school — slot: name — kind: person
- Op: follow
- Binds: ?attendee
- From: ?school (block 1)
- Constraints: c2 = ?attendee attended ?school

### Sub-question 3: ?twin is the individual from a different industry sharing ?attendee's first and last name — slot: name — kind: person
- Op: follow
- Binds: ?twin
- From: ?attendee (block 2)
- Constraints: c3 = ?twin shares ?attendee's first and last name

### Sub-question 4: ?answer is the shared first and last name — slot: name — kind: name
- Op: filter
- Binds: ?answer
- From: ?twin (block 3), ?attendee (block 2)
- Constraints: c4 = ?answer is the shared first and last name
`

// The measured bad line (#38 r1): the twin and answer slots filled while the
// attendee slot they stand on read empty. Both downstream fills are chain
// breaks; the answer filler itself traces to the twin set, so the provenance
// check adds nothing here.
func TestSurvivorsChainBreaksOnEmptyUpstream(t *testing.T) {
	final := "## Candidate Matrix\n" +
		"### Sub-question 4: ?answer is the shared first and last name\n" +
		"- Survivors: {?school: [Benenden School], ?attendee: [], ?twin: [Michael Jordan (NBA), Michael Jordan (UK)], ?answer: [Michael Jordan]}"
	blocks, _ := checkDecompositionBlocks(chainPlan)
	got := survivorsChainBreaks(final, blocks)
	if len(got) != 2 {
		t.Fatalf("want 2 chain breaks (twin, answer over empty attendee), got %d: %q", len(got), got)
	}
	joined := strings.Join(got, " | ")
	if !strings.Contains(joined, "?twin") || !strings.Contains(joined, "?attendee") {
		t.Fatalf("breaks must name the filled variable and its empty upstream, got %q", joined)
	}
}

// A chain built in order is silent: every non-empty variable's upstream sets
// are filled, and the answer filler traces to an upstream set.
func TestSurvivorsChainAcceptsOrderedFill(t *testing.T) {
	final := "## Candidate Matrix\n" +
		"- Survivors: {?school: [St Sithians], ?attendee: [Kevin Anderson], ?twin: [Kevin B. Anderson], ?answer: [Kevin Anderson]}"
	blocks, _ := checkDecompositionBlocks(chainPlan)
	if got := survivorsChainBreaks(final, blocks); len(got) != 0 {
		t.Fatalf("an ordered fill must yield no findings, got %q", got)
	}
}

// The answer-provenance half: an answer filler no upstream set holds entered
// from outside the run - named even when every slot is filled.
func TestSurvivorsChainNamesOutsideAnswer(t *testing.T) {
	final := "## Candidate Matrix\n" +
		"- Survivors: {?school: [St Sithians], ?attendee: [Kevin Anderson], ?twin: [Kevin B. Anderson], ?answer: [Steve McQueen]}"
	blocks, _ := checkDecompositionBlocks(chainPlan)
	got := survivorsChainBreaks(final, blocks)
	if len(got) != 1 || !strings.Contains(got[0], "Steve McQueen") {
		t.Fatalf("an outside answer filler must be named, got %q", got)
	}
}

// Tolerance: no Survivors line, no plan blocks, and a bolded Survivors
// variant must all behave - the check adds a refusal reason, not a new way
// for a malformed run to fail.
func TestSurvivorsChainToleratesMalformedInput(t *testing.T) {
	blocks, _ := checkDecompositionBlocks(chainPlan)
	if got := survivorsChainBreaks("## Candidate Matrix\nno survivors here", blocks); got != nil {
		t.Fatalf("a deliverable without a Survivors line must be skipped, got %q", got)
	}
	if got := survivorsChainBreaks("- Survivors: {?school: [x]}", nil); got != nil {
		t.Fatalf("no plan blocks must be skipped, got %q", got)
	}
	// Bold markup on the Survivors line parses like the plain form.
	bold := "## Candidate Matrix\n" +
		"- **Survivors:** {?school: [St Sithians], ?attendee: [Kevin Anderson], ?twin: [Kevin B. Anderson], ?answer: [Kevin Anderson]}"
	if got := survivorsChainBreaks(bold, blocks); len(got) != 0 {
		t.Fatalf("a bolded Survivors line must parse, got %q", got)
	}
}
