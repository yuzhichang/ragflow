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
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

const probeQuestion = "Two individuals from different industries share the same first name and surname, and their mothers share a first name. One attended a school with exactly 3 founders. What is the shared first and last name?"

// lawfulPlan is the shape the checks must accept: a root naming the question's
// own anchor, a chain edge with its pointer, and a partition of numbers.
const lawfulPlan = `### Sub-question 1: the school with 3 founders attended by one of the two individuals — slot: name — kind: institution
- Op: lookup
- Binds: ?school
- From: none
- Constraints: c1 = the school has 3 founders; c2 = the school's name is the birthplace of one of those 3 founders

### Sub-question 2: the shared first and last name of the two individuals at ?school — slot: name — kind: person_name
- Op: follow
- Binds: ?answer
- From: ?school (block 1)
- Constraints: c3 = same first and last name; c4 = different industries
`

func TestCheckDecompositionAcceptsLawfulPlan(t *testing.T) {
	if got := checkDecomposition(probeQuestion, lawfulPlan); len(got) != 0 {
		t.Fatalf("lawful plan reported findings: %v", got)
	}
}

// A plan written with markdown emphasis must read exactly like the same plan
// without it: one archived run wrote `- **Op:**` and every header-derived metric
// it had went to zero.
func TestCheckDecompositionToleratesBoldMarkup(t *testing.T) {
	bold := strings.NewReplacer(
		"- Op:", "- **Op:**",
		"- Binds:", "- **Binds:**",
		"- From:", "- **From:**",
		"- Constraints:", "- **Constraints:**",
	).Replace(lawfulPlan)
	if got := checkDecomposition(probeQuestion, bold); len(got) != 0 {
		t.Fatalf("bold plan reported findings: %v", got)
	}
}

func TestCheckDecompositionFindings(t *testing.T) {
	cases := []struct {
		name    string
		plan    string
		want    string // a substring of the expected finding
		notWant string
	}{
		{
			name: "answer bound twice",
			plan: strings.Replace(lawfulPlan, "- Binds: ?school", "- Binds: ?answer", 1),
			want: "schema integrity: variable is bound twice (?answer",
		},
		{
			name: "title inlines another block's candidate",
			plan: strings.Replace(lawfulPlan, "the school with 3 founders attended by one of the two individuals",
				"the school attended by Jane Doe", 1),
			want: "suspect: sub-question inlines another block's candidate - use its variable (Doe, Jane, block 1)",
		},
		{
			name: "constraint inlines another block's candidate",
			plan: strings.Replace(lawfulPlan, "c3 = same first and last name", "c3 = the school where Jane Doe taught", 1),
			want: "suspect: constraint inlines another block's candidate - use its variable",
		},
		{
			name: "title with a variable still inlines a name",
			plan: strings.Replace(lawfulPlan, "the school with 3 founders attended by one of the two individuals",
				"the school attended by Jane Doe", 1),
			want: "suspect: sub-question inlines another block's candidate - use its variable",
		},
		{
			name: "title names no anchor",
			plan: strings.Replace(lawfulPlan, "the school with 3 founders attended by one of the two individuals", "the school of the person", 1),
			want: "suspect: sub-question is not self-contained - it names no anchor (block 1)",
		},
		{
			name: "block number crosses a boundary",
			plan: strings.Replace(lawfulPlan, "c4 = different industries", "c4 = the school of block 1", 1),
			want: "schema integrity: block number or prose is used across a block boundary",
		},
		{
			name: "root whose variable nobody consumes",
			plan: strings.Replace(lawfulPlan, "- From: ?school (block 1)", "- From: none", 1),
			want: "schema integrity: root block's variable is never consumed (block 1)",
		},
		{
			name:    "lawful plan has no orphan finding",
			plan:    lawfulPlan,
			notWant: "never consumed",
		},
		{
			name: "constraint number restarts",
			plan: strings.Replace(lawfulPlan, "c3 = same first and last name", "c1 = same first and last name", 1),
			want: "schema integrity: constraint number is reused for a different claim (c1)",
		},
		{
			name: "pointer mismatch",
			plan: strings.Replace(lawfulPlan, "- From: ?school (block 1)", "- From: ?school (block 2)", 1),
			want: "schema integrity: dependency pointer does not match the binding block (?school",
		},
		{
			name: "from entry without pointer",
			plan: strings.Replace(lawfulPlan, "- From: ?school (block 1)", "- From: ?school", 1),
			want: "schema integrity: from entry has no block pointer (?school",
		},
		{
			name: "follow without a bound input",
			plan: strings.Replace(lawfulPlan, "- From: ?school (block 1)", "- From: none", 1),
			want: "schema integrity: follow op has no bound input (block 2)",
		},
		{
			name: "constraints line missing",
			plan: strings.Replace(lawfulPlan, "- Constraints: c3 = same first and last name; c4 = different industries\n", "", 1),
			want: "schema integrity: constraints line is missing (block 2)",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := checkDecomposition(probeQuestion, tc.plan)
			joined := strings.Join(got, "\n")
			if tc.want != "" && !strings.Contains(joined, tc.want) {
				t.Fatalf("want a finding containing %q, got %v", tc.want, got)
			}
			if tc.notWant != "" && strings.Contains(joined, tc.notWant) {
				t.Fatalf("did not want %q, got %v", tc.notWant, got)
			}
		})
	}
}

// TestProbeArchivedDecompositions replays the checker over archived runs. It is
// skipped unless DECOMP_PROBE_DIR names a directory of answers.jsonl files, so
// the offline replay needs no absolute path in the repository.
func TestProbeArchivedDecompositions(t *testing.T) {
	dir := os.Getenv("DECOMP_PROBE_DIR")
	if dir == "" {
		t.Skip("set DECOMP_PROBE_DIR to a directory of answers.jsonl files to replay archived decompositions")
	}
	entries, err := filepath.Glob(filepath.Join(dir, "*", "answers.jsonl"))
	if err != nil || len(entries) == 0 {
		t.Skipf("no answers.jsonl under %s", dir)
	}
	thinkRe := regexp.MustCompile(`(?s)<think>.*?</think>`)
	for _, path := range entries {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		for _, line := range strings.Split(string(raw), "\n") {
			if strings.TrimSpace(line) == "" {
				continue
			}
			var row struct {
				QuestionID string `json:"question_id"`
				Question   string `json:"question"`
				Answer     string `json:"ragflow_answer"`
			}
			if err := json.Unmarshal([]byte(line), &row); err != nil {
				continue
			}
			plan := thinkRe.ReplaceAllString(row.Answer, "")
			findings := checkDecomposition(row.Question, plan)
			t.Logf("%s q%s: %d finding(s) %v", filepath.Base(filepath.Dir(path)), row.QuestionID, len(findings), findings)
		}
	}
}
