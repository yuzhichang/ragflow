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
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const twoTemplateYAML = `templates:
  - id: smart-reasoning
    name: Progressive Agentic RAG
    description: full retrieval
    tools:
      - think
      - grep_chunks
      - search_chunks
      - list_chunks
    content: |
      FULL-MODE PROMPT
  - id: smart-grep
    name: Grep-Only Agentic RAG
    description: grep retrieval only
    tools:
      - think
      - grep_chunks
      - list_chunks
    content: |
      GREP-ONLY PROMPT
`

func setupTestConfig(t *testing.T) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "agentic_rag.yaml")
	if err := os.WriteFile(path, []byte(twoTemplateYAML), 0o600); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	t.Setenv("AGENTIC_RAG_CONFIG", path)
	// Force a reload of the process-wide cache so each test sees its own file.
	configMu.Lock()
	cachedFile = nil
	configMu.Unlock()
	t.Cleanup(func() {
		configMu.Lock()
		cachedFile = nil
		configMu.Unlock()
	})
}

func TestResolveTemplateForExplicitIDPicksRequestedTemplate(t *testing.T) {
	setupTestConfig(t)

	tmpl, err := resolveTemplateFor("smart-grep")
	if err != nil {
		t.Fatalf("resolve smart-grep: %v", err)
	}
	if tmpl.ID != "smart-grep" {
		t.Fatalf("template id = %q, want smart-grep", tmpl.ID)
	}
	for _, tool := range tmpl.Tools {
		if tool == "search_chunks" {
			t.Fatalf("grep-only template must not include search_chunks: %v", tmpl.Tools)
		}
	}
	if tmpl.Content != "GREP-ONLY PROMPT\n" {
		t.Fatalf("content = %q, want GREP-ONLY PROMPT", tmpl.Content)
	}
}

func TestResolveTemplateForUnknownIDIsHardError(t *testing.T) {
	setupTestConfig(t)

	if _, err := resolveTemplateFor("no-such-template"); err == nil ||
		!strings.Contains(err.Error(), "not found") {
		t.Fatalf("unknown id must fail loudly, got err=%v", err)
	}
}

func TestResolveTemplateForEmptyIDIsHardError(t *testing.T) {
	setupTestConfig(t)

	if _, err := resolveTemplateFor(""); err == nil ||
		!strings.Contains(err.Error(), "empty agent_mode") {
		t.Fatalf("empty id must fail loudly, got err=%v", err)
	}
}

// TestAuditMaxPassComesFromConfig pins the gate's budget to the template: the
// value lives in conf/agentic_rag.yaml (`audit_max_pass`), so an operator can
// retune how much a template is willing to spend on auditing without a
// rebuild. A template that omits it gets 0 — which means unaudited, not
// "default budget" — so opting a template into the gate stays explicit.
func TestAuditMaxPassComesFromConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agentic_rag.yaml")
	yaml := `templates:
  - id: audited
    name: Audited
    description: ships through the gate
    audit_max_pass: 7
    tools:
      - list_chunks
    content: |
      AUDITED PROMPT
  - id: unaudited
    name: Unaudited
    description: ships as-is
    tools:
      - list_chunks
    content: |
      UNAUDITED PROMPT
`
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	t.Setenv("AGENTIC_RAG_CONFIG", path)
	configMu.Lock()
	cachedFile = nil
	configMu.Unlock()
	t.Cleanup(func() {
		configMu.Lock()
		cachedFile = nil
		configMu.Unlock()
	})

	audited, err := resolveTemplateFor("audited")
	if err != nil {
		t.Fatalf("resolve audited: %v", err)
	}
	if audited.AuditMaxPass != 7 {
		t.Fatalf("audit_max_pass = %d, want 7", audited.AuditMaxPass)
	}

	unaudited, err := resolveTemplateFor("unaudited")
	if err != nil {
		t.Fatalf("resolve unaudited: %v", err)
	}
	if unaudited.AuditMaxPass != 0 {
		t.Fatalf("audit_max_pass = %d, want 0 (absent means no gate)", unaudited.AuditMaxPass)
	}
}

// TestShippedConfigAuditsResearchTemplates guards the wiring end to end: the
// research templates in conf/agentic_rag.yaml must opt into the gate, and the
// auditor template must NOT (it is the gate's tool, not a producer).
func TestShippedConfigAuditsResearchTemplates(t *testing.T) {
	t.Setenv("AGENTIC_RAG_CONFIG", filepath.Join("..", "..", "conf", "agentic_rag.yaml"))
	configMu.Lock()
	cachedFile = nil
	configMu.Unlock()
	t.Cleanup(func() {
		configMu.Lock()
		cachedFile = nil
		configMu.Unlock()
	})

	for _, id := range []string{"smart-reasoning", "smart-grep", "smart-grep-bm25"} {
		tmpl, err := resolveTemplateFor(id)
		if err != nil {
			t.Fatalf("resolve %s: %v", id, err)
		}
		if tmpl.AuditMaxPass <= 0 {
			t.Errorf("%s: audit_max_pass = %d, want > 0 (the gate is what catches fabricated citations)", id, tmpl.AuditMaxPass)
		}
	}

	auditor, err := resolveTemplateFor(answerAuditorTemplateID)
	if err != nil {
		t.Fatalf("resolve %s: %v", answerAuditorTemplateID, err)
	}
	if auditor.AuditMaxPass != 0 {
		t.Errorf("%s: audit_max_pass = %d, want 0 (the auditor is not audited)", answerAuditorTemplateID, auditor.AuditMaxPass)
	}
	if auditor.Temperature == nil || *auditor.Temperature != answerAuditorTemperature {
		t.Errorf("%s: temperature = %v, want the shipped %v — the auditor's verdict is machine-parsed and decides whether the deliverable ships, so it must not sample",
			answerAuditorTemplateID, auditor.Temperature, answerAuditorTemperature)
	}
	// The snippet rule must not send the auditor after typography. It used to
	// demand the snippet be copied CHARACTER-FOR-CHARACTER with "nothing ...
	// normalized (quotes/whitespace/punctuation as-is)", and q221's pass 3 spent a
	// whole repair turn on a space the producer inserted after "citizens." — a
	// turn that fixes nothing. Content defects stay defects; spacing does not.
	if strings.Contains(auditor.Content, "CHARACTER-FOR-CHARACTER") {
		t.Error("the auditor's snippet rule must judge CONTENT, not character-for-character typography")
	}
	if !strings.Contains(auditor.Content, "Whitespace and punctuation are not content") {
		t.Error("the auditor's snippet rule must state that whitespace and punctuation are never defects")
	}
	if !strings.Contains(auditor.Content, "elision marker") {
		t.Error("an explicit elision marker must be the stated way to quote non-adjacent passages")
	}
	// Elimination has exactly two admissible grounds (a refuting chunk, or the
	// corpus never describing the candidate), and BOTH sides must say so: the
	// producer shipped a self-conceding elimination on q221 ("both satisfy the
	// stated constraints") and the auditor rejected it, correctly, for the only
	// ground its own prompt allowed — a contradiction. One side alone leaves the
	// producer writing a ground the auditor cannot accept, or the auditor
	// accepting a ground the producer was never told to use.
	for _, want := range []string{
		"no corpus content describes it",
		"elimination ground not on the record",
		"eliminated for weakness, not evidence",
		"elimination ground contradicted by cited chunk",
	} {
		if !strings.Contains(auditor.Content, want) {
			t.Errorf("the auditor must define the elimination ground %q", want)
		}
	}
	// A declared tie is a first-class outcome, not something the auditor repairs
	// away: every rival must be grounded, every clause must be checked (not just
	// the first), and a tie can never ride on the `Final Answer` label it
	// contradicts by definition.
	for _, want := range []string{
		"tie declares an ungrounded candidate",
		"tie names no rival",
		"tie declares no reason",
		"tie contradicted by cited chunk",
		"Final Answer claims a tie",
		"tie written as two answers",
		// (b)+(a): a Tested line must state the candidate's fitness (a
		// bibliography line merely names it), and an elimination may not hide
		// inside one — q221's audit PASSed a deliverable whose value rested on
		// the citation line of the rival it silently promoted.
		"Tested line merely names the candidate",
		"elimination disguised as a Tested line",
		"elimination line names no candidate",
		"citation line cannot support a clue",
		// The tie is not merely permitted: the trigger makes it REQUIRED when a
		// grounded rival stands unrefuted and the discriminating constraint is
		// unestablished for the retained candidate.
		"grounded rival not refuted - declare a tie or refute it",
		// The family check is corpus-side on purpose: the competitor rules are
		// deliverable-relative, so a producer that never surfaces the family can
		// pass with any member (x1: [0], PASS, a wrong sibling, 39s).
		"sibling work named by a cited document never tested",
		"answer value appears in no cited evidence",
		// The value's own FORM: a sentence/lyric is not a slot-filler, and a
		// partial name is not a filled full-name slot (q521 and q253 passed both).
		"answer value is not a single entity",
		"answer value is a partial form",
		// A variant tie the corpus can actually weigh: the better-attested
		// spelling ships (q283 shipped the single-source one and lost the point).
		"the better-attested variant was not preferred",
		// The counting rule must sit where the VALUE is formed, not only on the
		// tie path: three q283 re-runs picked a spelling silently (no tie clause
		// at all), so a tie-only rule never fires.
		"the better-attested form is the one that ships",
		// A variant tie is STILL a tie: the count orders WHICH form ships and
		// never the label (q283 run 1 shipped the single-source spelling under a
		// bare `Final Answer` and the audit PASSed it).
		"the count decides WHICH form ships and never the label",
		// The gate reads labels only, so the value-level judgements it used to
		// approximate with text heuristics are the auditor's: grounding in the CITED
		// evidence, and a value resting on a listing entry.
		"answer value rests on a listing entry",
		"The GATE READS LABELS ONLY",
	} {
		if !strings.Contains(auditor.Content, want) {
			t.Errorf("the auditor must define the tie defect %q", want)
		}
	}
	// The gate sends no prechecks any more, and the prompt must not describe
	// them: a stale paragraph re-teaches the auditor to wait for a field the
	// payload never carries, and its absence then reads as "the gate found
	// nothing wrong" instead of "the gate no longer looks". The payload's own
	// shape is pinned in delivery_gate_test.
	for _, gone := range []string{
		"gate_prechecks",
		"gate precheck cleared",
		"answer_line_missing",
		"ungrounded_answer_value",
		"list_only_answer_value",
		"answer_value_missing",
		"citation_only_grounding",
	} {
		if strings.Contains(auditor.Content, gone) {
			t.Errorf("the auditor prompt must not describe the retired precheck machinery (%q)", gone)
		}
	}
	for _, id := range []string{"smart-reasoning", "smart-grep", "smart-grep-bm25"} {
		tmpl, err := resolveTemplateFor(id)
		if err != nil {
			t.Fatalf("resolve %s: %v", id, err)
		}
		if tmpl.Temperature != nil {
			t.Errorf("%s: temperature = %v, want nil — this knob is the auditor's, and pinning a producer's sampling from this file would silently take away the exploration its temperature buys", id, *tmpl.Temperature)
		}
		for _, want := range []string{
			"no corpus content describes it",
			"a named, grounded rival you cannot refute is RETAINED",
			"once per rival",
			"EVERY tied candidate must be GROUNDED",
			"asserting SUPPORT",
			"elimination argument NEVER rides on a",
			"THE TRIGGER: declare the tie as soon as a rival is GROUNDED",
			"those siblings are CANDIDATES",
		} {
			if !strings.Contains(tmpl.Content, want) {
				t.Errorf("%s: the Eliminated spec must offer the absence ground (%q) and forbid dropping a candidate for weakness", id, want)
			}
		}
	}
}

// TestAuditTemperatureKnob pins the auditor's sampling policy: an undeclared
// temperature means 0 (the auditor must not sample), a declared one wins, and a
// config without an auditor template still yields 0 rather than making every
// caller invent a fallback.
func TestAuditTemperatureKnob(t *testing.T) {
	cases := []struct {
		name string
		yaml string
		want float64
	}{
		{"undeclared pins zero", auditorTemplateYAML(""), 0},
		{"declared wins", auditorTemplateYAML("    temperature: 0.4\n"), 0.4},
		{"no auditor template still zero", "templates:\n  - id: smart-grep\n    name: x\n    tools:\n      - think\n    content: |\n      P\n", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "agentic_rag.yaml")
			if err := os.WriteFile(path, []byte(tc.yaml), 0o600); err != nil {
				t.Fatalf("write temp config: %v", err)
			}
			t.Setenv("AGENTIC_RAG_CONFIG", path)
			configMu.Lock()
			cachedFile = nil
			configMu.Unlock()
			t.Cleanup(func() {
				configMu.Lock()
				cachedFile = nil
				configMu.Unlock()
			})
			if got := AuditTemperature(); got != tc.want {
				t.Errorf("AuditTemperature() = %v, want %v", got, tc.want)
			}
		})
	}
}

// auditorTemplateYAML builds a minimal config whose only template is the
// auditor's, with tempLine (possibly empty) spliced into its header.
func auditorTemplateYAML(tempLine string) string {
	return "templates:\n  - id: " + answerAuditorTemplateID + "\n    name: Answer Auditor Agent\n" +
		tempLine + "    tools:\n      - list_chunks\n    content: |\n      AUDIT\n"
}
