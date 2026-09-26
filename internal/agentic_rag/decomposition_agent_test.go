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

// The two-stage Run lives or dies with the wiring: the decomposition stage's
// template must exist, own ONLY the checker (the one thing the stage must
// never do is retrieve), carry the conversational escape marker, and the
// explorer must have lost the checker from ITS toolset - a prompt that says
// "do not call check_decomposition" next to a toolset that offers it is the
// contradiction this stage exists to remove.
func TestQuestionDecompositionStageWiring(t *testing.T) {
	// The wiring test reads the REPO conf (not a fixture): it asserts the real
	// deployment shape, so it loads the same file the server does, through the
	// same AGENTIC_RAG_CONFIG + cache-reset dance the config tests use.
	raw, err := os.ReadFile(filepath.Join("..", "..", "conf", "agentic_rag.yaml"))
	if err != nil {
		t.Skipf("prompt not available here: %v", err)
	}
	path := filepath.Join(t.TempDir(), "agentic_rag.yaml")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
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
	tmpl, err := resolveTemplateFor(questionDecompositionTemplateID)
	if err != nil {
		t.Fatalf("the question-decomposition template is missing: %v", err)
	}
	if len(tmpl.Tools) != 1 || tmpl.Tools[0] != "check_decomposition" {
		t.Fatalf("the decomposition stage must own exactly the checker, got %v", tmpl.Tools)
	}
	if !strings.Contains(tmpl.Content, decompositionNoPlanMarker) {
		t.Fatalf("the planner prompt must carry the conversational escape marker %q", decompositionNoPlanMarker)
	}
	if !strings.Contains(tmpl.Content, "Example (two blocks") ||
		!strings.Contains(tmpl.Content, "Example (four blocks") {
		t.Fatalf("the worked examples must travel with the writer, not the consumer")
	}
	explorer, err := resolveTemplateFor("smart-reasoning")
	if err != nil {
		t.Fatalf("smart-reasoning missing: %v", err)
	}
	for _, name := range explorer.Tools {
		if name == "check_decomposition" {
			t.Fatalf("the explorer still holds the checker - the two-stage split is not real")
		}
	}
	if !strings.Contains(explorer.Content, "## The question decomposition") ||
		!strings.Contains(explorer.Content, "never re-decompose") {
		t.Fatalf("the explorer prompt must consume the pinned plan, not promise to write one")
	}
}

// extractPlan takes everything from the first block heading down — plus a
// leading `## Resolved question` line, which is part of the plan because every
// downstream consumer reads it as the question. Prose or a code fence before
// the plan is the agent's business, the plan itself must be clean for
// checkDecomposition, and a message with no heading yields nothing.
func TestExtractPlan(t *testing.T) {
	plan := "### Sub-question 1: ?x is the one — slot: name\n- Binds: ?x\n"
	for name, in := range map[string]string{
		"bare":           plan,
		"fenced":         "```\n" + plan + "```\n",
		"prose-prefixed": "Here is the plan.\n" + plan,
	} {
		if got := extractPlan(in); !strings.HasPrefix(got, "### Sub-question 1") || !strings.Contains(got, "- Binds: ?x") {
			t.Fatalf("%s: extractPlan mangled the plan: %q", name, got)
		}
	}
	if got := extractPlan("no plan here, just prose"); got != "" {
		t.Fatalf("a plan-less message must extract to nothing, got %q", got)
	}
	resolved := "## Resolved question: the spouse of the painter from turn one\n\n" + plan
	got := extractPlan(resolved)
	if !strings.HasPrefix(got, "## Resolved question") || !strings.Contains(got, "- Binds: ?x") {
		t.Fatalf("extractPlan must keep the resolved-question header: %q", got)
	}
	if q := extractResolvedQuestion(got); q != "the spouse of the painter from turn one" {
		t.Fatalf("extractResolvedQuestion = %q", q)
	}
	if q := extractResolvedQuestion(plan); q != "" {
		t.Fatalf("a plan without the header must resolve to nothing, got %q", q)
	}
}
