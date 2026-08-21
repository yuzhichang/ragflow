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
}
