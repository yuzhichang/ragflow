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
	"context"
	"strings"
	"testing"

	"ragflow/internal/entity/models"
)

// TestRun_NilModel: Run must reject a nil model up front.
func TestRun_NilModel(t *testing.T) {
	_, err := Run(context.Background(), Input{Model: nil})
	if err == nil {
		t.Fatal("expected error for nil model")
	}
}

// TestAuditModelFor pins which model the auditor runs on: its own when the
// caller built one (pinned sampling, separate failover state), the producer's
// when not — and nil when neither, so Run's up-front nil check owns that error.
func TestAuditModelFor(t *testing.T) {
	producer := &models.EinoChatModel{}
	auditor := &models.EinoChatModel{}

	if got := auditModelFor(Input{Model: producer}); got != producer {
		t.Error("with no AuditModel the auditor must run on the producer's model")
	}
	if got := auditModelFor(Input{Model: producer, AuditModel: auditor}); got != auditor {
		t.Error("with an AuditModel the auditor must run on ITS OWN instance, not the producer's")
	}
	if got := auditModelFor(Input{Model: producer, AuditModel: nil}); got != producer {
		t.Error("an explicit nil AuditModel must fall back to Model")
	}
	if got := auditModelFor(Input{}); got != nil {
		t.Error("with no model at all the fallback must stay nil")
	}
}

// TestPrompt: the prompt must declare the six tools and contain no removed
// tool references (get_document_info / web_search / query_knowledge_graph) and
// no leftover placeholders.
func TestPrompt(t *testing.T) {
	p := Prompt()
	for _, want := range []string{
		"grep_chunks", "search_chunks", "list_chunks",
		"todo_write", "think", "run_javascript",
	} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt must mention %q", want)
		}
	}
	for _, banned := range []string{
		"get_document_info", "query_knowledge_graph", "web_search", "web_fetch",
		"{{", "}}",
	} {
		if strings.Contains(p, banned) {
			t.Errorf("prompt must not contain %q", banned)
		}
	}
}
