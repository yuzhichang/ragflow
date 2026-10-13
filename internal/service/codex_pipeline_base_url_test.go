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

package service

import (
	"strings"
	"testing"
)

// TestNormalizeResponsesBaseURL pins that a version-less provider base gains "/v1" for the
// Codex Responses endpoint, while a base that already names a version is untouched.
func TestNormalizeResponsesBaseURL(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"minimax appends v1", "https://api.minimaxi.com", "https://api.minimaxi.com/v1"},
		{"trailing slash", "https://api.minimaxi.com/", "https://api.minimaxi.com/v1"},
		{"openai kept", "https://api.openai.com/v1", "https://api.openai.com/v1"},
		{"versioned kept", "https://api.siliconflow.cn/v1", "https://api.siliconflow.cn/v1"},
		{"beta version kept", "https://generativelanguage.googleapis.com/v1beta", "https://generativelanguage.googleapis.com/v1beta"},
		{"query preserved", "https://x.example.com/openai/deployments/d?api-version=2024-01-01", "https://x.example.com/openai/deployments/d/v1?api-version=2024-01-01"},
		{"empty stays empty", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeResponsesBaseURL(tc.in); got != tc.want {
				t.Fatalf("normalizeResponsesBaseURL(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// TestCodexCitationContractIsGenericAndForbidsTheShell pins the two properties the base
// instructions must have now that web search is a PER-TURN decision while the instructions
// are baked into a thread at create (and not reloaded by resume):
//
//   - they are generic about the toolset — the MCP tool list is the single source of truth
//     for whether web search exists, so the prompt must not claim either way, and must stay
//     identical across a web-search toggle (otherwise prompt and tool list are two copies
//     of one fact that can disagree);
//   - they forbid the shell as a path to the web, so the model — which sees an advertised
//     shell in its environment — does not burn the turn on `curl`.
func TestCodexCitationContractIsGenericAndForbidsTheShell(t *testing.T) {
	quoted := codexCitationContract(true)

	// Capability comes from the tool list, not the prompt: conditional everywhere.
	if !strings.Contains(quoted, "authoritative") || !strings.Contains(quoted, "tool list") {
		t.Errorf("contract does not defer capability to the MCP tool list:\n%s", quoted)
	}
	if !strings.Contains(quoted, "no shell and no internet access") {
		t.Errorf("contract does not state the shell/network gap:\n%s", quoted)
	}
	for _, must := range []string{"exec_command", "curl", "not a way to reach the web"} {
		if !strings.Contains(quoted, must) {
			t.Errorf("contract does not forbid the shell clearly (%q missing):\n%s", must, quoted)
		}
	}
	// It must not assert web search either way — that is the tool list's job.
	if strings.Contains(quoted, "You have") {
		t.Errorf("contract asserts capabilities rather than deferring to the tool list:\n%s", quoted)
	}
	if !strings.Contains(quoted, "when this conversation enables web search") {
		t.Errorf("contract does not present web search conditionally:\n%s", quoted)
	}

	// Both provenance forms are described, the web one conditionally on the tool.
	if !strings.Contains(quoted, "(doc: <doc_name>") || !strings.Contains(quoted, "(source: web") {
		t.Errorf("contract omits a provenance form:\n%s", quoted)
	}
	if !strings.Contains(quoted, "returned by the web search tool") {
		t.Errorf("the web form is not tied to the web tool:\n%s", quoted)
	}

	// Quote off still states capabilities and forbids the shell; it only drops the format.
	plain := codexCitationContract(false)
	if !strings.Contains(plain, "no shell and no internet access") || !strings.Contains(plain, "curl") {
		t.Errorf("quote-off contract dropped the capability statement:\n%s", plain)
	}
	if strings.Contains(plain, "(doc: <doc_name>") {
		t.Errorf("quote-off contract should not prescribe a provenance format:\n%s", plain)
	}
}
