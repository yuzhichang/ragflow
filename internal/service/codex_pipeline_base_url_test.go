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

// TestCodexCitationContractStatesCapabilities pins the base instructions' capability
// statement: with web search off the model must be told it has no shell and no internet
// (it otherwise tries `curl` and re-tries after each refusal, instead of reading the
// corpus), and the web-passage form must not be described — naming it invites the very
// search the conversation turned off. With web search on, the web tool and its tuple are
// described instead of the prohibition.
func TestCodexCitationContractStatesCapabilities(t *testing.T) {
	off := codexCitationContract(true, false)
	if !strings.Contains(off, "no shell and no internet access") {
		t.Errorf("web-search-off contract does not state the capability gap:\n%s", off)
	}
	if strings.Contains(off, "web search tool") || strings.Contains(off, "source: web") {
		t.Errorf("web-search-off contract still advertises web search:\n%s", off)
	}
	if !strings.Contains(off, "(doc: <doc_name>") {
		t.Errorf("web-search-off contract lost the corpus provenance form:\n%s", off)
	}

	on := codexCitationContract(true, true)
	if !strings.Contains(on, "web search tool") || !strings.Contains(on, "source: web") {
		t.Errorf("web-search-on contract omits the web tool/tuple:\n%s", on)
	}
	if strings.Contains(on, "no shell and no internet access") {
		t.Errorf("web-search-on contract claims there is no internet:\n%s", on)
	}

	// Quote off still states the capabilities; it only drops the provenance format.
	plain := codexCitationContract(false, false)
	if !strings.Contains(plain, "no shell and no internet access") {
		t.Errorf("quote-off contract does not state the capability gap:\n%s", plain)
	}
	if strings.Contains(plain, "(doc: <doc_name>") {
		t.Errorf("quote-off contract should not prescribe a provenance format:\n%s", plain)
	}
}
