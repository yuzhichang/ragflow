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

import "testing"

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
