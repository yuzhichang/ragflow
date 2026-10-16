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
	"context"
	"errors"
	"testing"

	"ragflow/internal/codexagent"
	"ragflow/internal/mcp"

	codexgo "github.com/zealbase/codex-app-server-go"
)

// TestCodexFailoverEligible pins when a failed turn may be retried on the next provider. The
// refusals matter as much as the acceptance: a held session lock is not a provider failure, a
// cancelled request has nobody to answer, an interrupted turn was stopped on purpose, and once
// answer text has reached the client a retry would append a second answer.
func TestCodexFailoverEligible(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()

	providerFailure := &codexagent.TurnError{Status: "failed", Kind: codexgo.CodexErrorInternalServer, HTTPStatus: 500}
	cases := []struct {
		name     string
		ctx      context.Context
		runErr   error
		emitted  int
		wantOK   bool
		wantKind codexgo.CodexErrorKind
	}{
		{"provider failure", context.Background(), providerFailure, 0, true, codexgo.CodexErrorInternalServer},
		{"no error", context.Background(), nil, 0, false, ""},
		{"session busy is a lock, not a provider", context.Background(), mcp.ErrCodexTurnBusy, 0, false, ""},
		{"cancelled request", cancelled, providerFailure, 0, false, ""},
		{"answer already streamed", context.Background(), providerFailure, 12, false, ""},
		{"interrupted turn", context.Background(),
			&codexagent.TurnError{Status: "interrupted", Kind: codexgo.CodexErrorInternalServer}, 0, false, ""},
		{"catch-all other", context.Background(),
			&codexagent.TurnError{Kind: codexgo.CodexErrorOther, Message: "unknown model"}, 0, false, ""},
		{"untyped error", context.Background(), errors.New("plain"), 0, false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			turnErr, ok := codexFailoverEligible(tc.ctx, tc.runErr, tc.emitted)
			if ok != tc.wantOK {
				t.Fatalf("codexFailoverEligible() ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && turnErr.Kind != tc.wantKind {
				t.Fatalf("kind = %q, want %q", turnErr.Kind, tc.wantKind)
			}
			if !ok && turnErr != nil {
				t.Fatalf("a refused failure must not return a TurnError, got %v", turnErr)
			}
		})
	}
}

// TestCodexResponsesUsable (moved to codex_responses_probe_test.go) pins the bar a mode-8
// target must clear; it now covers both the known-capable fast path and the dynamic probe.

// TestCandidateKey pins the dedup key: two candidates are the same only when they share both the
// endpoint and the model, so the same model on another endpoint is still a distinct failover
// member.
func TestCandidateKey(t *testing.T) {
	base := &ResolvedModelTarget{BaseURL: "https://a/v1", ModelName: "m1"}
	if candidateKey(base) != candidateKey(&ResolvedModelTarget{BaseURL: "https://a/v1", ModelName: "m1"}) {
		t.Fatal("same endpoint and model must share a key")
	}
	if candidateKey(base) == candidateKey(&ResolvedModelTarget{BaseURL: "https://b/v1", ModelName: "m1"}) {
		t.Fatal("a different endpoint is a different candidate")
	}
	if candidateKey(base) == candidateKey(&ResolvedModelTarget{BaseURL: "https://a/v1", ModelName: "m2"}) {
		t.Fatal("a different model is a different candidate")
	}
}

// TestCodexTargetLabel pins the human label the failover narration uses. Candidates routinely
// share provider and model and differ only by instance, so the instance must qualify the model —
// otherwise the line reads "<model> failed; retrying on <model>".
func TestCodexTargetLabel(t *testing.T) {
	cases := []struct {
		name   string
		target *ResolvedModelTarget
		want   string
	}{
		{"instance qualified", &ResolvedModelTarget{ModelName: "MiniMax-M3", InstanceName: "zyf"}, "MiniMax-M3(zyf)"},
		{"no instance name", &ResolvedModelTarget{ModelName: "MiniMax-M3"}, "MiniMax-M3"},
		{"nil target", nil, "unknown model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := codexTargetLabel(tc.target); got != tc.want {
				t.Fatalf("codexTargetLabel() = %q, want %q", got, tc.want)
			}
		})
	}
}
