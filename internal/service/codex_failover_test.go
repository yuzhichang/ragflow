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
	modelModule "ragflow/internal/entity/models"
	"ragflow/internal/mcp"
	serverconfig "ragflow/internal/server/config"

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

// TestCodexCandidateUsable pins the bar a failover member must clear: a Responses-capable
// provider (when an allowlist is configured) whose instance declares a base_url, since the
// thread declares exactly that URL as its endpoint.
func TestCodexCandidateUsable(t *testing.T) {
	saved := getCodexConfig()
	defer SetCodexConfig(saved)

	driver := modelModule.NewDummyModel(nil, modelModule.URLSuffix{}) // Name() == "dummy"

	// No allowlist => only the base_url matters.
	SetCodexConfig(nil)
	if _, ok := codexCandidateUsable(nil); ok {
		t.Fatal("a nil target must not be usable")
	}
	if _, ok := codexCandidateUsable(&ResolvedModelTarget{Driver: driver}); ok {
		t.Fatal("an empty base_url must not be usable")
	}
	if _, ok := codexCandidateUsable(&ResolvedModelTarget{Driver: driver, BaseURL: "https://x/v1"}); !ok {
		t.Fatal("a resolvable target with a base_url must be usable when no allowlist is set")
	}

	// An allowlist that excludes this provider rejects it even with a base_url.
	SetCodexConfig(&serverconfig.CodexConfig{ResponsesProviders: []string{"someoneelse"}})
	if _, ok := codexCandidateUsable(&ResolvedModelTarget{Driver: driver, BaseURL: "https://x/v1"}); ok {
		t.Fatal("a provider outside the allowlist must not be usable")
	}
	// ...and one that names it accepts it.
	SetCodexConfig(&serverconfig.CodexConfig{ResponsesProviders: []string{"dummy"}})
	if _, ok := codexCandidateUsable(&ResolvedModelTarget{Driver: driver, BaseURL: "https://x/v1"}); !ok {
		t.Fatal("a provider inside the allowlist must be usable")
	}
}

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
