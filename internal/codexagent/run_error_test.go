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

package codexagent

import (
	"encoding/json"
	"errors"
	"testing"

	codexgo "github.com/zealbase/codex-app-server-go"
)

// TestTurnErrorDetailPrefersTurnError pins that a failed turn surfaces the turn's own
// error payload (which carries the real cause, e.g. the provider's HTTP status) rather
// than the bare, often empty, `error` notification that arrives first.
func TestTurnErrorDetailPrefersTurnError(t *testing.T) {
	const detailed = `{"message":"unexpected status 404 Not Found, url: https://api.minimaxi.com/responses"}`
	turn := &codexgo.Turn{Error: json.RawMessage(detailed)}

	if got := turnErrorDetail(turn, errors.New("(code=\"\"): ")); got != detailed {
		t.Fatalf("turn error detail: got %q, want %q", got, detailed)
	}
}

func TestTurnErrorDetailFallsBackToStreamError(t *testing.T) {
	streamErr := errors.New("codexagent: the Codex server reported an error with no detail")

	if got := turnErrorDetail(nil, streamErr); got != streamErr.Error() {
		t.Fatalf("fallback to stream error: got %q, want %q", got, streamErr.Error())
	}
	if got := turnErrorDetail(&codexgo.Turn{}, streamErr); got != streamErr.Error() {
		t.Fatalf("empty turn error falls back: got %q, want %q", got, streamErr.Error())
	}
}

func TestTurnErrorDetailPlaceholder(t *testing.T) {
	if got := turnErrorDetail(nil, nil); got != "no error detail" {
		t.Fatalf("placeholder: got %q, want %q", got, "no error detail")
	}
}

// reasoningItemText backs the completed-item fallback: a provider that does not stream
// reasoning deltas sends one finished reasoning item instead. Its raw reasoning rides
// `content`; `summary` is the model-written summary and is often empty for those
// providers, so content wins and summary is the fallback.
func TestReasoningItemText(t *testing.T) {
	item := func(payload map[string]any) codexgo.Item {
		t.Helper()
		raw, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal payload: %v", err)
		}
		return codexgo.Item{ID: "r1", Kind: codexgo.ItemKindReasoning, Payload: raw}
	}

	if got := reasoningItemText(item(map[string]any{
		"summary": []string{"a summary"},
		"content": []string{"raw ", "reasoning"},
	})); got != "raw reasoning" {
		t.Fatalf("content must win: got %q", got)
	}
	if got := reasoningItemText(item(map[string]any{"summary": []string{"only summary"}})); got != "only summary" {
		t.Fatalf("summary must be the fallback: got %q", got)
	}
	if got := reasoningItemText(codexgo.Item{Kind: codexgo.ItemKindReasoning}); got != "" {
		t.Fatalf("an empty payload must yield no text: got %q", got)
	}
}

// usageOrNil distinguishes "the server reported no usage" from "a turn that cost zero
// tokens": a caller must not publish an all-zero usage block in the former case.
func TestUsageOrNil(t *testing.T) {
	if usageOrNil(TurnUsage{}) != nil {
		t.Fatal("no model calls must yield nil usage, not a zero block")
	}
	got := usageOrNil(TurnUsage{PromptTokens: 3, CompletionTokens: 4, TotalTokens: 7, LLMCalls: 1})
	if got == nil || got.TotalTokens != 7 || got.LLMCalls != 1 {
		t.Fatalf("usageOrNil dropped a reported usage: %+v", got)
	}
}
