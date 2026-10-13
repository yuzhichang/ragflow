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
	"strings"
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

// mcpToolRoundTrip reads the call half of an MCP tool call, which is what makes a
// retrieval step visible: without the tool name and its arguments the thinking stream
// cannot say what the model reached for.
func TestMcpToolRoundTrip(t *testing.T) {
	item := func(payload map[string]any) codexgo.Item {
		t.Helper()
		raw, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		return codexgo.Item{ID: "c1", Kind: codexgo.ItemKindMCPToolCall, Payload: raw}
	}

	got, ok := mcpToolRoundTrip(item(map[string]any{
		"server":     "ragflow",
		"tool":       "search_bm25_chunks",
		"status":     "completed",
		"arguments":  map[string]any{"queries": []string{"曹操 生卒年", "孙权 生卒年"}},
		"durationMs": 340,
	}))
	if !ok {
		t.Fatal("a well-formed mcpToolCall must parse")
	}
	if got.Tool != "search_bm25_chunks" || got.Status != ToolStatusCompleted || got.DurationMS != 340 {
		t.Fatalf("round trip = %+v", got)
	}
	if got.Args != `{"queries":["曹操 生卒年","孙权 生卒年"]}` {
		t.Fatalf("args must be compact JSON: %q", got.Args)
	}

	// No status yet (the item/started half) reads as running, not as an empty status.
	if running, ok := mcpToolRoundTrip(item(map[string]any{"tool": "grep_chunks"})); !ok || running.Status != ToolStatusRunning {
		t.Fatalf("missing status must default to running: %+v ok=%v", running, ok)
	}
	// An item with no tool name is not a tool call.
	if _, ok := mcpToolRoundTrip(item(map[string]any{"server": "ragflow"})); ok {
		t.Fatal("an item without a tool name must not parse as a tool call")
	}
}

// compactArgs keeps the thinking line on one line and bounded, so a long query list
// cannot flood the panel.
func TestCompactArgs(t *testing.T) {
	if got := compactArgs(json.RawMessage(`{"a": 1,  "b":"x"}`)); got != `{"a":1,"b":"x"}` {
		t.Fatalf("compactArgs = %q", got)
	}
	if got := compactArgs(nil); got != "" {
		t.Fatalf("nil args = %q, want empty", got)
	}
	if got := compactArgs(json.RawMessage(`null`)); got != "" {
		t.Fatalf("null args = %q, want empty", got)
	}
	long := json.RawMessage(`{"q":"` + strings.Repeat("x", 400) + `"}`)
	if got := compactArgs(long); len(got) > 303 || !strings.HasSuffix(got, "...") {
		t.Fatalf("long args not truncated: len=%d tail=%q", len(got), got[len(got)-8:])
	}
}

// toolTally is what makes a mode-8 turn report tool_call_counts / tool_call_errors in the
// same shape as the in-process engines: a benchmark reads them without caring which engine
// produced the row. A call is counted once (the item/started half, or the completed half
// when the provider sends no started); an outcome only records failures.
func TestToolTally(t *testing.T) {
	var tt toolTally
	tt.call("search_bm25_chunks")
	tt.call("search_bm25_chunks")
	tt.call("list_chunks")
	tt.call("") // no tool name: not a call
	tt.outcome(RoundTrip{Tool: "search_bm25_chunks", Status: ToolStatusCompleted})
	tt.outcome(RoundTrip{Tool: "list_chunks", Status: ToolStatusFailed})

	if got := tt.counts["search_bm25_chunks"]; got != 2 {
		t.Errorf("search_bm25_chunks = %d, want 2", got)
	}
	if got := tt.counts["list_chunks"]; got != 1 {
		t.Errorf("list_chunks = %d, want 1", got)
	}
	if len(tt.counts) != 2 {
		t.Errorf("counts = %v, want two tools", tt.counts)
	}
	if got := tt.errors["list_chunks"]; got != 1 {
		t.Errorf("errors[list_chunks] = %d, want 1", got)
	}
	if len(tt.errors) != 1 {
		t.Errorf("a completed call must not be an error: %v", tt.errors)
	}
}

// The v2 status is camelCase ("inProgress"), while the narration branches on our own
// constant: without normalization a call would be narrated as a RESULT.
func TestMcpToolRoundTripNormalizesInProgress(t *testing.T) {
	raw, _ := json.Marshal(map[string]any{"tool": "grep_chunks", "status": "inProgress"})
	rt, ok := mcpToolRoundTrip(codexgo.Item{ID: "c", Kind: codexgo.ItemKindMCPToolCall, Payload: raw})
	if !ok || rt.Status != ToolStatusRunning {
		t.Fatalf("inProgress must normalize to running: %+v ok=%v", rt, ok)
	}
}
