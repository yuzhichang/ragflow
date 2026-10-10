//go:build integration

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

package codexagent_test

import (
	"context"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	codexgo "github.com/zealbase/codex-app-server-go"

	"ragflow/internal/codexagent"
	"ragflow/internal/mcp"
)

// TestRunnerAgainstLocalCodex exercises the whole runner against the local codex
// app-server: MCP injection, a real turn, and thread reuse on the second turn.
//
//	RAGFLOW_CODEX_IT=1 OPENAI_BASE_URL=... OPENAI_API_KEY=... OPENAI_MODEL=... \
//	  go test -tags integration -run TestRunnerAgainstLocalCodex ./internal/codexagent/
func TestRunnerAgainstLocalCodex(t *testing.T) {
	if os.Getenv("RAGFLOW_CODEX_IT") != "1" {
		t.Skip("set RAGFLOW_CODEX_IT=1 to run against the local codex app-server")
	}
	bin, err := codexgo.FindBinary()
	if err != nil {
		t.Skipf("codex binary not found: %v", err)
	}

	// A live MCP bridge, scoped to one KB, behind an httptest server.
	tickets := mcp.NewCodexTickets(time.Hour)
	token, err := tickets.Mint(mcp.CodexScope{TenantID: "it-tenant", DatasetIDs: []string{"it-kb"}})
	if err != nil {
		t.Fatalf("mint ticket: %v", err)
	}
	srv := httptest.NewServer(mcp.NewCodexHandler(tickets))
	defer srv.Close()
	mcpURL := srv.URL + "/mcp/codex/" + token

	client, err := codexgo.New(codexgo.WithStdioProcess(bin, "app-server"))
	if err != nil {
		t.Fatalf("codex client: %v", err)
	}
	defer client.Close()

	// A working model provider must be supplied per-thread (the shared Codex's own
	// default provider may be unconfigured/unreachable in the test environment).
	baseURL := os.Getenv("OPENAI_BASE_URL")
	apiKey := os.Getenv("OPENAI_API_KEY")
	model := os.Getenv("OPENAI_MODEL")
	if baseURL == "" || apiKey == "" || model == "" {
		t.Skip("set OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL to a Responses-API provider")
	}

	runner := codexagent.NewRunner(client, codexagent.NewMemoryThreadStore())
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	base := codexagent.TurnRequest{
		TenantID:        "it-tenant",
		SessionID:       "it-session",
		DatasetIDs:      []string{"it-kb"},
		Model:           model,
		ProviderID:      "minimax",
		ProviderBaseURL: baseURL,
		ProviderToken:   apiKey,
		MCPURL:          mcpURL,
		Ticket:          token,
		TurnTimeout:     3 * time.Minute,
		Instructions:    "Answer concisely.",
	}

	first := base
	first.Messages = []map[string]interface{}{{"role": "user", "content": "Reply with exactly: PONG"}}
	out1, err := runner.RunTurn(ctx, first)
	if err != nil {
		t.Fatalf("first turn: %v", err)
	}
	if out1.ThreadID == "" || !out1.Rebuilt {
		t.Fatalf("first turn should create a thread: %+v", out1)
	}
	t.Logf("turn1 thread=%s rebuilt=%v content=%q", out1.ThreadID, out1.Rebuilt, out1.Content)

	// Second turn: same scope, extended history -> must reuse the thread (resume).
	second := base
	second.Messages = []map[string]interface{}{
		{"role": "user", "content": "Reply with exactly: PONG"},
		{"role": "assistant", "content": out1.Content},
		{"role": "user", "content": "Now reply with exactly: PONG2"},
	}
	out2, err := runner.RunTurn(ctx, second)
	if err != nil {
		t.Fatalf("second turn: %v", err)
	}
	if out2.ThreadID != out1.ThreadID {
		t.Fatalf("second turn should reuse thread %s, got %s", out1.ThreadID, out2.ThreadID)
	}
	if out2.Rebuilt {
		t.Fatal("second turn with unchanged scope must NOT rebuild the thread")
	}
	t.Logf("turn2 thread=%s rebuilt=%v content=%q", out2.ThreadID, out2.Rebuilt, out2.Content)
}
