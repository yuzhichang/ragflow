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

package mcp_test

import (
	"context"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	codexgo "github.com/zealbase/codex-app-server-go"
	"gorm.io/gorm"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/codexagent"
	"ragflow/internal/mcp"
)

const (
	cannedChunkID = "chunk-e2e-0001"
	cannedDocID   = "doc-e2e-0001"
)

type fakeRetrieval struct{}

func (fakeRetrieval) Search(_ context.Context, _ *gorm.DB, _ runtime.RetrievalRequest) ([]runtime.RetrievalChunk, error) {
	return []runtime.RetrievalChunk{{
		ID: cannedChunkID, Content: "hello from the knowledge base",
		DocumentID: cannedDocID, DocumentName: "e2e.md", DatasetID: "kb-e2e", PageNum: 1, Score: 0.91,
	}}, nil
}

// TestCodexCallsBridgeTool (M1 E2E) drives codex to call the bridge's retrieval tool and
// asserts the bridge served a hit. Config is env-driven while investigating codex's MCP
// tool-call approval.
func TestCodexCallsBridgeTool(t *testing.T) {
	if os.Getenv("RAGFLOW_CODEX_IT") != "1" {
		t.Skip("set RAGFLOW_CODEX_IT=1")
	}
	bin, err := codexgo.FindBinary()
	if err != nil {
		t.Skipf("codex binary not found: %v", err)
	}
	baseURL, apiKey, model := os.Getenv("OPENAI_BASE_URL"), os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_MODEL")
	if baseURL == "" || apiKey == "" || model == "" {
		t.Skip("set OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL")
	}

	runtime.SetRetrievalService(fakeRetrieval{})
	defer runtime.SetRetrievalService(nil)

	tickets := mcp.NewCodexTickets(time.Hour)
	ledger := mcp.NewChunkLedger()
	token, err := tickets.Mint(mcp.CodexScope{TenantID: "e2e-tenant", DatasetIDs: []string{"kb-e2e"}, Ledger: ledger})
	if err != nil {
		t.Fatalf("mint ticket: %v", err)
	}
	srv := httptest.NewServer(mcp.NewCodexHandler(tickets))
	defer srv.Close()

	// codexagent.Dispatcher installs the elicitation handler that accepts MCP tool calls;
	// without it codex declines them (see internal/codexagent/approval.go).
	client, err := codexgo.New(
		codexgo.WithStdioProcess(bin, "app-server"),
		codexgo.WithRequestHandler(codexagent.Dispatcher()),
	)
	if err != nil {
		t.Fatalf("codex client: %v", err)
	}
	defer client.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	opts := []codexgo.ThreadOption{
		codexgo.WithThreadModel(model),
		codexgo.WithThreadModelProvider("minimax"),
		codexgo.WithThreadConfigOverride("model_providers.minimax", map[string]interface{}{
			"name": "minimax", "base_url": baseURL, "wire_api": "responses", "experimental_bearer_token": apiKey,
		}),
		codexgo.WithThreadConfigOverride("mcp_servers.ragflow", map[string]interface{}{
			"url": srv.URL + "/mcp/codex/" + token,
		}),
		codexgo.WithThreadSandbox(codexgo.SandboxReadOnly),
		codexgo.WithThreadApprovalPolicy(codexgo.ApprovalUntrusted()),
	}

	th, err := client.StartThread(ctx, opts...)
	if err != nil {
		t.Fatalf("StartThread: %v", err)
	}
	defer th.Close()

	prompt := "Use the MCP tool `search_semantic_chunks` from the `ragflow` server with the " +
		"argument query=\"hello\". Then reply with only the chunk_id value that tool returned."
	ch, err := th.RunStreamed(ctx, prompt)
	if err != nil {
		t.Fatalf("RunStreamed: %v", err)
	}
	answer := ""
	for ev := range ch {
		switch e := ev.Raw.(type) {
		case codexgo.ItemAgentMessageDeltaEvent:
			answer += e.Text
		case codexgo.TurnCompletedEvent:
			if answer == "" && e.Turn != nil {
				answer = (&codexgo.TurnResult{Turn: *e.Turn}).FinalAgentText()
			}
		case codexgo.ErrorEvent:
			t.Fatalf("turn error: code=%q msg=%q", e.Code, e.Message)
		}
	}
	t.Logf("answer=%q ledger.calls=%v ledger.chunks=%v", answer, ledger.ToolCallCounts(), ledger.IDSet())
	if ledger.ToolCallCounts()["search_semantic_chunks"] < 1 {
		t.Fatalf("codex did not call the tool: %v", ledger.ToolCallCounts())
	}
	if _, ok := ledger.IDSet()[cannedChunkID]; !ok {
		t.Fatalf("served chunk not recorded: %v", ledger.IDSet())
	}
}
