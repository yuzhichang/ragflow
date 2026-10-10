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

package mcp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestCodexTicketsMintResolveRevoke(t *testing.T) {
	tickets := NewCodexTickets(time.Hour)
	scope := CodexScope{TenantID: "t1", DatasetIDs: []string{"kb1", "kb2"}}
	token, err := tickets.Mint(scope)
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	if token == "" {
		t.Fatal("Mint returned an empty token")
	}

	got, ok := tickets.Resolve(token)
	if !ok || got.TenantID != "t1" || len(got.DatasetIDs) != 2 {
		t.Fatalf("Resolve = %+v, %v", got, ok)
	}

	tickets.Revoke(token)
	if _, ok := tickets.Resolve(token); ok {
		t.Fatal("revoked ticket still resolves")
	}
	if _, ok := tickets.Resolve(""); ok {
		t.Fatal("empty token resolves")
	}
	if _, ok := tickets.Resolve("never-minted"); ok {
		t.Fatal("unknown token resolves")
	}
}

func TestCodexTicketExpiry(t *testing.T) {
	tickets := NewCodexTickets(time.Minute)
	now := time.Now()
	tickets.now = func() time.Time { return now }

	token, err := tickets.Mint(CodexScope{TenantID: "t"})
	if err != nil {
		t.Fatalf("Mint: %v", err)
	}
	if _, ok := tickets.Resolve(token); !ok {
		t.Fatal("fresh ticket did not resolve")
	}
	now = now.Add(2 * time.Minute)
	if _, ok := tickets.Resolve(token); ok {
		t.Fatal("expired ticket still resolves")
	}
}

// connectSession wires an in-memory MCP client to the given server.
func connectSession(t *testing.T, server *sdk.Server) *sdk.ClientSession {
	t.Helper()
	ctx := context.Background()
	client := sdk.NewClient(&sdk.Implementation{Name: "codex-test", Version: "0"}, nil)
	st, ct := sdk.NewInMemoryTransports()
	serverSession, err := server.Connect(ctx, st, nil)
	if err != nil {
		t.Fatalf("server.Connect: %v", err)
	}
	clientSession, err := client.Connect(ctx, ct, nil)
	if err != nil {
		t.Fatalf("client.Connect: %v", err)
	}
	t.Cleanup(func() {
		_ = clientSession.Close()
		_ = serverSession.Close()
	})
	return clientSession
}

func TestCodexToolServerListsExpectedTools(t *testing.T) {
	cs := connectSession(t, NewCodexToolServer(CodexScope{TenantID: "t", DatasetIDs: []string{"kb"}}))
	res, err := cs.ListTools(context.Background(), nil)
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	names := make(map[string]bool, len(res.Tools))
	for _, tool := range res.Tools {
		names[tool.Name] = true
	}
	for _, want := range []string{"search_semantic_chunks", "search_bm25_chunks", "grep_chunks", "list_chunks"} {
		if !names[want] {
			t.Errorf("tool %q not exposed", want)
		}
	}
	if names["search_chunks"] {
		t.Error("search_chunks must not be exposed (mode 8 D9)")
	}
}

func TestCodexToolServerRejectsNoScope(t *testing.T) {
	// A conversation with no bound KB: retrieval must fail explicitly, never search an
	// empty/default scope (mode 8 D8).
	cs := connectSession(t, NewCodexToolServer(CodexScope{TenantID: "t"}))
	res, err := cs.CallTool(context.Background(), &sdk.CallToolParams{
		Name:      "list_chunks",
		Arguments: map[string]any{"doc_id": "d1", "anchor_chunk_ids": []string{"c1"}},
	})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected IsError with no KB scope, got %+v", res)
	}
	text := ""
	for _, c := range res.Content {
		if tc, ok := c.(*sdk.TextContent); ok {
			text += tc.Text
		}
	}
	if !strings.Contains(text, "no knowledge base") {
		t.Fatalf("error text does not explain the missing scope: %q", text)
	}
}

func TestCodexHandlerRejectsUnknownToken(t *testing.T) {
	h := NewCodexHandler(NewCodexTickets(time.Hour))
	req := httptest.NewRequest(http.MethodPost, "/mcp/codex/unknown-token", strings.NewReader("{}"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("unknown token: status = %d, want 401", w.Code)
	}
}

func TestChunkLedgerRecordsAndDedupes(t *testing.T) {
	var nilLedger *ChunkLedger
	nilLedger.Record("x", `chunk_id="a"`) // must not panic
	if nilLedger.IDSet() != nil {
		t.Fatal("nil ledger returns a non-nil set")
	}

	l := NewChunkLedger()
	if l.IDSet() != nil || l.ToolCallCounts() != nil || l.DocIDs() != nil {
		t.Fatal("empty ledger returns non-nil getters")
	}
	l.Record("search_semantic_chunks", `<search_results count="2"><chunk rank="0" chunk_id="c1" doc_id="d1" doc_name="n1"/><chunk rank="1" chunk_id="c2" dataset_id="k"/></search_results>`)
	l.Record("search_semantic_chunks", `<chunk chunk_id="c1" doc_id="d1" doc_name="n1"/>`) // duplicate + repeat call
	l.Record("list_chunks", `<chunks doc_id="d2"><chunk chunk_id="c3" doc_id="d2"/></chunks>`)
	l.Record("", "") // no-op output still counts the call
	set := l.IDSet()
	if len(set) != 3 {
		t.Fatalf("chunk set = %v, want c1,c2,c3", set)
	}
	if l.ToolCallCounts()["search_semantic_chunks"] != 2 || l.ToolCallCounts()["list_chunks"] != 1 {
		t.Fatalf("tool call counts = %v", l.ToolCallCounts())
	}
	docs := l.DocIDs()
	if len(docs) != 3 { // d1, n1 (doc_name), d2
		t.Fatalf("doc ids = %v, want d1,n1,d2", docs)
	}
	// list_chunks is a deep read; the locate tool is shallow.
	if deep := l.DeepChunkIDs(); len(deep) != 1 || deep[0] != "c3" {
		t.Fatalf("deep ids = %v, want [c3]", deep)
	}
	if shallow := l.ShallowChunkIDs(); len(shallow) != 2 {
		t.Fatalf("shallow ids = %v, want c1,c2", shallow)
	}
}
