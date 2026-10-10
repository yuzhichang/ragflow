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
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"

	"ragflow/internal/agentic_rag"
)

// CodexScope is the tenant + dataset scope a mode 8 MCP ticket grants. It is the ONLY
// authority for what the bridge may retrieve: tool arguments cannot widen it.
type CodexScope struct {
	TenantID   string
	DatasetIDs []string
	// Ledger, when non-nil, records the chunk ids this turn's tool calls returned. The
	// chat layer uses it as the citation whitelist (only chunks served this turn may be
	// cited). It is a pointer shared with the caller that minted the ticket.
	Ledger *ChunkLedger
}

// ChunkLedger is a concurrency-safe record of one mode 8 turn's retrieval activity: the
// chunk ids served (the citation whitelist), the documents touched, per-tool call counts,
// and the deep- vs shallow-read split. The retrieval tools return XML with
// `chunk_id="..."` / `doc_id="..."` attributes; Record extracts them.
type ChunkLedger struct {
	mu      sync.Mutex
	ids     map[string]struct{}
	docs    map[string]struct{}
	calls   map[string]int
	deep    map[string]struct{}
	shallow map[string]struct{}
}

var (
	chunkIDAttrRe = regexp.MustCompile(`chunk_id="([^"]+)"`)
	docIDAttrRe   = regexp.MustCompile(`doc_id="([^"]+)"`)
	docNameAttrRe = regexp.MustCompile(`doc_name="([^"]+)"`)
)

// NewChunkLedger returns an empty ledger.
func NewChunkLedger() *ChunkLedger {
	return &ChunkLedger{
		ids:     make(map[string]struct{}),
		docs:    make(map[string]struct{}),
		calls:   make(map[string]int),
		deep:    make(map[string]struct{}),
		shallow: make(map[string]struct{}),
	}
}

// Record accounts one tool call and extracts the chunk/doc ids its output served.
// list_chunks is a full-text deep read; the locate tools (search_*/grep_chunks) are
// shallow reads.
func (l *ChunkLedger) Record(toolName, output string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	l.calls[toolName]++
	l.mu.Unlock()
	if output == "" {
		return
	}
	chunks := chunkIDAttrRe.FindAllStringSubmatch(output, -1)
	docs := docIDAttrRe.FindAllStringSubmatch(output, -1)
	docNames := docNameAttrRe.FindAllStringSubmatch(output, -1)
	deep := toolName == "list_chunks"
	l.mu.Lock()
	for _, m := range chunks {
		if m[1] == "" {
			continue
		}
		l.ids[m[1]] = struct{}{}
		if deep {
			l.deep[m[1]] = struct{}{}
		} else {
			l.shallow[m[1]] = struct{}{}
		}
	}
	// RetrievedDocIDs is documented as doc_id AND doc_name, so both are recorded.
	for _, matches := range [][][]string{docs, docNames} {
		for _, m := range matches {
			if m[1] != "" {
				l.docs[m[1]] = struct{}{}
			}
		}
	}
	l.mu.Unlock()
}

// Reset clears the ledger so it can be reused for a new turn (a session's ledger is
// shared across its threads' tickets; see codex_pipeline).
func (l *ChunkLedger) Reset() {
	if l == nil {
		return
	}
	l.mu.Lock()
	l.ids = make(map[string]struct{})
	l.docs = make(map[string]struct{})
	l.calls = make(map[string]int)
	l.deep = make(map[string]struct{})
	l.shallow = make(map[string]struct{})
	l.mu.Unlock()
}

// IDSet returns the recorded chunk ids as a set (nil when empty).
func (l *ChunkLedger) IDSet() map[string]struct{} {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.ids) == 0 {
		return nil
	}
	out := make(map[string]struct{}, len(l.ids))
	for id := range l.ids {
		out[id] = struct{}{}
	}
	return out
}

// ToolCallCounts returns how many times each tool ran (nil when none).
func (l *ChunkLedger) ToolCallCounts() map[string]int {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.calls) == 0 {
		return nil
	}
	out := make(map[string]int, len(l.calls))
	for name, n := range l.calls {
		out[name] = n
	}
	return out
}

// DocIDs returns the documents touched (sorted; nil when none).
func (l *ChunkLedger) DocIDs() []string {
	return l.sortedKeys(func() map[string]struct{} { return l.docs })
}

// DeepChunkIDs returns the chunk ids read in full via list_chunks (sorted; nil when none).
func (l *ChunkLedger) DeepChunkIDs() []string {
	return l.sortedKeys(func() map[string]struct{} { return l.deep })
}

// ShallowChunkIDs returns the chunk ids seen via the locate tools (sorted; nil when none).
func (l *ChunkLedger) ShallowChunkIDs() []string {
	return l.sortedKeys(func() map[string]struct{} { return l.shallow })
}

func (l *ChunkLedger) sortedKeys(pick func() map[string]struct{}) []string {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	src := pick()
	out := make([]string, 0, len(src))
	for k := range src {
		out = append(out, k)
	}
	l.mu.Unlock()
	if len(out) == 0 {
		return nil
	}
	sort.Strings(out)
	return out
}

// CodexTickets is the per-thread ticket registry: an opaque, unguessable token maps to
// one scope. mode 8 mints a ticket when it creates a Codex thread, bakes the resulting
// /mcp/codex/<token> URL into that thread's config, and revokes it when the thread (or
// its session) ends. A ticket's lifetime is the thread's, not a turn's.
type CodexTickets struct {
	mu    sync.Mutex
	ttl   time.Duration
	items map[string]codexTicket
	now   func() time.Time
}

type codexTicket struct {
	scope   CodexScope
	expires time.Time
}

// NewCodexTickets returns a ticket registry. A non-positive ttl falls back to 24h.
func NewCodexTickets(ttl time.Duration) *CodexTickets {
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}
	return &CodexTickets{ttl: ttl, items: make(map[string]codexTicket), now: time.Now}
}

// Mint registers a scope and returns its opaque token.
func (t *CodexTickets) Mint(scope CodexScope) (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	token := base64.RawURLEncoding.EncodeToString(buf)
	now := t.now()
	t.mu.Lock()
	// Sweep expired tickets: one whose holder never Resolves it again would otherwise
	// stay resident forever.
	for tok, item := range t.items {
		if now.After(item.expires) {
			delete(t.items, tok)
		}
	}
	t.items[token] = codexTicket{scope: scope, expires: now.Add(t.ttl)}
	t.mu.Unlock()
	return token, nil
}

// Resolve returns the scope for a token, or false when the token is unknown or expired.
func (t *CodexTickets) Resolve(token string) (CodexScope, bool) {
	if token == "" {
		return CodexScope{}, false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	item, ok := t.items[token]
	if !ok {
		return CodexScope{}, false
	}
	if t.now().After(item.expires) {
		delete(t.items, token)
		return CodexScope{}, false
	}
	return item.scope, true
}

// Revoke invalidates a token immediately.
func (t *CodexTickets) Revoke(token string) {
	t.mu.Lock()
	delete(t.items, token)
	t.mu.Unlock()
}

// defaultCodexTickets is the process-wide registry: the HTTP bridge resolves against
// it, and the mode 8 engine (internal/codexagent) mints into it, so one code path owns
// every ticket. Tests construct their own registry and pass it explicitly.
var defaultCodexTickets = NewCodexTickets(0)

// DefaultCodexTickets returns the process-wide ticket registry.
func DefaultCodexTickets() *CodexTickets { return defaultCodexTickets }

// codexToolSpecs are the agent_rag retrieval tools the Codex bridge exposes. The names
// and schemas come from the tools themselves (see addCodexTool), so the two never drift.
// search_chunks is intentionally NOT here (see the mode 8 plan, D9).
func codexToolSpecs(scope CodexScope) []einotool.BaseTool {
	return []einotool.BaseTool{
		agentic_rag.NewSearchSemanticChunksTool(scope.TenantID, scope.DatasetIDs),
		agentic_rag.NewSearchBm25ChunksTool(scope.TenantID, scope.DatasetIDs),
		agentic_rag.NewGrepChunksTool(scope.TenantID, scope.DatasetIDs),
		agentic_rag.NewListChunksTool(scope.TenantID, scope.DatasetIDs),
	}
}

// NewCodexToolServer builds an MCP server whose tools are scoped to one ticket. The
// scope is captured at construction (per request); tool arguments can only narrow it
// (resolveDatasetScope inside the tools enforces that), never widen it.
func NewCodexToolServer(scope CodexScope) *sdk.Server {
	server := sdk.NewServer(&sdk.Implementation{Name: "ragflow-codex-mcp", Version: "1.0.0"}, &sdk.ServerOptions{
		Capabilities: &sdk.ServerCapabilities{Tools: &sdk.ToolCapabilities{}},
	})
	for _, t := range codexToolSpecs(scope) {
		addCodexTool(server, scope, t)
	}
	return server
}

// addCodexTool registers one agent_rag tool on the MCP server, reusing the tool's own
// description and JSON schema and forwarding calls to its InvokableRun. The MCP layer
// adds exactly two things the tool does not know about: the no-scope guard, and the
// MCP result envelope.
func addCodexTool(server *sdk.Server, scope CodexScope, t einotool.BaseTool) {
	invokable, ok := t.(einotool.InvokableTool)
	if !ok {
		panic("mcp: codex tool is not invokable")
	}
	info, err := t.Info(context.Background())
	if err != nil {
		panic(fmt.Sprintf("mcp: codex tool Info: %v", err))
	}
	server.AddTool(&sdk.Tool{
		Name:        info.Name,
		Description: info.Desc,
		InputSchema: codexInputSchema(info),
	}, func(ctx context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
		// The conversation binds no knowledge base: fail the retrieval call rather
		// than search an empty/default scope. Codex reads the error and carries on.
		if len(scope.DatasetIDs) == 0 {
			return codexToolError(info.Name + " is unavailable: this conversation has no knowledge base bound"), nil
		}
		args := "{}"
		if req != nil && req.Params != nil && req.Params.Arguments != nil {
			if raw, err := json.Marshal(req.Params.Arguments); err == nil {
				args = string(raw)
			}
		}
		out, err := invokable.InvokableRun(ctx, args)
		if err != nil {
			return codexToolError(fmt.Sprintf("%s: %v", info.Name, err)), nil
		}
		// Record the chunk ids this call returned, so the chat layer can whitelist
		// citations to chunks actually served this turn.
		scope.Ledger.Record(info.Name, out)
		return textResult(out), nil
	})
}

// codexInputSchema marshals the eino tool's own parameter schema for the MCP tool
// declaration, so Codex sees the same contract the in-process agent sees. The schema is
// an object (AddTool requires it); the fallback is only for a tool that declares none.
func codexInputSchema(info *schema.ToolInfo) json.RawMessage {
	if info.ParamsOneOf != nil {
		if js, err := info.ParamsOneOf.ToJSONSchema(); err == nil && js != nil {
			if raw, err := json.Marshal(js); err == nil {
				return raw
			}
		}
	}
	return json.RawMessage(`{"type":"object"}`)
}

func codexToolError(message string) *sdk.CallToolResult {
	result := textResult(message)
	result.IsError = true
	return result
}

// --- HTTP transport ---

// codexScopeKey carries the resolved scope from the auth wrapper into the per-request
// server factory.
type codexScopeKey struct{}

// NewCodexHandler serves the Codex bridge over streamable HTTP at
// POST/GET/DELETE /mcp/codex/{token}. An unknown or expired token is rejected (401)
// before any server is built, so a bad ticket can never reach a tool.
func NewCodexHandler(tickets *CodexTickets) http.Handler {
	streamable := sdk.NewStreamableHTTPHandler(func(r *http.Request) *sdk.Server {
		scope, _ := r.Context().Value(codexScopeKey{}).(CodexScope)
		return NewCodexToolServer(scope)
	}, &sdk.StreamableHTTPOptions{
		Stateless:                    true,
		JSONResponse:                 true,
		MaxRequestBodyBytes:          1 << 20,
		PropagateRequestCancellation: true,
	})
	mux := http.NewServeMux()
	serve := func(w http.ResponseWriter, r *http.Request) {
		scope, ok := tickets.Resolve(codexToken(r))
		if !ok {
			http.Error(w, "invalid or expired ticket", http.StatusUnauthorized)
			return
		}
		streamable.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), codexScopeKey{}, scope)))
	}
	mux.HandleFunc("POST /mcp/codex/{token}", serve)
	mux.HandleFunc("GET /mcp/codex/{token}", serve)
	mux.HandleFunc("DELETE /mcp/codex/{token}", serve)
	return mux
}

// codexToken extracts the ticket from the request path. It prefers PathValue (set when
// this handler's own ServeMux routed the request) and falls back to the path suffix so
// the handler also works when mounted behind another router (e.g. gin.WrapH).
func codexToken(r *http.Request) string {
	if v := r.PathValue("token"); v != "" {
		return v
	}
	const prefix = "/mcp/codex/"
	if p := r.URL.Path; strings.HasPrefix(p, prefix) {
		return strings.TrimPrefix(p, prefix)
	}
	return ""
}
