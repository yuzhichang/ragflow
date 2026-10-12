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
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"

	"ragflow/internal/agentic_rag"
	"ragflow/internal/codexagent"
	"ragflow/internal/common"
	"ragflow/internal/dao"
	"ragflow/internal/entity"
	"ragflow/internal/mcp"
	serverconfig "ragflow/internal/server/config"
)

// versionSegmentRe matches an API version path segment such as "/v1", "/v1beta" or
// "/v2/". Used to decide whether a provider base URL already names a version.
var versionSegmentRe = regexp.MustCompile(`/v[0-9]+[a-z0-9]*(/|$)`)

// normalizeResponsesBaseURL makes a provider base URL usable as the Codex Responses
// endpoint. The Codex client appends "/responses", so a base that carries no version
// segment — e.g. MiniMax's "https://api.minimaxi.com" (its chat URL is built from a
// separate "v1/text/chatcompletion_v2" suffix) — would call ".../responses" and 404.
// Append "/v1" unless the path already names a version (OpenAI's ".../v1" is left
// alone). Query/fragment are preserved and inspected apart from the path.
func normalizeResponsesBaseURL(raw string) string {
	base := strings.TrimSpace(raw)
	if base == "" {
		return base
	}
	suffix := ""
	if i := strings.IndexAny(base, "?#"); i >= 0 {
		suffix = base[i:]
		base = base[:i]
	}
	base = strings.TrimRight(base, "/")
	if versionSegmentRe.MatchString(base) {
		return base + suffix
	}
	return base + "/v1" + suffix
}

// codexConfig holds the mode 8 Codex configuration. It is installed once at server
// boot (see cmd/ragflow_server.go) and read on the chat path. A mutex makes the swap
// and the reads race-free.
var codexConfig struct {
	mu  sync.RWMutex
	cfg *serverconfig.CodexConfig
}

// SetCodexConfig installs the mode 8 configuration. A nil argument disables mode 8.
func SetCodexConfig(cfg *serverconfig.CodexConfig) {
	codexConfig.mu.Lock()
	codexConfig.cfg = cfg
	codexConfig.mu.Unlock()
}

// getCodexConfig returns the installed mode 8 configuration, or nil when none is set.
func getCodexConfig() *serverconfig.CodexConfig {
	codexConfig.mu.RLock()
	defer codexConfig.mu.RUnlock()
	return codexConfig.cfg
}

// codexConfigured reports whether mode 8 can run: a shared Codex app-server endpoint
// and the public MCP base URL are configured. Without them, AsyncChat drops an
// engineCodex turn to the regular pipeline rather than dispatching to an engine that
// cannot connect.
func codexConfigured() bool {
	cfg := getCodexConfig()
	return cfg != nil && cfg.Endpoint != "" && cfg.MCPPublicBase != ""
}

// codexRunner is the process-wide runner (one Codex connection, reused across turns and
// sessions), built lazily on first use.
var (
	codexRunnerMu sync.Mutex
	codexRunner   *codexagent.Runner
)

// codexBinder adapts the shared (KV-backed) mode 8 handle/binding store to the
// codexagent.TurnBinder seam — so codexagent depends only on the interface, not the bridge
// store — and owns the turn's keep-alive: Bind starts a goroutine that renews the handle
// TTL and the binding lease every codexLeaseRenewInterval, and Unbind stops it. Tying the
// heartbeat to the Bind/Unbind lifecycle means the lease is refreshed exactly while the
// binding is held, and a crashed turn's lease simply expires.
type codexBinder struct {
	mu     sync.Mutex
	leases map[string]func()
}

const codexLeaseRenewInterval = time.Minute

func newCodexBinder() *codexBinder {
	return &codexBinder{leases: map[string]func(){}}
}

func (b *codexBinder) Bind(ctx context.Context, handle string, binding any) error {
	scope, ok := binding.(mcp.CodexScope)
	if !ok {
		return fmt.Errorf("codex: invalid binding type %T", binding)
	}
	if err := mcp.DefaultCodexTickets().Bind(ctx, handle, scope); err != nil {
		return err
	}
	b.startLease(handle)
	return nil
}

func (b *codexBinder) Unbind(ctx context.Context, handle string) {
	b.stopLease(handle)
	mcp.DefaultCodexTickets().Unbind(ctx, handle)
}

// startLease runs the keep-alive heartbeat for handle until stopLease(handle) is called.
func (b *codexBinder) startLease(handle string) {
	stop := make(chan struct{})
	b.mu.Lock()
	if prev := b.leases[handle]; prev != nil {
		prev() // defensive: Bind is exclusive, so this should not happen
	}
	b.leases[handle] = func() { close(stop) }
	b.mu.Unlock()

	go func() {
		ticker := time.NewTicker(codexLeaseRenewInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				mcp.DefaultCodexTickets().Renew(context.Background(), handle)
			}
		}
	}()
}

func (b *codexBinder) stopLease(handle string) {
	b.mu.Lock()
	stop := b.leases[handle]
	delete(b.leases, handle)
	b.mu.Unlock()
	if stop != nil {
		stop()
	}
}

// getCodexRunner builds the process-wide runner on first use. The initial connection has
// its own bounded deadline — a stalled handshake must not block every mode 8 request
// forever — but the client built on success lives for the process: it is deliberately NOT
// bound to a request context, and the transport reconnects on its own. A failure is not
// cached: the next turn retries.
func getCodexRunner() (*codexagent.Runner, error) {
	codexRunnerMu.Lock()
	defer codexRunnerMu.Unlock()
	if codexRunner != nil {
		return codexRunner, nil
	}
	cfg := getCodexConfig()
	if cfg == nil {
		return nil, fmt.Errorf("codex (mode 8) is not configured")
	}
	dialCtx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	client, err := codexagent.Dial(dialCtx, codexagent.ClientConfig{
		Endpoint:    cfg.Endpoint,
		BearerToken: cfg.BearerToken,
	})
	if err != nil {
		return nil, fmt.Errorf("codex dial: %w", err)
	}
	codexRunner = codexagent.NewRunner(client, dao.NewCodexThreadStore(dao.DB), newCodexBinder())
	return codexRunner, nil
}

// InstallCodexWebSearchResolver wires the MCP bridge's web-search seam. The bridge must
// not import this layer, so the API process installs a resolver that reads the turn's
// conversation and builds its provider callback on demand. Call once at boot.
//
// The callback is rebuilt per request from the DB and never persisted, so a turn bound
// on one replica still resolves its provider when the MCP request lands on another — and
// the provider's credentials never enter the shared KV.
func InstallCodexWebSearchResolver() {
	svc := NewChatPipelineService()
	mcp.SetCodexWebSearchResolver(func(ctx context.Context, scope mcp.CodexScope) agentic_rag.WebSearchFunc {
		if scope.ChatID == "" || dao.DB == nil {
			return nil
		}
		chat, err := dao.NewChatDAO().GetByID(ctx, dao.DB, scope.ChatID)
		if err != nil || chat == nil {
			common.WarnCtx(ctx, "codex: cannot resolve chat for web search",
				zap.String("chat_id", scope.ChatID), zap.Error(err))
			return nil
		}
		// nil when the conversation has no web-search provider: the bridge then
		// simply does not advertise the tool.
		return svc.webSearchFunc(chat.PromptConfig)
	})
}

// codexResponsesCapable reports whether the dialog's model provider speaks the
// Responses wire API Codex requires. A provider outside the configured allowlist makes
// mode 8 fall back to the regular pipeline (see the plan, decision U2).
func (s *ChatPipelineService) codexResponsesCapable(ctx context.Context, userID string, chat *entity.Chat) bool {
	cfg := getCodexConfig()
	if cfg == nil || len(cfg.ResponsesProviders) == 0 {
		return true // no allowlist configured => no restriction
	}
	target, err := s.resolveCodexModelTarget(ctx, userID, chat)
	if err != nil || target == nil {
		return false
	}
	name := strings.ToLower(target.Driver.Name())
	allowed := false
	for _, p := range cfg.ResponsesProviders {
		if p == name {
			allowed = true
			break
		}
	}
	if !allowed {
		return false
	}
	// The thread declares the instance's base_url as the provider endpoint; if it is
	// empty, Codex has no Responses endpoint and the turn cannot run. Refuse now with a
	// clear reason instead of failing deep in the turn (see the plan, §13 / P2-13).
	if strings.TrimSpace(target.BaseURL) == "" {
		common.WarnCtx(ctx, "codex (mode 8) requires the dialog model instance to set base_url (a Responses-capable endpoint); using the regular pipeline",
			zap.String("chat_id", chat.ID), zap.String("provider", target.Driver.Name()))
		return false
	}
	return true
}

// resolveCodexModelTarget resolves the dialog's chat model (or the tenant default) to its
// provider details. Unlike the general model path (which only needs a constructed model),
// mode 8 must hand Codex the provider driver, endpoint and token, so it uses the
// ModelFactory's credential-exposing accessor.
func (s *ChatPipelineService) resolveCodexModelTarget(ctx context.Context, userID string, chat *entity.Chat) (*ResolvedModelTarget, error) {
	access := ModelAccess{UserID: userID, TenantID: chat.TenantID}
	var (
		target *ResolvedModelTarget
		err    error
	)
	if chat.LLMID == "" {
		target, err = s.ModelFactory.ResolveDefaultChatTarget(ctx, access)
	} else {
		target, err = s.ModelFactory.ResolveChatTarget(ctx, access, chat.LLMID)
	}
	if err != nil || target == nil {
		return target, err
	}
	// Codex appends "/responses" to this base; normalize the version segment first.
	target.BaseURL = normalizeResponsesBaseURL(target.BaseURL)
	return target, nil
}

// codexAgent runs one mode 8 (Codex) turn: it brokers the session's Codex thread (via
// internal/codexagent) and lets Codex call the agent_rag retrieval tools through this
// process's MCP bridge.
//
// It mirrors the channel contract of iterativeSynthesis/agenticRag: reasoning deltas
// wrapped in <think> boundaries, then answer deltas, then a Final result carrying the
// complete answer and the citation payload.
func (s *ChatPipelineService) codexAgent(
	ctx context.Context,
	userID string,
	chat *entity.Chat,
	messages []map[string]interface{},
	stream bool,
	kwargs map[string]interface{},
	useWebSearch bool,
	quote bool,
) (<-chan AsyncChatResult, error) {
	cfg := getCodexConfig()
	if cfg == nil || cfg.Endpoint == "" || cfg.MCPPublicBase == "" {
		return nil, fmt.Errorf("codex (mode 8) is not configured")
	}
	target, err := s.resolveCodexModelTarget(ctx, userID, chat)
	if err != nil || target == nil {
		return nil, fmt.Errorf("codex: resolve dialog model: %w", err)
	}
	if strings.TrimSpace(target.BaseURL) == "" {
		return nil, fmt.Errorf("codex: the dialog model instance must set base_url to a Responses-capable endpoint")
	}

	sessionID := common.SessionIDFromContext(ctx)
	// A non-persistent request (store_history_messages=false) supplies its full history
	// every turn and stores nothing, so it must not own a persistent thread: force the
	// one-shot path (see the plan, §5.2 / P1#7).
	if store, storeErr := ResolveStoreHistoryMessages(kwargs); storeErr == nil && !store {
		sessionID = ""
	}
	datasetIDs := chatKBIDs(chat)

	runner, err := getCodexRunner()
	if err != nil {
		return nil, err
	}

	out := make(chan AsyncChatResult, 16)
	go func() {
		defer close(out)
		started := time.Now()

		// A stable per-session handle is baked into the Codex thread's MCP URL. It carries
		// no authority: the turn's scope is bound to it for the duration of the turn (by
		// the runner, under the session lock) and removed when the turn ends, so the URL
		// authorizes nothing between turns.
		handle, handleErr := mcp.DefaultCodexTickets().HandleForSession(ctx, chat.TenantID, sessionID)
		if handleErr != nil {
			out <- AsyncChatResult{Answer: fmt.Sprintf("**ERROR**: codex: acquire MCP handle: %v", handleErr), Final: true}
			return
		}
		mcpURL := cfg.MCPPublicBase + "/mcp/codex/" + handle
		// A one-shot turn (no session) owns a throwaway handle, revoked at the end. A
		// session handle outlives the turn — later turns reuse it and the session delete
		// revokes it.
		if sessionID == "" {
			defer mcp.DefaultCodexTickets().RevokeHandle(context.Background(), handle)
		}

		var (
			thinking     bool
			contentDelta strings.Builder
		)
		emitReasoning := func(text string) {
			if text == "" {
				return
			}
			if !thinking {
				out <- AsyncChatResult{Final: false, StartToThink: true}
				thinking = true
			}
			out <- AsyncChatResult{Final: false, Reasoning: text}
		}
		emitContent := func(text string) {
			if text == "" {
				return
			}
			if thinking {
				out <- AsyncChatResult{Final: false, EndToThink: true}
				thinking = false
			}
			contentDelta.WriteString(text)
			out <- AsyncChatResult{Final: false, Answer: text}
		}

		req := codexagent.TurnRequest{
			TenantID:   chat.TenantID,
			SessionID:  sessionID,
			DatasetIDs: datasetIDs,
			WebSearch:  useWebSearch,
			Model:      target.ModelName,
			ProviderID: target.Driver.Name(),
			MCPURL:     mcpURL,
			Ticket:     handle,
			// The binding carries the conversation id and this turn's web-search
			// decision; the bridge resolves the provider from them at request time,
			// so the provider's credentials stay in the DB and never reach the
			// KV-serialized binding (see D11).
			Binding: mcp.CodexScope{
				TenantID:   chat.TenantID,
				DatasetIDs: datasetIDs,
				UserID:     userID,
				ChatID:     chat.ID,
				WebSearch:  useWebSearch,
			},
			WorkDir:      cfg.WorkDir,
			Instructions: codexCitationContract(quote),
			TurnTimeout:  cfg.TurnTimeout,
			Messages:     messages,
			OnContent:    emitContent,
			OnReasoning:  emitReasoning,
			// A thread is only resumed when its stored handle is still alive; otherwise
			// its baked-in MCP URL would 401 (a restart lost the handle) and the
			// retrieval path would silently break.
			HandleAlive: func(tok string) bool {
				alive, _ := mcp.DefaultCodexTickets().HandleAlive(ctx, tok)
				return alive
			},
		}
		// The codex thread's per-thread provider declares exactly the instance's
		// base_url. mode 8 requires the dialog's model instance to set a base_url that
		// speaks the Responses API (Codex's wire_api); there is deliberately no fallback
		// to the provider factory's default URL (see the plan, §13 / P2-13).
		req.ProviderBaseURL = target.BaseURL
		req.ProviderToken = target.APIKey

		outcome, runErr := runner.RunTurn(ctx, req)
		if runErr != nil {
			if thinking {
				out <- AsyncChatResult{Final: false, EndToThink: true}
				thinking = false
			}
			// The binding is the per-session turn lock: a losing turn reports the session
			// as busy rather than clobbering the running one.
			msg := runErr.Error()
			if errors.Is(runErr, mcp.ErrCodexTurnBusy) {
				msg = fmt.Sprintf("session %s is busy with another turn; try again when it finishes", sessionID)
			}
			out <- AsyncChatResult{Answer: fmt.Sprintf("**ERROR**: %s", msg), Final: true}
			return
		}
		// The runner has already unbound this turn's scope; a session handle lives on for
		// later turns, and a one-shot handle was revoked by the deferred cleanup above.

		if thinking {
			out <- AsyncChatResult{Final: false, EndToThink: true}
			thinking = false
		}
		// Prefer the runner's authoritative answer (the turn's final agent message); the
		// delta accumulation can include process narration and can miss text after a
		// reconnect.
		final := outcome.Content
		if final == "" {
			final = contentDelta.String()
		}
		reference := map[string]interface{}{}
		if quote {
			// Citations come from the model's own `chunk_id:` markers, each verified by
			// the backend against the turn's KB scope (fetchChunksByIDs): the id must
			// exist AND belong to this conversation's datasets to be marked. A nil
			// whitelist is the scope-only path — there is no served-chunk ledger in mode
			// 8, so a chunk the model names but never retrieved still resolves if it is
			// in scope.
			reference, final = s.buildAgenticReferenceWithWhitelist(ctx, chat.TenantID, datasetIDs, final, nil)
		}
		out <- AsyncChatResult{
			Answer:         final,
			Reference:      reference,
			Final:          true,
			ElapsedSeconds: time.Since(started).Seconds(),
		}
	}()
	return out, nil
}

// codexCitationContract is the citation instruction folded into the thread's base
// instructions. It asks Codex to name the chunk ids it used, in a form the backend can
// parse (see the plan, D5/§4.7) — the backend, not the model, assigns [ID:N].
func codexCitationContract(quote bool) string {
	if !quote {
		return "Answer the user. You have retrieval tools; use them when the knowledge base is relevant."
	}
	return "Answer the user. You have retrieval tools; call them when the knowledge base is relevant. " +
		"When you use a retrieved chunk, cite it by writing `chunk_id: <the exact chunk_id from the tool output>` " +
		"on the line where you use it. Copy the chunk_id verbatim; do not invent ids and do not write [ID:N] markers."
}

// chatKBIDs extracts the dialog's knowledge-base ids as strings.
func chatKBIDs(chat *entity.Chat) []string {
	ids := make([]string, 0, len(chat.KBIDs))
	for _, raw := range chat.KBIDs {
		if id, ok := raw.(string); ok && id != "" {
			ids = append(ids, id)
		}
	}
	return ids
}

// cleanupCodexThread drops a deleted session's mode 8 state: the persisted thread mapping
// and the session's MCP handle (so the thread can no longer reach this process — see the
// plan, §5.2). The remote Codex thread is left to the server's own lifecycle. Per-session
// mutual exclusion is the KV binding, not a lock held here, so a delete can run alongside
// an in-flight turn — that turn's mapping write is guarded on the handle still being alive.
func cleanupCodexThread(ctx context.Context, sessionID string) {
	if sessionID == "" {
		return
	}
	if dao.DB == nil {
		return
	}
	handle, err := dao.NewCodexThreadStore(dao.DB).DeleteBySessionID(ctx, sessionID)
	if err != nil {
		common.WarnCtx(ctx, "codex: failed to clear thread mapping",
			zap.String("session_id", sessionID), zap.Error(err))
		return
	}
	if handle != "" {
		mcp.DefaultCodexTickets().RevokeHandle(ctx, handle)
	}
}
