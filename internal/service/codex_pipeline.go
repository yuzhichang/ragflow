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

	// The dialog model's real context window and output cap, from the provider catalog
	// (conf/all_models.json). Codex resolves any model outside its own manifest to
	// fallback metadata with no context window — and then neither knows the limit nor
	// auto-compacts (context_window.rs), so a full-history rebuild could overflow the
	// provider's real window and fail there. Handing Codex the true window lets it
	// compact first; zero means "unknown, leave Codex's own value".
	contextWindow := dao.ResolveModelContentLength(ctx, nil, chat.TenantID, chat.LLMID, target.Driver.Name(), target.ModelName)
	maxOutput := dao.ResolveModelMaxOutput(ctx, nil, chat.TenantID, chat.LLMID, target.Driver.Name(), target.ModelName)

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
		// The last byte written to the reasoning stream, so a narrated tool step can be
		// separated from the text around it by EXACTLY one newline: the model's reasoning
		// runs straight into the step otherwise ("...characteristics.search_bm25_chunks
		// {...}"), which is unreadable. A fresh think block starts at a line boundary.
		var reasoningTail byte
		emitReasoning := func(text string) {
			if text == "" {
				return
			}
			if !thinking {
				out <- AsyncChatResult{Final: false, StartToThink: true}
				thinking = true
				reasoningTail = 0
			}
			out <- AsyncChatResult{Final: false, Reasoning: text}
			reasoningTail = text[len(text)-1]
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
		// A retrieval step rides the thinking stream: mode 8 has no client-side tool loop,
		// so without this a user watching the run sees reasoning but never which tool the
		// model reached for or with what arguments. The in-process engines narrate steps
		// into their thinking text for the same reason; the structured ThinkEvent travels
		// alongside for a client that renders steps rather than paragraphs.
		emitTool := func(rt codexagent.RoundTrip) {
			// A local shell execution is narrated like a tool step, but under its own
			// stage and with exit-status wording: a command's outcome is a process exit,
			// not a tool result.
			stage, line := "MCP", toolNarration(rt)
			switch rt.Kind {
			case codexagent.RoundTripKindCommand:
				stage, line = "shell", commandNarration(rt)
			case codexagent.RoundTripKindAgent:
				stage, line = "agent", agentNarration(rt)
			case codexagent.RoundTripKindBuiltin:
				stage, line = "builtin", builtinNarration(rt)
			}
			if line == "" {
				return
			}
			if !thinking {
				out <- AsyncChatResult{Final: false, StartToThink: true}
				thinking = true
				reasoningTail = 0
			}
			out <- AsyncChatResult{
				Final:     false,
				Reasoning: narrationChunk(reasoningTail, line),
				ThinkEvent: &ThinkEvent{
					Kind:       toolThinkKind(rt.Status),
					Stage:      stage,
					Tool:       rt.Tool,
					Args:       rt.Args,
					Status:     rt.Status,
					DurationMS: rt.DurationMS,
					Summary:    line,
				},
			}
			reasoningTail = '\n'
		}
		// A context compaction is narrated like a tool step. It is a real event in the
		// turn — Codex summarized the model's history to fit the window — and the think
		// panel is the only place a user watches the run, so it belongs there. It does
		// not change how the thread is rebuilt: mode 8 never trims a rebuilt thread's
		// history, so RAGFlow's stored conversation keeps every turn.
		emitCompaction := func() {
			line := "context compacted: Codex summarized the earlier conversation to fit the model's context window"
			if !thinking {
				out <- AsyncChatResult{Final: false, StartToThink: true}
				thinking = true
				reasoningTail = 0
			}
			out <- AsyncChatResult{
				Final:     false,
				Reasoning: narrationChunk(reasoningTail, line),
				ThinkEvent: &ThinkEvent{
					Kind:    "context_compaction",
					Stage:   "context",
					Summary: line,
				},
			}
			reasoningTail = '\n'
		}

		req := codexagent.TurnRequest{
			TenantID:   chat.TenantID,
			SessionID:  sessionID,
			DatasetIDs: datasetIDs,
			WebSearch:  useWebSearch,
			Model:      target.ModelName,
			ProviderID: target.Driver.Name(),
			// The dialog model's window / output cap and the fixed reasoning effort are
			// baked into the thread config so Codex knows the real limit (see above).
			ModelContextWindow: contextWindow,
			ModelMaxOutput:     maxOutput,
			ReasoningEffort:    cfg.ReasoningEffort,
			MCPURL:             mcpURL,
			Ticket:             handle,
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
			OnTool:       emitTool,
			OnCompaction: emitCompaction,
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
			// Citations come from the model's provenance tuples: a corpus chunk_id is
			// verified against this conversation's datasets (fetchChunksByIDs); a web url
			// is the model's own claim, since the turn records nothing about what a search
			// served. There is deliberately no served-chunk ledger, so a corpus chunk the
			// model names but never retrieved still resolves if it is in scope (D12).
			reference, final = s.buildCodexReference(ctx, chat.TenantID, datasetIDs, final)
		}
		// Codex reports one token-usage notification per model call; the runner sums them
		// into the same shape the in-process engines produce, so a mode-8 benchmark row
		// carries prompt/completion/total/llm_calls exactly like mode 5's.
		var usage *TurnUsage
		if outcome.Usage != nil {
			usage = &TurnUsage{
				PromptTokens:     outcome.Usage.PromptTokens,
				CompletionTokens: outcome.Usage.CompletionTokens,
				TotalTokens:      outcome.Usage.TotalTokens,
				LLMCalls:         outcome.Usage.LLMCalls,
			}
		}
		out <- AsyncChatResult{
			Answer:              final,
			Reference:           reference,
			Final:               true,
			Usage:               usage,
			ToolCallCounts:      outcome.ToolCallCounts,
			ToolCallErrors:      outcome.ToolCallErrors,
			RetrievedDocIDs:     outcome.RetrievedDocIDs,
			ServedDocIDs:        outcome.ServedDocIDs,
			DeepReadChunks:      outcome.DeepReadChunks,
			ShallowReadChunks:   outcome.ShallowReadChunks,
			DeepReadChunkIDs:    outcome.DeepReadChunkIDs,
			ShallowReadChunkIDs: outcome.ShallowReadChunkIDs,
			Compactions:         outcome.Compactions,
			ElapsedSeconds:      time.Since(started).Seconds(),
		}
	}()
	return out, nil
}

// codexCitationContract is the thread's citation contract (see the plan, D5/§4.7). It is
// delivered as DEVELOPER instructions — additive to Codex's own prompt — and covers the
// provenance format (the model ENDS each sourced line with that passage's tuple, and the
// backend, not the model, assigns [ID:N]) plus the tool split the model would otherwise get
// wrong.
//
// It is DELIBERATELY generic about web search. The contract is baked into a thread at
// create and thread/resume does not reload it, while web search is a per-turn decision;
// encoding the decision here would make the prompt and the tool list two copies of one fact
// that must be kept in agreement. Instead the MCP tool list is the single source of truth —
// the bridge advertises its web search tool only when the conversation enables it — and the
// prompt points at that list ("use exactly the retrieval tools it contains"). The prompt
// therefore never goes stale: it is the same whether web search is on or off.
//
// The shell is a LOCAL COMPUTATION channel, not a retrieval one. mode 8 runs with
// ApprovalNever + a read-only sandbox whose network_access is false, so a short script
// (e.g. `python3 -c ...`) runs for arithmetic but cannot reach the network — enabling Codex
// to compute exactly without giving the model a way around the conversation's web-search
// setting. The prompt therefore ALLOWS the shell for calculation and forbids it for the
// web: left unsaid, the model either refuses to compute, or burns the turn on `curl`.
func codexCitationContract(quote bool) string {
	var b strings.Builder
	b.WriteString("Answer the user in prose. Your RETRIEVAL tools are the MCP tools provided for this conversation: " +
		"the knowledge-base retrieval tools, and — when this conversation enables web search — a web search tool. " +
		"The tool list you are given is authoritative: use exactly the retrieval tools it contains, and never assume " +
		"a retrieval capability it does not list. ")
	b.WriteString("You also have a shell for LOCAL computation only: when a question needs arithmetic, counting or a " +
		"short script, run it (e.g. python3 or node) instead of guessing. The shell has NO network access, so never " +
		"use it to fetch or search the web (curl and the like) — such attempts are blocked by the sandbox and only " +
		"waste the turn. All retrieval therefore comes from the tools above: if the web search tool is available and " +
		"the knowledge base does not cover the question, use that tool; otherwise answer from the retrieved passages " +
		"and say plainly what they do not cover. ")
	if !quote {
		return b.String()
	}
	b.WriteString("Every line that states a fact backed by a retrieved passage must END with that passage's provenance, in exactly this form:\n" +
		"(doc: <doc_name>, doc_id: <doc_id>, chunk_id: <chunk_id>, snippet: \"<short verbatim excerpt>\")\n" +
		"A passage returned by the web search tool uses this form instead:\n" +
		"(source: web, url: <url>, title: <title>, snippet: \"<short verbatim excerpt>\")\n")
	b.WriteString("Copy every value verbatim from that passage's tool output; never invent ids, urls or titles. " +
		"Do not write [ID:N] markers - the system assigns them.")
	return b.String()
}

// chatKBIDs extracts the dialog's knowledge-base ids as strings.
// narrationChunk frames one narrated tool line for the thinking stream: exactly one
// newline before it (only when the stream is mid-line — a fresh block already starts on a
// line boundary) and exactly one after, so the line never runs into the reasoning text
// around it.
func narrationChunk(tail byte, line string) string {
	if tail != 0 && tail != '\n' {
		return "\n" + line + "\n"
	}
	return line + "\n"
}

// toolNarration is the thinking-stream line for one MCP tool step: the call with its
// arguments, then the outcome. emitTool frames it with exactly one newline on each side.
func toolNarration(rt codexagent.RoundTrip) string {
	if rt.Tool == "" {
		return ""
	}
	switch rt.Status {
	case codexagent.ToolStatusCompleted:
		return fmt.Sprintf("%s returned in %dms", rt.Tool, rt.DurationMS)
	case codexagent.ToolStatusFailed:
		return fmt.Sprintf("%s failed in %dms", rt.Tool, rt.DurationMS)
	default:
		if rt.Args == "" {
			return rt.Tool
		}
		return rt.Tool + " " + rt.Args
	}
}

// commandNarration renders one local shell execution for the thinking stream: the command
// line, then its exit status and duration. It mirrors toolNarration, whose "returned in"
// wording is about a tool result rather than a process exit.
func commandNarration(rt codexagent.RoundTrip) string {
	if rt.Args == "" {
		return rt.Tool
	}
	switch rt.Status {
	case codexagent.ToolStatusCompleted:
		return fmt.Sprintf("%s: %s (exit 0, %dms)", rt.Tool, rt.Args, rt.DurationMS)
	case codexagent.ToolStatusFailed:
		if rt.ExitCode != nil {
			return fmt.Sprintf("%s: %s (exit %d, %dms)", rt.Tool, rt.Args, *rt.ExitCode, rt.DurationMS)
		}
		return fmt.Sprintf("%s: %s (failed, %dms)", rt.Tool, rt.Args, rt.DurationMS)
	default:
		return fmt.Sprintf("%s: %s", rt.Tool, rt.Args)
	}
}

// agentNarration renders one sub-agent collaboration step (features.multi_agent_v2) for the
// thinking stream: the collaboration tool and the delegated task (or the targeted agents).
// The tool arrives namespaced ("agent:spawnAgent"); the line reads better un-namespaced.
func agentNarration(rt codexagent.RoundTrip) string {
	name := strings.TrimPrefix(rt.Tool, "agent:")
	switch {
	case rt.Args == "":
		return "agent: " + name
	case rt.Status == codexagent.ToolStatusRunning:
		return fmt.Sprintf("agent: %s %s", name, rt.Args)
	default:
		return fmt.Sprintf("agent: %s %s (%s)", name, rt.Args, rt.Status)
	}
}

// builtinNarration renders any other Codex built-in action (plan, fileChange,
// dynamicToolCall, webSearch, imageView, sleep, review mode, image generation) as one brief
// line, so the thinking stream shows every tool the model used, not only MCP ones.
func builtinNarration(rt codexagent.RoundTrip) string {
	name := strings.TrimPrefix(rt.Tool, "builtin:")
	if rt.Args == "" {
		return "builtin: " + name
	}
	return fmt.Sprintf("builtin: %s %s", name, rt.Args)
}

// toolThinkKind maps a round trip's status onto the shared ThinkEvent kinds
// (runtime.ThinkKindToolCall / ThinkKindToolResult).
func toolThinkKind(status string) string {
	if status == codexagent.ToolStatusRunning {
		return "tool_call"
	}
	return "tool_result"
}

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
