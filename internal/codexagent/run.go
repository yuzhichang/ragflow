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
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	codexgo "github.com/zealbase/codex-app-server-go"
)

// ThreadRecord is the persisted mapping from a RAGFlow session to a Codex thread.
type ThreadRecord struct {
	TenantID           string
	SessionID          string
	ThreadID           string
	ScopeFingerprint   string
	HistoryFingerprint string
	// Ticket is the stable per-session MCP handle baked into this thread's config, kept
	// so a reuse can check it is still alive and a session delete can revoke it.
	Ticket string
}

// ThreadStore persists session -> thread mappings. A nil store is allowed for
// ephemeral-only use (a session id with no store always runs a one-shot thread).
type ThreadStore interface {
	Get(ctx context.Context, tenantID, sessionID string) (ThreadRecord, bool, error)
	Put(ctx context.Context, rec ThreadRecord) error
	Delete(ctx context.Context, tenantID, sessionID string) error
}

// TurnRequest is one mode 8 turn.
type TurnRequest struct {
	TenantID  string
	SessionID string // empty => ephemeral one-shot thread (no persistence)
	// DatasetIDs is the conversation's current KB scope; it selects the ticket and
	// is folded into the scope fingerprint.
	DatasetIDs []string
	// WebSearch is this turn's web-search decision (the chat has a provider AND the
	// request enabled internet). It is part of the thread's reuse key: Codex caches
	// the MCP tool catalog for a thread's life, so flipping web search must rebuild
	// the thread — otherwise the bridge's new tool set is never listed (see the mode
	// 8 plan, §5.7③).
	WebSearch bool

	// Model / provider: the resolved dialog model (see the plan, D4). Model is the
	// real model name, not the composite llm_id.
	Model           string
	ProviderID      string
	ProviderBaseURL string
	ProviderToken   string

	// MCPURL is the full /mcp/codex/<handle> URL this process exposes to Codex.
	MCPURL string
	// Ticket is the stable session handle inside MCPURL. It is persisted per thread so a
	// reuse can check it is still alive and a session delete can revoke it.
	Ticket string
	// Binding is the per-turn authorization scope installed on Ticket for the duration of
	// this turn, and removed when the turn ends. Its concrete type is owned by the
	// injected TurnBinder; opaque here (nil skips binding).
	Binding any
	// WorkDir is the empty directory Codex runs in. It must exist on the machine the
	// shared Codex runs on (see codex.work_dir); empty means a fresh local temp dir.
	WorkDir string
	// Instructions is the citation contract (see the plan, D5).
	Instructions string

	TurnTimeout time.Duration
	// Messages is the full history with the current user message LAST.
	Messages []map[string]interface{}

	// OnContent receives answer deltas; OnReasoning receives reasoning deltas.
	OnContent   func(string)
	OnReasoning func(string)
	// HandleAlive reports whether a stored handle is still live. When it returns false a
	// reuse is refused and the thread is rebuilt, so a thread whose MCP URL carries a
	// dead handle (a restart lost it, or it expired between turns) is never resumed. nil
	// means "assume alive".
	HandleAlive func(token string) bool
}

// TurnBinder installs and removes the per-turn authorization a turn's MCP handle needs.
// Because the binding is a single atomically-acquired KV value, it doubles as the
// per-session turn lock: Bind returns an error (e.g. another turn already holds it) when
// the turn must not proceed. The implementation also owns the binding's keep-alive
// heartbeat — started on a successful Bind and stopped on Unbind — so codexagent stays free
// of a dependency on the store. Binding is opaque: the caller owns the scope type.
type TurnBinder interface {
	// Bind atomically acquires the turn's binding; a non-nil error (e.g. another turn is
	// running) aborts the turn.
	Bind(ctx context.Context, handle string, binding any) error
	// Unbind releases the binding and stops its keep-alive.
	Unbind(ctx context.Context, handle string)
}

// TurnOutcome reports the terminal state of one turn.
type TurnOutcome struct {
	ThreadID  string
	Content   string
	Reasoning string
	// Rebuilt is true when the turn ran on a freshly created thread (first turn, or
	// a rebuild after scope/history drift).
	Rebuilt bool
}

// Runner executes mode 8 turns against a shared Codex client.
//
// Per-session mutual exclusion is NOT a process-local lock here: it comes from the turn's
// MCP binding, acquired atomically in KV (TurnBinder.Bind) and released when the turn ends.
// That serializes same-session turns across processes and replicas, and a second concurrent
// turn fails fast with mcp.ErrCodexTurnBusy instead of queueing.
type Runner struct {
	client *codexgo.Client
	store  ThreadStore
	binder TurnBinder
}

// NewRunner returns a runner. store may be nil (ephemeral-only); binder may be nil (no
// per-turn MCP binding).
func NewRunner(client *codexgo.Client, store ThreadStore, binder TurnBinder) *Runner {
	return &Runner{client: client, store: store, binder: binder}
}

// bindTurn acquires the turn's MCP binding. A non-nil error (e.g. another turn is already
// running for the session) aborts the turn: the binding is the per-session turn lock, so
// proceeding without it would either run unauthorized or clobber the other turn's scope.
func (r *Runner) bindTurn(ctx context.Context, req TurnRequest) error {
	if r.binder == nil || req.Ticket == "" || req.Binding == nil {
		return nil
	}
	if err := r.binder.Bind(ctx, req.Ticket, req.Binding); err != nil {
		return fmt.Errorf("codexagent: bind MCP handle %q: %w", req.Ticket, err)
	}
	return nil
}

// unbindTurn removes the turn's MCP binding so the handle stops being authorized once the
// turn is over. It always uses a detached, bounded context: the turn's ctx may already be
// canceled, and a failed unbind would leave the handle authorized until its TTL.
func (r *Runner) unbindTurn(req TurnRequest) {
	if r.binder == nil || req.Ticket == "" || req.Binding == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	r.binder.Unbind(ctx, req.Ticket)
}

// RunTurn runs one turn, reusing or (re)creating the session's thread as needed.
func (r *Runner) RunTurn(ctx context.Context, req TurnRequest) (*TurnOutcome, error) {
	if req.MCPURL == "" {
		return nil, errors.New("codexagent: empty MCP URL")
	}
	if len(req.Messages) == 0 {
		return nil, errors.New("codexagent: no messages")
	}
	current := req.Messages[len(req.Messages)-1]

	// The directory Codex runs in (see newWorkDir). Failure is fatal: running in the
	// shared Codex's default cwd would expose that host's tree.
	workDir, cleanupDir, err := newWorkDir(req.WorkDir)
	if err != nil {
		return nil, err
	}
	defer cleanupDir()

	// Bind BEFORE any thread work: the binding IS the per-session turn lock (one atomically
	// acquired KV value), so if another turn already holds it, Bind returns
	// mcp.ErrCodexTurnBusy and this turn must not proceed. It stays held until the turn's
	// deferred unbind — there is no separate in-process lock.
	if err := r.bindTurn(ctx, req); err != nil {
		return nil, err
	}
	defer r.unbindTurn(req)

	// Ephemeral: no session id or no store => one-shot thread, never persisted.
	if req.SessionID == "" || r.store == nil {
		return r.runOnThread(ctx, req, current, req.Messages, workDir)
	}

	scopeFP := ScopeFingerprint(req.TenantID, req.DatasetIDs)
	// The web-search capability is decided per turn, but Codex caches an MCP tool
	// catalog for the life of a thread: a change must rebuild the thread, or the new
	// tool set is never advertised. Fold the decision into the reuse key.
	scopeFP = CombineFingerprints(scopeFP, WebSearchFingerprint(req.WebSearch))
	// The base instructions (citation contract) are baked into the thread at create and
	// not reloaded on resume, so they are part of the reuse key too. Fold them in as ONE
	// further hash: two 64-char hex strings joined would exceed the size:64 column.
	if req.Instructions != "" {
		scopeFP = CombineFingerprints(scopeFP, InstructionsFingerprint(req.Instructions))
	}
	// The history the thread must ALREADY have is everything before the current user
	// message. Normal growth (the previous answer + this question) is not drift; a
	// deletion or edit makes this differ from what we recorded after the last turn.
	seenFP := HistoryFingerprint(HistoryWithoutCurrent(req.Messages))

	rec, found, err := r.store.Get(ctx, req.TenantID, req.SessionID)
	if err != nil {
		return nil, err
	}

	// Reuse the thread only when scope AND the already-seen history still match
	// RAGFlow's truth AND the stored handle is still alive. thread/resume does not reload
	// mcp_servers, so a scope change still rebuilds (see the plan, D3); a history change
	// must replay (see P1#6); a dead handle means the thread's baked-in MCP URL would 401,
	// so it MUST rebuild (see P1#4).
	handleAlive := req.HandleAlive == nil || rec.Ticket == "" || req.HandleAlive(rec.Ticket)
	if found && rec.ScopeFingerprint == scopeFP && rec.HistoryFingerprint == seenFP && handleAlive {
		if th, err := r.client.ResumeThread(ctx, rec.ThreadID, r.threadOptions(req, workDir)...); err == nil {
			defer th.Close()
			out, err := r.consume(ctx, th, req, current)
			if err != nil {
				// Resume connected but the turn failed on a live thread: surface it
				// rather than silently rebuilding and double-running.
				return nil, err
			}
			out.ThreadID = rec.ThreadID
			// Advance the recorded history so the NEXT turn reuses: include this turn's
			// answer. (A failed write only costs a rebuild next turn.) Skip it if the
			// session was deleted mid-turn (handle revoked) — otherwise this Put would
			// resurrect the mapping the delete just removed.
			if r.handleLive(req) {
				_ = r.store.Put(ctx, ThreadRecord{
					TenantID:           req.TenantID,
					SessionID:          req.SessionID,
					ThreadID:           rec.ThreadID,
					ScopeFingerprint:   scopeFP,
					HistoryFingerprint: HistoryFingerprint(appendAssistant(req.Messages, out.Content)),
					Ticket:             rec.Ticket,
				})
			}
			return out, nil
		}
		// Resume failed (server restarted / thread reclaimed): fall through to rebuild.
	}

	// (Re)build: first turn, scope/history drift, or resume failure. The handle is
	// per-session and stable, so the rebuilt thread reuses the same MCP URL — there is
	// nothing to revoke.
	out, threadID, err := r.createAndRun(ctx, req, current, workDir)
	if err != nil {
		return nil, err
	}
	out.ThreadID = threadID
	out.Rebuilt = true
	// Record the history the thread now has: the incoming history PLUS this turn's
	// answer. The NEXT turn's already-seen history must equal this to reuse. Skip it if
	// the session was deleted mid-turn (handle revoked) — otherwise this Put would
	// resurrect the mapping the delete just removed.
	if r.handleLive(req) {
		if err := r.store.Put(ctx, ThreadRecord{
			TenantID:           req.TenantID,
			SessionID:          req.SessionID,
			ThreadID:           threadID,
			ScopeFingerprint:   scopeFP,
			HistoryFingerprint: HistoryFingerprint(appendAssistant(req.Messages, out.Content)),
			Ticket:             req.Ticket,
		}); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// handleLive reports whether this turn's handle is still alive, i.e. the session was not
// deleted mid-turn. When no liveness hook is set the turn is assumed live.
func (r *Runner) handleLive(req TurnRequest) bool {
	if req.HandleAlive == nil || req.Ticket == "" {
		return true
	}
	return req.HandleAlive(req.Ticket)
}

// appendAssistant returns msgs with an assistant message carrying content appended.
func appendAssistant(msgs []map[string]interface{}, content string) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(msgs)+1)
	out = append(out, msgs...)
	return append(out, map[string]interface{}{"role": "assistant", "content": content})
}

// createAndRun starts a fresh thread, replays the history (minus the current message),
// then runs the current turn.
func (r *Runner) createAndRun(ctx context.Context, req TurnRequest, current map[string]interface{}, workDir string) (*TurnOutcome, string, error) {
	th, err := r.client.StartThread(ctx, r.threadOptions(req, workDir)...)
	if err != nil {
		return nil, "", err
	}
	defer th.Close()

	items := MessagesToItems(HistoryWithoutCurrent(req.Messages))
	if len(items) > 0 {
		if err := r.client.ThreadInjectItems(ctx, codexgo.ThreadInjectItemsRequest{ThreadID: th.ID(), Items: items}); err != nil {
			return nil, th.ID(), err
		}
	}
	out, err := r.consume(ctx, th, req, current)
	return out, th.ID(), err
}

// runOnThread is the ephemeral path: a one-shot thread with the whole history replayed.
func (r *Runner) runOnThread(ctx context.Context, req TurnRequest, current map[string]interface{}, all []map[string]interface{}, workDir string) (*TurnOutcome, error) {
	th, err := r.client.StartThread(ctx, r.threadOptions(req, workDir)...)
	if err != nil {
		return nil, err
	}
	defer th.Close()
	items := MessagesToItems(HistoryWithoutCurrent(all))
	if len(items) > 0 {
		if err := r.client.ThreadInjectItems(ctx, codexgo.ThreadInjectItemsRequest{ThreadID: th.ID(), Items: items}); err != nil {
			return nil, err
		}
	}
	out, err := r.consume(ctx, th, req, current)
	if err != nil {
		return nil, err
	}
	out.ThreadID = th.ID()
	out.Rebuilt = true
	return out, nil
}

func (r *Runner) threadOptions(req TurnRequest, workDir string) []codexgo.ThreadOption {
	opts := []codexgo.ThreadOption{
		codexgo.WithThreadConfigOverride("mcp_servers.ragflow", map[string]interface{}{
			"url": req.MCPURL,
		}),
		// Turn off Codex's own provider-hosted (server-side) tools. For a custom
		// Responses provider the only hosted tool Codex emits is the built-in
		// `web_search`: the PROVIDER runs it and bills it (e.g. MiniMax's tools
		// pricing), and it is advertised to the model regardless of the conversation's
		// settings, so a chat with web search off would still reach the web. We never
		// want that — retrieval and any web search must go through this process, where
		// the conversation's scope and enablement are enforced. `web_search` is a
		// top-level config key (WebSearchMode; "disabled" removes the tool entirely);
		// image generation, the other hosted tool, is never emitted for a custom
		// provider (it requires OpenAI-issued auth), so this single switch closes every
		// server-side tool Codex could expose here.
		codexgo.WithThreadConfigOverride("web_search", "disabled"),
		// Capability isolation (see the plan, §5.5): reason + call MCP only.
		codexgo.WithThreadSandbox(codexgo.SandboxReadOnly),
		codexgo.WithThreadApprovalPolicy(codexgo.ApprovalUntrusted()),
	}
	if req.Model != "" {
		opts = append(opts, codexgo.WithThreadModel(req.Model))
	}
	// Only declare a per-thread provider when one is supplied; otherwise the thread
	// uses whatever provider the shared Codex is configured with.
	if req.ProviderID != "" {
		opts = append(opts,
			codexgo.WithThreadModelProvider(req.ProviderID),
			codexgo.WithThreadConfigOverride("model_providers."+req.ProviderID, map[string]interface{}{
				"name":                      req.ProviderID,
				"base_url":                  req.ProviderBaseURL,
				"wire_api":                  "responses",
				"experimental_bearer_token": req.ProviderToken,
			}),
		)
	}
	if req.Instructions != "" {
		opts = append(opts, codexgo.WithThreadBaseInstructions(req.Instructions))
	}
	// Keep Codex inside the empty, process-private directory created for this turn.
	if workDir != "" {
		opts = append(opts, codexgo.WithThreadCWD(workDir))
	}
	return opts
}

// newWorkDir returns the directory Codex runs in for one turn. A configured dir is
// operator-managed (created if absent, never removed); otherwise a fresh local temp dir
// is created and removed at turn end. Either way, failure is an error — silently falling
// back to the shared Codex's default cwd would expose that host's tree.
func newWorkDir(configured string) (string, func(), error) {
	if configured != "" {
		if err := os.MkdirAll(configured, 0o700); err != nil {
			return "", func() {}, fmt.Errorf("codexagent: work dir %q: %w", configured, err)
		}
		return configured, func() {}, nil
	}
	dir, err := os.MkdirTemp("", "codex-m7-*")
	if err != nil {
		return "", func() {}, fmt.Errorf("codexagent: create work dir: %w", err)
	}
	return dir, func() { _ = os.RemoveAll(dir) }, nil
}

// consume runs one turn on an already-connected thread via RunStreamed (the sync Run
// is incompatible with the app-server: it polls thread/read, which this codex does not
// support) and assembles the answer from the event stream.
//
// A stream that closes without a terminal turn is a failure, never a partial success: a
// dropped connection must not ship a half answer as final (P1#5).
func (r *Runner) consume(ctx context.Context, th *codexgo.SessionThread, req TurnRequest, current map[string]interface{}) (*TurnOutcome, error) {
	runCtx := ctx
	if req.TurnTimeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(ctx, req.TurnTimeout)
		defer cancel()
	}

	// RunStreamedTurn returns the server-assigned turn id synchronously, before any event
	// is delivered, so a timeout can always interrupt the remote turn — even if no
	// id-carrying event was consumed yet. The event stream is otherwise identical to
	// RunStreamed (subscribe-before-start included).
	events, turnID, err := th.RunStreamedTurn(runCtx, currentText(current))
	if err != nil {
		return nil, err
	}

	out := &TurnOutcome{}
	var content, reasoning strings.Builder
	var items []codexgo.Item
	completed := false
	// A detail-less error notification captured before the terminal turn.
	var streamErr error

	for {
		select {
		case <-runCtx.Done():
			// The local cancel only frees this consumer; the server-side turn keeps
			// running (calling MCP, burning model budget) unless we interrupt it.
			r.interrupt(th, turnID)
			return nil, runCtx.Err()
		case ev, ok := <-events:
			if !ok {
				// ctx cancellation also tears the turn's subscription down, so this
				// branch — not only the ctx.Done case above — is a cancellation exit.
				// Interrupt the remote turn on any non-terminal exit.
				if !completed {
					r.interrupt(th, turnID)
					if err := runCtx.Err(); err != nil {
						return nil, err
					}
					if streamErr != nil {
						return nil, streamErr
					}
					return nil, errors.New("codexagent: event stream closed before the turn completed")
				}
				out.Content = content.String()
				out.Reasoning = reasoning.String()
				return out, nil
			}
			switch e := ev.Raw.(type) {
			case codexgo.TurnStartedEvent:
			case codexgo.ItemStartedEvent:
			case codexgo.ItemCompletedEvent:
				if e.Item != nil {
					items = append(items, *e.Item)
				}
			case codexgo.ItemAgentMessageDeltaEvent:
				if e.Text != "" {
					content.WriteString(e.Text)
					if req.OnContent != nil {
						req.OnContent(e.Text)
					}
				}
			case codexgo.ItemReasoningTextDeltaEvent:
				if e.Text != "" {
					reasoning.WriteString(e.Text)
					if req.OnReasoning != nil {
						req.OnReasoning(e.Text)
					}
				}
			case codexgo.ItemReasoningSummaryTextDeltaEvent:
				if e.Text != "" {
					reasoning.WriteString(e.Text)
					if req.OnReasoning != nil {
						req.OnReasoning(e.Text)
					}
				}
			case codexgo.ErrorEvent:
				msg := e.Message
				if msg == "" && len(e.Data) > 0 {
					msg = string(e.Data)
				}
				if e.Code != "" || msg != "" {
					return nil, fmt.Errorf("codexagent: turn error (code=%q): %s", e.Code, msg)
				}
				// The app-server can emit a detail-less error notification just
				// before a failed turn/completed, whose turn.error carries the real
				// cause (e.g. the upstream provider's HTTP status). Keep the bare
				// notification and keep reading so that richer detail wins —
				// returning here would surface an empty message and hide the cause.
				if streamErr == nil {
					streamErr = errors.New("codexagent: the Codex server reported an error with no detail")
				}
			case codexgo.TurnCompletedEvent:
				completed = true
				// A terminal turn that failed or was interrupted is NOT success.
				if status := turnStatus(e); status == codexgo.TurnStatusFailed || status == codexgo.TurnStatusInterrupted {
					return nil, fmt.Errorf("codexagent: turn %s: %s", status, turnErrorDetail(e.Turn, streamErr))
				}
				// The authoritative answer is the turn's final agent message. Prefer
				// the completed message items (item/completed), then the turn's own
				// items, then the raw delta stream — which can carry earlier process
				// narration ("Searching…") and can miss text after a reconnect.
				res := &codexgo.TurnResult{Items: items, DeltaText: content.String()}
				if e.Turn != nil {
					res.Turn = *e.Turn
				}
				out.Content = res.FinalAgentText()
				if out.Content == "" {
					out.Content = content.String()
				}
				out.Reasoning = reasoning.String()
				return out, nil
			}
		}
	}
}

func turnStatus(e codexgo.TurnCompletedEvent) codexgo.TurnStatus {
	if e.Turn != nil && e.Turn.Status != "" {
		return e.Turn.Status
	}
	return e.Status
}

// turnErrorDetail returns the most specific failure text available: the turn's own
// error payload when present, otherwise a detail-less error notification captured
// earlier, otherwise a fixed placeholder.
func turnErrorDetail(t *codexgo.Turn, streamErr error) string {
	if t != nil && len(t.Error) > 0 {
		return string(t.Error)
	}
	if streamErr != nil {
		return streamErr.Error()
	}
	return "no error detail"
}

// interrupt asks the server to stop a turn we have given up waiting on, in its own short
// deadline so a dead transport cannot wedge the caller.
func (r *Runner) interrupt(th *codexgo.SessionThread, turnID string) {
	if turnID == "" {
		return
	}
	ictx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = th.Interrupt(ictx, turnID)
}

func currentText(msg map[string]interface{}) string {
	if msg == nil {
		return ""
	}
	s, _ := msg["content"].(string)
	return s
}
