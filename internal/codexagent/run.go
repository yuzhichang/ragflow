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
	"sync"
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
	// Ticket is the MCP ticket token baked into this thread's config, kept so a
	// rebuild or a session delete can revoke it.
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

	// Model / provider: the resolved dialog model (see the plan, D4). Model is the
	// real model name, not the composite llm_id.
	Model           string
	ProviderID      string
	ProviderBaseURL string
	ProviderToken   string

	// MCPURL is the full /mcp/codex/<ticket> URL this process exposes to Codex.
	MCPURL string
	// Ticket is the token inside MCPURL (for revocation on rebuild).
	Ticket string
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
	// TicketUsable reports whether a stored ticket is still resolvable. When it returns
	// false a reuse is refused and the thread is rebuilt, so a thread whose MCP URL has a
	// dead ticket (expired, or in-memory tickets lost to a restart / another replica) is
	// never resumed. nil means "assume usable".
	TicketUsable func(token string) bool
	// OnThreadReplaced is called when a thread is rebuilt or rebuilt-on-miss, so the
	// caller can revoke the previous thread's ticket.
	OnThreadReplaced func(oldThreadID, oldTicket string)
	// OnTurnStart runs under the session turn lock, just before the turn executes.
	// OnTurnEnd runs after it, still under the same lock. Together they let a caller
	// reset and snapshot per-session state (e.g. a shared ledger) without a second turn
	// for the same session interleaving between the reset and the snapshot.
	OnTurnStart func()
	OnTurnEnd   func()
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
type Runner struct {
	client *codexgo.Client
	store  ThreadStore
	locks  *keyedLocks
}

// keyedLocks hands out one mutex per (tenant, session), refcounted so an idle entry can be
// pruned without ever letting two turns for the same session run at once. A plain
// map[key]*sync.Mutex cannot be pruned safely: deleting an entry while a caller still
// holds its mutex lets a later caller get a fresh mutex for the same key and run in
// parallel.
//
// This serialization is PROCESS-LOCAL by design: RAGFlow guarantees single sign-on, so a
// session is served by one API process at a time. If that ever changes (multi-process
// concurrent turns for one session), this must become a cross-instance lock (e.g. a DB
// advisory lock on (tenant, session) held for the whole turn) — see the plan, §13 (P1-5).
type keyedLocks struct {
	mu    sync.Mutex
	locks map[string]*keyedLock
}

type keyedLock struct {
	mu        sync.Mutex
	refs      int
	forgotten bool
}

func newKeyedLocks() *keyedLocks {
	return &keyedLocks{locks: make(map[string]*keyedLock)}
}

// acquire returns the lock for key, already held. The caller MUST release it with the
// same key and lock.
func (k *keyedLocks) acquire(key string) *keyedLock {
	k.mu.Lock()
	l := k.locks[key]
	if l == nil {
		l = &keyedLock{}
		k.locks[key] = l
	}
	l.refs++
	k.mu.Unlock()
	l.mu.Lock()
	return l
}

func (k *keyedLocks) release(key string, l *keyedLock) {
	l.mu.Unlock()
	k.mu.Lock()
	l.refs--
	if l.refs == 0 && l.forgotten {
		delete(k.locks, key)
	}
	k.mu.Unlock()
}

// forget drops key once no holder remains. While a turn holds the lock the entry stays
// (marked), so an in-flight turn is never orphaned and mutual exclusion is preserved.
func (k *keyedLocks) forget(key string) {
	k.mu.Lock()
	if l := k.locks[key]; l != nil {
		if l.refs == 0 {
			delete(k.locks, key)
		} else {
			l.forgotten = true
		}
	}
	k.mu.Unlock()
}

func sessionKey(tenantID, sessionID string) string {
	return tenantID + "\x00" + sessionID
}

// NewRunner returns a runner. store may be nil (ephemeral-only).
func NewRunner(client *codexgo.Client, store ThreadStore) *Runner {
	return &Runner{client: client, store: store, locks: newKeyedLocks()}
}

// ForgetSession drops the session's in-memory lock entry (called when the session is
// deleted). It is a no-op while a turn still holds the lock.
func (r *Runner) ForgetSession(tenantID, sessionID string) {
	r.locks.forget(sessionKey(tenantID, sessionID))
}

// Exclusive runs fn while holding the session's turn lock. A session delete uses it so it
// cannot interleave with an in-flight turn — otherwise the turn could Put (re-create) the
// mapping and ticket the delete just removed.
func (r *Runner) Exclusive(tenantID, sessionID string, fn func()) {
	key := sessionKey(tenantID, sessionID)
	l := r.locks.acquire(key)
	defer r.locks.release(key, l)
	fn()
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

	// Ephemeral: no session id or no store => one-shot thread, never persisted.
	if req.SessionID == "" || r.store == nil {
		if req.OnTurnStart != nil {
			req.OnTurnStart()
		}
		out, err := r.runOnThread(ctx, req, current, req.Messages, workDir)
		if req.OnTurnEnd != nil {
			req.OnTurnEnd()
		}
		return out, err
	}

	lockKey := sessionKey(req.TenantID, req.SessionID)
	lock := r.locks.acquire(lockKey)
	defer r.locks.release(lockKey, lock)
	// Reset/snapshot shared per-session state strictly inside the turn lock: a second
	// turn for this session cannot interleave between the start reset and the end
	// snapshot. (release is deferred first, so OnTurnEnd runs before it.)
	if req.OnTurnStart != nil {
		req.OnTurnStart()
	}
	if req.OnTurnEnd != nil {
		defer req.OnTurnEnd()
	}

	scopeFP := ScopeFingerprint(req.TenantID, req.DatasetIDs)
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
	// RAGFlow's truth AND the stored ticket is still usable. thread/resume does not
	// reload mcp_servers, so a scope change MUST rebuild (see the plan, D3); a history
	// change must replay (see P1#6); a dead ticket means the thread's baked-in MCP URL
	// would 401, so it MUST rebuild (see P1#4).
	ticketUsable := req.TicketUsable == nil || rec.Ticket == "" || req.TicketUsable(rec.Ticket)
	if found && rec.ScopeFingerprint == scopeFP && rec.HistoryFingerprint == seenFP && ticketUsable {
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
			// answer. (A failed write only costs a rebuild next turn.)
			_ = r.store.Put(ctx, ThreadRecord{
				TenantID:           req.TenantID,
				SessionID:          req.SessionID,
				ThreadID:           rec.ThreadID,
				ScopeFingerprint:   scopeFP,
				HistoryFingerprint: HistoryFingerprint(appendAssistant(req.Messages, out.Content)),
				Ticket:             rec.Ticket,
			})
			return out, nil
		}
		// Resume failed (server restarted / thread reclaimed): fall through to rebuild.
	}

	// (Re)build: first turn, scope/history drift, or resume failure.
	if found && req.OnThreadReplaced != nil {
		req.OnThreadReplaced(rec.ThreadID, rec.Ticket)
	}
	out, threadID, err := r.createAndRun(ctx, req, current, workDir)
	if err != nil {
		return nil, err
	}
	out.ThreadID = threadID
	out.Rebuilt = true
	// Record the history the thread now has: the incoming history PLUS this turn's
	// answer. The NEXT turn's already-seen history must equal this to reuse.
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
	return out, nil
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
