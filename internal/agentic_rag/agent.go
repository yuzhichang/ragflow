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

package agentic_rag

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"go.uber.org/zap"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"
	"ragflow/internal/entity/models"
)

// Input carries everything Run needs to spin up a one-shot ReAct conversation
// turn. It is intentionally decoupled from the canvas runtime: the caller
// builds the model and messages directly.
type Input struct {
	// Model is the chat model backing the agent. It must support tool calling
	// (model.ToolCallingChatModel) — models.EinoChatModel does.
	Model *models.EinoChatModel
	// SynthModel, when non-nil, is the model the last-resort synthesis
	// (finalizeAnswer) calls. It must be a SEPARATE instance over the same
	// failover chain as Model: a failover instance caches the error of its
	// last full-chain failure and short-circuits every later call with it for
	// 30s, so sharing the agent's instance hands the synthesis a stale error
	// from whatever malformed call tripped the cooldown — which is how a
	// request with no tool messages in it once "failed" with
	// "tool result's tool id ... not found". Nil falls back to Model.
	SynthModel *models.EinoChatModel
	// Messages are the conversation history plus the current user message —
	// one turn per Run. The next user input is the NEXT Run (AsyncChat is
	// re-entered per request), so the history the caller passes in must
	// already be compact: one final answer per earlier turn, never the ReAct
	// trajectory that produced it.
	Messages []*schema.Message
	// MaxIterations caps the ReAct loop. Zero falls back to a sane default.
	MaxIterations int
	// Stream controls whether model output is streamed into the event iterator.
	Stream bool
	// TenantID is the conversation's tenant (the chat's bound tenant, decided by
	// the session created in the UI). It is injected into the retrieval tools so
	// they search the right index without a canvas runtime.
	TenantID string
	// DatasetIDs is the conversation's bound dataset scope, decided by the
	// session created in the UI. It is injected into the retrieval tools.
	DatasetIDs []string
	// Tools are the eino tools the agent may call. When empty, the tool set of
	// the resolved template (TemplateID, else the config's default) is used.
	// The web_search tool is NOT passed here: it is injected from the run's
	// context (see WithWebSearch), which is what keeps one conversation's
	// internet capability from leaking into another's.
	Tools []tool.BaseTool
	// TemplateID selects the agent template from conf/agentic_rag.yaml by id
	// (e.g. "smart-reasoning", "smart-grep"), carried through from the
	// request's agent_mode. It is mandatory: an empty or unknown id fails the
	// run with a clear error — there is no default/env/first-template fallback.
	TemplateID string
	// OnDelta, when non-nil, receives each incremental (content, reasoning)
	// delta as it streams. When nil, deltas fall back to runtime.EmitAgentMessage.
	OnDelta func(contentDelta, thinkingDelta string)
	// ToolCallCounts, when non-nil, is filled with the number of times each tool
	// was invoked during this agent turn, keyed by tool name. Tools never called
	// are omitted. Useful for per-question usage accounting.
	ToolCallCounts map[string]int
	// ToolCallErrors, when non-nil, tallies how many tool calls returned a
	// failure notice (<tool_error> / severity="error") during this turn,
	// keyed by tool name. A backend outage or repeated invalid arguments
	// shows up here instead of hiding inside the call counts.
	ToolCallErrors map[string]int
	// ToolErrorSamples, when non-nil, records one representative failure
	// message per tool (the first occurrence, truncated) so a benchmark row
	// is diagnosable without opening the server logs.
	ToolErrorSamples map[string]string
	// ToolCallDurations, when non-nil, accumulates the total wall-clock duration
	// of every tool invocation during this agent turn, keyed by tool name. Pair
	// this with ToolCallCounts to derive per-call average latency per tool.
	ToolCallDurations *durationAccumulator
	// RetrievedDocIDs, when non-nil, collects every document identifier any
	// retrieval tool surfaced during this agent turn (both doc_id and doc_name,
	// deduplicated). Citation payloads only carry what the final answer quotes,
	// but benchmarks score retrieval recall against everything the agent ever
	// pulled, so the tally has to be taken per tool call.
	RetrievedDocIDs *docIDLedger
	// ChunkReads, when non-nil, accumulates how many chunks the retrieval
	// tools put in front of the model, split by read depth (full content =
	// deep, snippet window = shallow). Benchmarks report both totals per
	// question next to ToolCallCounts.
	ChunkReads *chunkReadLedger
	// GateAudit, when non-nil, is filled by the delivery gate with the
	// answer auditor's per-round suspect accounting (see GateAuditRecord).
	// Benchmarks report it next to ToolCallCounts to show how much audit
	// effort a question consumed.
	GateAudit *GateAuditRecord
}

// GateAuditRecord carries the delivery gate's per-round audit accounting for
// one agent turn. Suspects holds the suspect count of EVERY audit the gate
// ran, oldest first — the whole curve, pass rounds included (a PASS verdict
// reports 0 suspects) — so a consumer can see how the count evolved
// (15→11→6→5→0 reads very differently from a flat 1,1,1,1). Passed reports
// whether the LAST audit the gate ran concluded PASS; a gate that exited on
// stall, budget or time leaves it false.
type GateAuditRecord struct {
	Suspects []int `json:"suspects,omitempty"`
	Passed   bool  `json:"passed"`
}

// defaultMaxIterations caps the ReAct loop before the agent must answer. It is
// raised beyond the original 50 because multi-step benchmark questions (multiple
// hops, off-by-one arithmetic) routinely need more than 50 model turns, and the
// duplicate-retrieval guard (todo 3) prevents the extra budget from being wasted
// on repeated identical searches.
const defaultMaxIterations = 120

var errNilModel = errors.New("agentic_rag: model is required")

// llmRetryMax bounds retry attempts per model call on top of the initial one
// (adk semantics: MaxRetries=3 → up to 4 calls). With the backoff below the
// worst-case added latency per call is ~17s.
const llmRetryMax = 3

// agentModelRetryConfig returns the eino-native retry policy attached to
// every ChatModelAgent this package builds (explorer, repair turns, and the
// answer_auditor auditor): provider hiccups — MiniMax 529 overload, 429 rate
// limits, 5xx, connection resets — are retried with seconds-scale backoff,
// while client errors (400/401) and exhausted deadlines fail fast.
//
// Two design guards:
//  1. Retries are refused once partial content has streamed out
//     (OutputMessage != nil): adk forwards stream deltas to the client in
//     real time and only defers the retry decision, so a retried mid-stream
//     failure would duplicate already-emitted text.
//  2. context.Canceled / DeadlineExceeded are never retried — the 300s
//     Generate budget already spent is not doubled by another attempt.
func agentModelRetryConfig() *adk.ModelRetryConfig {
	return &adk.ModelRetryConfig{
		MaxRetries: llmRetryMax,
		ShouldRetry: func(_ context.Context, rc *adk.RetryContext) *adk.RetryDecision {
			if rc.Err == nil || rc.OutputMessage != nil {
				return nil // success, or partial output already delivered: accept as-is
			}
			if errors.Is(rc.Err, context.Canceled) || errors.Is(rc.Err, context.DeadlineExceeded) {
				// A send/transport-phase failure is transient even though it
				// wraps a deadline ("failed to send request: Post ...:
				// context deadline exceeded" is the endpoint being briefly
				// unreachable, ~seconds); only the Generate-level budget
				// ("EinoChatModel.Generate(...): context deadline exceeded
				// (300s...)") has already spent the turn's allowance and must
				// fail fast instead of doubling it.
				if !transientLLMError(rc.Err) {
					return nil
				}
			}
			if transientLLMError(rc.Err) {
				return &adk.RetryDecision{Retry: true}
			}
			return nil // 400/401/403 and friends: retrying cannot help
		},
		BackoffFunc: func(_ context.Context, attempt int) time.Duration {
			// Overload (529) recovery needs seconds, not adk's default
			// 100ms ramp: 2s → 5s → 10s.
			switch {
			case attempt <= 1:
				return 2 * time.Second
			case attempt == 2:
				return 5 * time.Second
			default:
				return 10 * time.Second
			}
		},
	}
}

// transientLLMError reports whether an LLM error is worth retrying: provider
// overload (529), rate limiting (429), other 5xx server errors, and
// network-level hiccups. Matched by message substrings because the drivers
// wrap provider bodies as text (e.g. "status 529: ... overloaded_error").
//
// 用量上限 IS included even though it names MiniMax's plan wall: the gateway
// reports short TPM/RPM bursts with the SAME wording it uses for true plan
// exhaustion, and the message carries no HTTP status to tell them apart. A
// burst recovers within seconds, so backing off is right; a genuine wall costs
// at most llmRetryMax retries (2s → 5s → 10s, ~17s) before failing fast, which
// is the price of not aborting whole benchmark runs on a momentary burst. The
// alternative - treating the string as terminal - is what made concurrent runs
// report "quota exhausted" while the plan still had headroom.
func transientLLMError(err error) bool {
	msg := err.Error()
	for _, marker := range []string{
		"529", "overloaded_error", "429", "rate limit", "status 5",
		// MiniMax's plan limits arrive as Chinese prose with no status code.
		// 速率限制 is the 429 wording and 用量上限 the wall/burst wording;
		// both are retryable - see the comment above for the burst rationale.
		"速率限制", "用量上限",
		// transport phase (safe to retry even when it wraps a deadline)
		"failed to send request", "dial tcp", "i/o timeout", "tls handshake",
		"connection reset", "connection refused",
	} {
		if strings.Contains(msg, marker) {
			return true
		}
	}
	return false
}

// durationAccumulator is a concurrency-safe total-duration tally keyed by tool
// name. The agent executes same-round tool_calls in parallel, so callers may
// invoke Add from multiple goroutines; the mutex guards the shared map.
type durationAccumulator struct {
	mu  sync.Mutex
	dur map[string]time.Duration
}

func NewDurationAccumulator() *durationAccumulator {
	return &durationAccumulator{dur: make(map[string]time.Duration)}
}

func (a *durationAccumulator) Add(name string, d time.Duration) {
	if a == nil || name == "" {
		return
	}
	a.mu.Lock()
	a.dur[name] += d
	a.mu.Unlock()
}

func (a *durationAccumulator) Snapshot() map[string]time.Duration {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make(map[string]time.Duration, len(a.dur))
	for k, v := range a.dur {
		out[k] = v
	}
	return out
}

// docIDLedger accumulates the set of documents a run's FULL-CONTENT tools
// (search_chunks, list_chunks) surfaced. Benchmarks (BrowseComp-Plus) score
// retrieval recall against this union, not just the cited subset, so the
// tally is taken as results come back. Only the tools' rendered doc_name is
// stored, normalized to its file stem (see recordedDocIDs): a benchmark
// corpus keys its evidence on file stems (e.g. "5412" from "5412.md"), which
// the opaque doc_id cannot match. Locate-only tools (search_bm25_chunks,
// grep_chunks) do NOT feed the ledger — they surface a snippet window, not
// the document.
type docIDLedger struct {
	mu  sync.Mutex
	ids map[string]struct{}
}

// NewDocIDLedger returns an empty document ledger for one run.
func NewDocIDLedger() *docIDLedger {
	return &docIDLedger{ids: make(map[string]struct{})}
}

// Add records the given document identifiers. Empty values are dropped and
// duplicates collapse, so callers can pass raw tool output matches as-is.
func (l *docIDLedger) Add(ids ...string) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, id := range ids {
		if id == "" {
			continue
		}
		l.ids[id] = struct{}{}
	}
}

// Snapshot returns the recorded identifiers in sorted order. Sorted so the
// payload a benchmark archives is byte-stable across runs.
func (l *docIDLedger) Snapshot() []string {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]string, 0, len(l.ids))
	for id := range l.ids {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// chunkReadLedger accumulates HOW MANY chunks a run's retrieval tools put in
// front of the model, split by read depth:
//   - DEEP reads (list_chunks, search_chunks): full chunk content rendered —
//     the expensive reads;
//   - SHALLOW reads (grep_chunks, search_bm25_chunks): <match_snippet> windows
//     only — the cheap triage reads.
//
// Benchmarks report the two totals per question next to tool_call_counts: the
// split shows whether a run's context weight came from reading documents or
// from triage traffic, which raw tool call counts cannot distinguish.
type chunkReadLedger struct {
	mu      sync.Mutex
	deep    int
	shallow int
}

// NewChunkReadLedger returns an empty chunk-read ledger for one run.
func NewChunkReadLedger() *chunkReadLedger {
	return &chunkReadLedger{}
}

// AddDeep records n deep-read chunks (full chunk content).
func (l *chunkReadLedger) AddDeep(n int) {
	if l == nil || n <= 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.deep += n
}

// AddShallow records n shallow-read chunks (snippet windows).
func (l *chunkReadLedger) AddShallow(n int) {
	if l == nil || n <= 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.shallow += n
}

// Snapshot returns (deepReadChunks, shallowReadChunks) accumulated so far.
func (l *chunkReadLedger) Snapshot() (deep, shallow int) {
	if l == nil {
		return 0, 0
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.deep, l.shallow
}

// instrumentedTool decorates a tool.InvokableTool, accumulating the total
// wall-clock duration of every InvokableRun into a shared accumulator keyed by
// the wrapped tool's name, and (when docs is non-nil) recording the documents
// a FULL-CONTENT tool surfaced into the run's retrieval ledger. It embeds
// tool.InvokableTool (which itself embeds tool.BaseTool), promoting Info, then
// overrides InvokableRun to time the call.

// fullContentDocTools are the retrieval tools whose output carries COMPLETE
// chunk content. Only these feed the retrieval ledger: the locate-only tools
// (search_bm25_chunks, grep_chunks) surface a <match_snippet> window for
// relevance triage, not the document itself — a doc seen only through them was
// located, not retrieved.
var fullContentDocTools = map[string]struct{}{
	"search_chunks": {},
	"list_chunks":   {},
}

// deepReadChunkTools render FULL chunk content — their <chunk> elements are
// counted as deep reads. shallowReadChunkTools render only a
// <match_snippet> window per chunk — counted as shallow reads.
var deepReadChunkTools = map[string]struct{}{
	"search_chunks": {},
	"list_chunks":   {},
}

var shallowReadChunkTools = map[string]struct{}{
	"grep_chunks":        {},
	"search_bm25_chunks": {},
}

// chunkElementRe counts the <chunk ...> elements a tool rendered. The
// <chunks ...> wrapper does NOT match: after "<chunk" comes "s", not a space
// or a tag close.
var chunkElementRe = regexp.MustCompile(`<chunk[ >]`)

// countChunkElements returns the number of <chunk ...> elements in a rendered
// tool payload — one element per chunk put in front of the model, whether it
// carries full content or a snippet.
func countChunkElements(out string) int {
	return len(chunkElementRe.FindAllStringIndex(out, -1))
}

// The ledger is fed by regex over the rendered output rather than by each tool
// reporting its chunks: every locate tool and list_chunks already render
// doc_id= / doc_name= attributes, so one wrapper covers the whole toolset
// without touching any tool implementation. web_search renders url/title
// instead, so web hits are correctly not counted as corpus documents.
type instrumentedTool struct {
	tool.InvokableTool
	acc    *durationAccumulator
	docs   *docIDLedger
	chunks *chunkReadLedger
}

// toolOutputDocNameRe pulls the document names out of the XML every retrieval
// tool renders. The opaque doc_id (32-hex) is deliberately NOT recorded: the
// benchmark corpora key their evidence and expected documents on file stems.
var toolOutputDocNameRe = regexp.MustCompile(`doc_name="([^"]+)"`)

// docNameStem normalizes a rendered doc_name to the file stem a benchmark
// corpus keys its evidence on: the base name with the ".md" extension
// stripped ("75314.md" -> "75314").
func docNameStem(name string) string {
	if i := strings.LastIndexByte(name, '/'); i >= 0 {
		name = name[i+1:]
	}
	return strings.TrimSuffix(name, ".md")
}

// recordedDocIDs returns the document stems mentioned in a tool's output,
// deduplicated. Both "75314.md" and "75314" normalize to "75314".
func recordedDocIDs(out string) []string {
	if out == "" {
		return nil
	}
	ids := make([]string, 0, 8)
	for _, match := range toolOutputDocNameRe.FindAllStringSubmatch(out, -1) {
		if stem := docNameStem(match[1]); stem != "" {
			ids = append(ids, stem)
		}
	}
	return ids
}

func (t *instrumentedTool) InvokableRun(ctx context.Context, args string, opts ...tool.Option) (string, error) {
	start := time.Now()
	out, err := t.InvokableTool.InvokableRun(ctx, args, opts...)
	cost := time.Since(start)
	name := ""
	if info, ierr := t.InvokableTool.Info(ctx); ierr == nil && info != nil {
		name = info.Name
	}
	t.acc.Add(name, cost)
	if err == nil {
		if _, full := fullContentDocTools[name]; full {
			t.docs.Add(recordedDocIDs(out)...)
		}
		// Chunk-read accounting: one <chunk> element per chunk the tool put
		// in front of the model, split by read depth (full content vs
		// snippet window). Only the four corpus retrieval tools render
		// <chunk> elements; web_search renders url/title instead.
		if n := countChunkElements(out); n > 0 {
			if _, deep := deepReadChunkTools[name]; deep {
				t.chunks.AddDeep(n)
			} else if _, shallow := shallowReadChunkTools[name]; shallow {
				t.chunks.AddShallow(n)
			}
		}
	}
	fields := []zap.Field{
		zap.String("tool", name),
		zap.Float64("cost_ms", float64(cost.Milliseconds())),
		zap.String("args", truncateForLog(args, 200)),
	}
	if err != nil {
		fields = append(fields, zap.Error(err))
	}
	common.DebugCtx(ctx, "agentic_rag: tool call", fields...)
	return out, err
}

// Run executes a smart-reasoning (ReAct) turn against the given model and
// returns the final assistant message content, streaming incremental deltas
// through in.OnDelta (or runtime.EmitAgentMessage when OnDelta is nil).
//
// One call answers ONE user turn: the caller's stored conversation comes in
// through in.Messages and the next user input arrives as the NEXT call
// (ChatPipelineService.AsyncChat is re-entered per request). Keeping the
// history compact is therefore the caller's job at the turn boundary — a
// stored assistant message must hold the turn's final answer, not the ReAct
// trajectory that produced it (see appendAssistantToSession's callers).
//
// It uses eino ADK's adk.ChatModelAgent (NOT flow/agent/react, and never
// adk/react.go directly), which provides the ReAct loop plus robustness
// (retry/failover/cancel monitoring) and recoverability (checkpoint/resume).
func Run(ctx context.Context, in Input) (string, error) {
	if in.Model == nil {
		return "", errNilModel
	}

	tmpl, errT := resolveTemplateFor(in.TemplateID)
	if errT != nil {
		common.ErrorCtx(ctx, "agentic_rag: resolve template", errT)
		return "", errT
	}

	// Answer-audit gate state: created for templates that declare
	// `audit_max_pass: N` (N > 0), which is both the switch and the budget —
	// how many audit passes the gate may spend. The auditor is a STANDALONE
	// agent deliberately kept out of the main agent's toolset — the agent only
	// produces the FINAL message, and the delivery gate drives the auditor
	// itself (see runDeliveryGate), so compliance cannot be defied. An explicit
	// tool set means the caller owns the toolset, so the template's auditor
	// does not apply.
	auditMaxPass := tmpl.AuditMaxPass
	if len(in.Tools) > 0 {
		auditMaxPass = 0
	}
	var auditorAgent adk.Agent

	// Shared per-run tool ledgers: initialized BEFORE the auditor is built so
	// its deep-reads land in the same accumulators as the explorer's calls.
	if in.ToolCallDurations == nil {
		in.ToolCallDurations = NewDurationAccumulator()
	}
	if in.RetrievedDocIDs == nil {
		in.RetrievedDocIDs = NewDocIDLedger()
	}
	if in.ChunkReads == nil {
		in.ChunkReads = NewChunkReadLedger()
	}

	tools := in.Tools
	if len(tools) == 0 {
		// No explicit tool set: build it from the requested template, which
		// lets operators change the tool list without recompiling. The config
		// is reloaded from disk when its mtime changes, so a quick edit +
		// re-run is enough to try a different tool subset or prompt variant.
		tools = toolsFor(tmpl, in.TenantID, in.DatasetIDs)
		// Web search is injected, never declared: the template's tool list
		// describes the corpus toolset, and a conversation whose context
		// carries a provider gets one extra tool at run time. The same
		// context reaches the auditor, so both agents run with the same
		// capability.
		tools = append(tools, webSearchTools(ctx)...)
		// The audited question is pinned into the auditor's system prompt, so
		// it is built for THIS run's question.
		if auditMaxPass > 0 {
			inner, aerr := NewAnswerAuditorAgent(ctx, in.Model,
				in.TenantID, in.DatasetIDs, lastUserQuestion(in.Messages),
				in.ToolCallDurations, in.RetrievedDocIDs, in.ChunkReads)
			if aerr != nil {
				// auditMaxPass drops to 0: no gate auditing.
				common.WarnCtx(ctx, "agentic_rag: answer auditor unavailable", zap.Error(aerr))
				auditMaxPass = 0
			} else {
				auditorAgent = inner
			}
		}
	}

	maxIter := in.MaxIterations
	if maxIter <= 0 {
		maxIter = defaultMaxIterations
	}

	// Wrap every invokable tool so each call is timed, emitted as a debug log,
	// accumulated into in.ToolCallDurations, and (for the full-content tools
	// search_chunks/list_chunks) has the documents it surfaced recorded into
	// in.RetrievedDocIDs. Tools that aren't InvokableTool (e.g.
	// streamable-only) are passed through unwrapped — all agent tools here are
	// InvokableTool.
	wrapped := make([]tool.BaseTool, len(tools))
	for i, t := range tools {
		if it, ok := t.(tool.InvokableTool); ok {
			wrapped[i] = &instrumentedTool{
				InvokableTool: it,
				acc:           in.ToolCallDurations,
				docs:          in.RetrievedDocIDs,
				chunks:        in.ChunkReads,
			}
		} else {
			wrapped[i] = t
		}
	}
	tools = wrapped

	common.DebugCtx(ctx, "agentic_rag: run start",
		zap.Int("max_iterations", maxIter),
		zap.Int("messages", len(in.Messages)),
		zap.Int("tools", len(tools)),
	)

	cfg := &adk.ChatModelAgentConfig{
		Name:          "smart-reasoning",
		Instruction:   instructionFor(tmpl),
		Model:         in.Model,
		MaxIterations: maxIter,
		// Retry transient provider failures (MiniMax 529 overload, 429, 5xx,
		// network hiccups) with seconds-scale backoff on EVERY model call of
		// this agent — main-loop research turns, repair turns, and finalize
		// alike. Without it one 529 mid-research aborts a 100-iteration run.
		ModelRetryConfig: agentModelRetryConfig(),
		ToolsConfig: adk.ToolsConfig{
			ToolsNodeConfig: compose.ToolsNodeConfig{
				Tools: tools,
				// Execute same-round tool calls (multiple tool_calls in one model
				// response) concurrently rather than one-after-another. This is
				// eino's default (false), but we state it explicitly so the
				// parallel intent is not accidental. All six tools
				// (think/todo_write/grep_chunks/search_chunks/list_chunks/run_javascript)
				// are concurrency-safe: they keep no shared mutable state across
				// calls (run_javascript creates a fresh goja VM per invocation).
				ExecuteSequentially: false,
			},
		},
	}

	explorerAgent, err := adk.NewChatModelAgent(ctx, cfg)
	if err != nil {
		return "", err
	}

	// explorer/auditor are driven as MANAGED SESSIONS: each turn hands over
	// nothing but its new messages and the Runner replays everything older from
	// the session's event log (see run_session.go). The caller above therefore
	// never reassembles a transcript — no history concatenation, no tail
	// trimming, no tool-result reordering.
	sess := newRunSession()

	// EnableStreaming lives on RunnerConfig, not ChatModelAgentConfig.
	explorerHead := sess.explorer.head(ctx)
	iter := sess.explorer.runner(ctx, explorerAgent, in.Stream).Run(ctx, in.Messages)
	final, evidence, runErr := consumeAgentEvents(ctx, iter, in.OnDelta, in.ToolCallCounts, in.ToolCallErrors, in.ToolErrorSamples)
	if runErr != nil {
		// The aborted turn's events must not reach the next one: a tool call
		// whose result never arrived is a provider error on replay. This also
		// drops the caller's history from the session, so the first repair
		// turn has to carry it again (conversation.turnMessages).
		sess.explorer.discardFailedTurn(ctx, explorerHead)
	} else {
		sess.explorer.markSeeded()
	}

	// Delivery gate. The agent's job ended with its FINAL message; the gate
	// OWNS verification: every pass audits the standing final with
	// answer_auditor, ships on PASS, and on FAIL hands the audit findings back
	// to the agent as one repair directive. Nothing depends on the model
	// choosing to call the auditor.
	//
	// The gate runs even when the main loop aborted (runErr != nil): an aborted
	// run typically leaves a narration tail as the final, which is exactly the
	// answer-less output the gate exists to catch. Skipping it there handed
	// process narration straight to the user (observed on q55, where one
	// grep_chunks error aborted the loop right before the synthesis step).
	preGateFinal := final
	gateFinal, auditedFinal := runDeliveryGate(ctx, deliveryGateInput{
		explorer:       explorerAgent,
		auditor:        auditorAgent,
		sess:           sess,
		baseMessages:   in.Messages,
		final:          final,
		auditMaxPass:   auditMaxPass,
		toolCallCounts: in.ToolCallCounts,
		audit:          in.GateAudit,
	})
	if runErr != nil && gateFinal != preGateFinal {
		runErr = nil
	}
	final = gateFinal
	if finalAnswerValue(final) == "" {
		// Recovery ladder, cheapest and most faithful first. A run that ends
		// on a narration tail or on a bare tool call still holds an answer
		// somewhere in its own turns, and that answer is the model's own
		// deliverable — complete with the Candidate Matrix and the citations
		// the auditor checks. Shipping it beats a synthesis call, which
		// discards that structure and, once a failover cooldown has set in,
		// may not even reach a provider (it would be short-circuited with the
		// stale error that tripped the cooldown).
		carried := sess.explorer.lastAssistant(ctx, func(m *schema.Message) bool {
			return strings.TrimSpace(m.Content) != "" && finalAnswerValue(m.Content) != ""
		})
		switch {
		case hasFOSStructure(auditedFinal):
			// The deliverable the auditor last examined first: the gate has
			// been iterating on it, so it is the most refined FOS the run
			// produced, and it is the one the auditor's findings describe.
			final = auditedFinal
		case carried != "":
			final = carried
		default:
			final = sess.explorer.lastAssistant(ctx, func(m *schema.Message) bool {
				return strings.TrimSpace(m.Content) != ""
			})
		}
		if final != "" {
			common.WarnCtx(ctx, "agentic_rag: final carried no answer line, recovered an earlier deliverable",
				zap.Int("recovered_bytes", len(final)))
		}
	}
	if runErr != nil && final != preGateFinal {
		// The gate adopted a fresh substantive continuation, so the main
		// loop's error no longer describes the answer being returned.
		runErr = nil
	}
	// Label governance (lever 1): `Final Answer` claims every discriminating
	// constraint is corpus-verified, so it may only survive an audit that
	// PASSED. When the gate stopped on a stall, a budget or the clock, the
	// claim is demoted to `Guessed Answer` - the value is untouched (that is
	// what the judge scores), only the confidence claim is corrected.
	if in.GateAudit != nil && len(in.GateAudit.Suspects) > 0 && !in.GateAudit.Passed {
		if demoted, changed := demoteFinalAnswerLabel(final, "the delivery gate did not conclude PASS, so this value is not fully corpus-verified"); changed {
			common.InfoCtx(ctx, "agentic_rag: demoted Final Answer to Guessed Answer (audit did not pass)")
			final = demoted
		}
	}

	// Terminating action: the gate loop has spent its budget (or every
	// continuation failed) and the answer still carries no machine-parseable
	// FOS deliverable. Synthesize one deterministically from the question,
	// whatever partial output exists, and the evidence gathered so far — the
	// turn must never end on narration or a blank.
	if final == "" || runErr != nil || finalAnswerValue(final) == "" {
		// Detach from the run's cancellation entirely: the fallback fires most
		// often BECAUSE the shared wall-clock budget expired, and a deadline
		// expiry and a client hang-up are indistinguishable to the derived
		// context (both close the same Done channel) — keeping cancellation
		// would reintroduce the failure this call exists to survive. The
		// unbounded-work risk that WithoutCancel normally carries is covered
		// by the bounded budget attached here.
		synthCtx, synthCancel := context.WithTimeout(context.WithoutCancel(ctx), finalizeTimeout)
		defer synthCancel()
		synthModel := in.Model
		if in.SynthModel != nil {
			synthModel = in.SynthModel
		}
		synth, synthErr := finalizeAnswer(synthCtx, synthModel, in.Messages, final, evidence)
		if synthErr != nil {
			// Without this an empty final is completely silent: the run looks
			// healthy in every other log line and the user simply gets nothing.
			common.WarnCtx(ctx, "agentic_rag: finalizeAnswer failed", zap.Error(synthErr))
		} else if strings.TrimSpace(synth) == "" {
			common.WarnCtx(ctx, "agentic_rag: finalizeAnswer returned an empty answer")
		}
		switch {
		case synthErr == nil && synth != "" && finalAnswerLineCount(synth) == 1:
			// Replace rather than append: appending leaves the reader with a
			// narration paragraph followed by the real answer. The single
			// answer emission at the end of Run carries it.
			final = synth
			runErr = nil
		case synthErr == nil && synth != "":
			// The synthesis DEGENERATED — zero or multiple FOS answer lines
			// (observed: the model re-rendered the whole deliverable inside a
			// `Final Answer: **## ...**` mega-value, shipping two Final
			// Answer lines). The pre-synthesis deliverable is the model's own
			// structured work and ships instead of the blob.
			common.WarnCtx(ctx, "agentic_rag: finalizeAnswer degenerated, keeping the standing deliverable",
				zap.Int("synth_bytes", len(synth)),
				zap.Int("answer_lines", finalAnswerLineCount(synth)))
			if final != "" {
				runErr = nil
			}
		case runErr == nil:
			// Normal termination but synthesis failed — surface the reason.
			runErr = synthErr
		}
	}

	// Final label check: whatever text won (gate deliverable, synthesized or
	// recovered message) must agree with itself — a `## Final Answer` heading
	// above a `**Guessed Answer: X**` value line claims more than the run
	// verified. The conservative label wins; the value is untouched.
	if reconciled := reconcileAnswerLabels(final); reconciled != final {
		common.InfoCtx(ctx, "agentic_rag: reconciled answer labels (heading disagreed with the value line)")
		final = reconciled
	}

	// The answer channel carries exactly one thing: the deliverable this run
	// is shipping, emitted once, now that the gate and the fallbacks have
	// settled on it. Everything the loop said on the way there already went
	// out live as thinking, so the user watched the work without being handed
	// several competing "Final Answer" blocks.
	emit(ctx, in.OnDelta, final, "")
	return final, runErr
}

// gateNoProgressLimit bounds CONSECUTIVE repair attempts that fail to advance
// the deliverable - a continuation that aborted on a tool error, went silent,
// or ended in narration without the FOS answer line. It is the gate's single
// "this is not working" budget, and such attempts are retried IN PLACE: they
// leave the deliverable unchanged, so re-auditing first would only reproduce
// the verdict already in hand (a re-audit is only worth running after the
// final has actually changed). Reaching the limit means the model cannot
// render or the tool layer is down; either way the gate gives up and falls
// through to finalizeAnswer.
//
// It is NOT redundant with the suspect-count stall check below, and removing
// it hangs the gate: `pass` counts only ADOPTED repairs, so a model that keeps
// emitting unusable continuations never advances the loop and the audit ceiling
// can never be reached; and since those attempts are retried in place they produce
// no verdict, so the suspect trend has nothing to observe. This counter is the
// only exit that does not wait for the run's deadline. Measured across the
// frames runs: 12 give-ups here against 5 stall cut-offs — disjoint paths.
const gateNoProgressLimit = 3

// gateStallWindow is how many consecutive audits are weighed to decide
// whether the suspect count is still falling (see the stall check in
// runDeliveryGate). With the last three counts f1, f2, f3 (oldest first), the
// repair is still working only while f3 is below f1 or f2 - the newest audit
// found fewer problems than at least one of the two before it. Once f3 sits at
// or above BOTH, the count has stopped falling (it may even be climbing back)
// and the loop stops.
//
// Three is the smallest window that can tell a trend from one noisier verdict.
// This replaces an earlier rule that fired only when three counts were exactly
// EQUAL and gave low counts 15 passes of slack: that slack was earned by q667
// (13 passes at 2 suspects, then it cleared and scored 4), but once the
// candidate-matrix audit was removed the failure shape changed - runs froze at
// 1 suspect instead (q371 and q629 burned 18 passes each, q646 17, q481 and
// q617 the full 20): 112 passes across 13 questions for five stalls of which
// only one ever broke. A flat trend test cuts those by pass 7 while a
// genuinely falling count still runs to the ceiling.
const gateStallWindow = 3

// deliveryGateInput carries the gate loop's dependencies.
type deliveryGateInput struct {
	// explorer is the research agent itself; every repair turn runs it inside
	// the managed session below, so the gate sends no history.
	explorer adk.Agent
	// auditor is the answer_auditor sub-agent, run BY THE GATE on every pass
	// inside the managed session below — the agent never sees it in its
	// toolset.
	auditor adk.Agent
	// sess owns the two managed conversations: the explorer's (research +
	// repair turns) and the auditor's continuous audit conversation. Each turn
	// hands over only its NEW messages; the Runner replays the rest.
	sess *runSession
	// toolCallCounts is the run's shared per-question tally (see
	// Input.ToolCallCounts). Repair turns and auditor passes append to the
	// SAME ledger so per-question usage accounting covers the whole turn.
	toolCallCounts map[string]int
	// audit, when non-nil, receives the per-round suspect accounting of every
	// audit this gate runs (see GateAuditRecord).
	audit *GateAuditRecord
	// baseMessages is the caller's chat history (ends on the current user
	// question); every repair turn is built on top of a fresh copy.
	baseMessages []*schema.Message
	final        string
	// auditMaxPass is the template's audit budget (Template.AuditMaxPass):
	// how many audit passes the gate may spend. Zero or less ships the
	// deliverable unaudited — the gate is not entered at all.
	auditMaxPass int
}

// runDeliveryGate drives the audit loop and returns the adopted final answer
// plus the deliverable the auditor last examined (auditedFinal) — the
// caller's fallback candidate when the loop exits without the FOS answer
// line. The agent's job ended with its FINAL message; the gate OWNS
// verification: every pass audits the standing final with answer_auditor,
// ships on PASS, and on FAIL hands the audit findings back to the agent as
// one repair directive (full toolset, so repairs can list_chunks-read the
// flagged steps). Nothing depends on the model choosing to call the auditor
// — the audit never runs inside the agent, so compliance cannot be defied.
func runDeliveryGate(ctx context.Context, in deliveryGateInput) (string, string) {
	if in.auditMaxPass <= 0 {
		return in.final, ""
	}
	final := in.final
	// auditedFinal: the deliverable the auditor last examined or was about to
	// examine (updated at the top of every pass). On any exit it is the
	// fallback the caller may ship verbatim when the standing final lacks the
	// FOS answer line — an audit-failed but structured deliverable beats blind
	// synthesis (see the finalizeAnswer gate in Run).
	auditedFinal := final
	// The auditor's OWN conversation, threaded across passes: user payload ->
	// (tool-call turns + verdict) -> next payload -> ... One continuous audit
	// session means the auditor remembers its earlier opinions and the chunks
	// it already deep-read — a re-audit focuses on what changed instead of
	// re-reading everything. The session (not the caller) owns that history.
	if in.sess == nil {
		in.sess = newRunSession()
	}
	// noProgress: consecutive repair attempts that failed to advance the
	// deliverable (see gateNoProgressLimit).
	noProgress := 0
	// suspectHist is the suspect count of EVERY audit this gate runs, oldest
	// first — the whole curve, not just the window that decides the stall. It
	// is bounded by the audit budget (auditMaxPass ints at most), so keeping it
	// costs nothing and the short-circuit log can show how the repair actually
	// behaved: a count that fell 15→11→6→5 before freezing reads very
	// differently from one that never moved. The stall check reads its tail.
	suspectHist := make([]int, 0, in.auditMaxPass)
	// Everything the run's tools actually returned: the grounding check below
	// tests the shipped value against it, so it is collected once.
	groundingHaystack := in.sess.explorer.toolResultText(ctx)
	// repairFeedback: why the last attempt was discarded, appended to the
	// next attempt's directive — a silently discarded attempt leaves the
	// model repeating the same narration (observed on q716).
	repairFeedback := ""

	// pass counts AUDITS, and only an adopted continuation increments it: a
	// re-audit is worth running solely when the deliverable under it changed.
	for pass := 0; pass < in.auditMaxPass; {
		// Deadline awareness (q268 lesson): every gate step costs minutes and
		// the run shares ONE wall-clock budget with the main loop. When it is
		// gone, auditing or repairing on a dead context can only fail — and
		// every in-flight tool call dies mid-flight — so stop and ship the
		// best deliverable in hand. The caller's finalizeAnswer fallback still
		// runs, on a context of its own.
		if err := ctx.Err(); err != nil {
			common.WarnCtx(ctx, "agentic_rag: delivery gate out of time",
				zap.Int("pass", pass+1), zap.Error(err))
			return final, auditedFinal
		}
		// A blank deliverable cannot produce anything but the auditor's
		// mechanical first opinion — `schema integrity: final_message is
		// missing` — so auditing it burns a call and delays the repair that is
		// obviously needed. Skip straight to the directive; the next pass
		// audits as soon as a deliverable exists.
		verdict := ""
		directive := ""
		if shipped := finalAnswerValue(final); strings.TrimSpace(final) != "" && shipped != "" && !answerValueIsGrounded(shipped, groundingHaystack) {
			// GROUNDING CHECK (lever 2): a named value that occurs in NO
			// chunk the run read cannot be corpus-supported - it was
			// synthesized. Send the agent back with a new anchor instead of
			// auditing a claim that has no evidence behind it.
			common.InfoCtx(ctx, "agentic_rag: delivery gate rejected an ungrounded answer value",
				zap.Int("pass", pass+1), zap.String("value", shipped))
			directive = ("The value on your answer line (" + shipped + ") appears in NO chunk you have read: nothing " +
				"you retrieved contains it, so it cannot be a corpus-supported answer. Do NOT re-render the same matrix. " +
				"Run at least one NEW retrieval with a different anchor first - the rarest proper noun of the question " +
				"queried ALONE, or a grep_chunks co-occurrence regex over two clue terms - then rebuild the Candidate " +
				"Matrix from what actually surfaces. If nothing supports any candidate, ship `Guessed Answer: **<value>** " +
				"(assumption: ...)` naming what is unverified, or state the insufficiency explicitly.")
		} else if strings.TrimSpace(final) != "" && hasFOSStructure(final) && finalAnswerValue(final) == "" {
			// A deliverable that carries no answer VALUE is not shippable:
			// the FOS contract requires `Final Answer: **<value>**` (or the
			// Guessed variant). Shipping one silently downgrades the run - the
			// caller's recovery ladder then adopts some earlier message and
			// the judge ends up reading a stray entity out of the prose
			// (q1228: bare `## Final Answer` heading, judge extracted a
			// different film from the body). Repair instead of shipping.
			common.InfoCtx(ctx, "agentic_rag: delivery gate rejected a value-less answer line",
				zap.Int("pass", pass+1))
			directive = ("Your FINAL message has a Final/Guessed Answer heading that carries NO value. " +
				"Restate it on ONE line in the exact shape `Final Answer: **<value>**` " +
				"(or `Guessed Answer: **<value>** (assumption: ...)` when the value rests on an assumption), " +
				"where <value> is the single named answer - one entity/title/date, never a sentence, never empty. " +
				"Re-render the COMPLETE FOS FINAL message (## Candidate Matrix, ## Reasoning Chain, then that line).")
		} else if strings.TrimSpace(final) != "" {
			// The gate audits the deliverable it actually holds — audit-target
			// freshness is structural, not tracked. The question lives in the
			// auditor's system prompt, so the payload carries the deliverable
			// only.
			auditedFinal = final
			var err error
			verdict, err = gateRunAudit(ctx, in.auditor, in.sess.auditor, final, in.toolCallCounts)
			if err != nil {
				// The auditor itself failed (LLM timeout, tool outage).
				// Retrying inside this request rarely helps; fall through to
				// finalizeAnswer, which still catches answer-less finals.
				common.WarnCtx(ctx, "agentic_rag: delivery gate audit failed",
					zap.Int("pass", pass+1), zap.Error(err))
				return final, auditedFinal
			}
			// Per-round suspect accounting for benchmarks (Input.GateAudit):
			// every audit is recorded, PASS rounds included — a PASS verdict
			// reports 0 suspects, so the list doubles as the round count.
			if in.audit != nil {
				in.audit.Suspects = append(in.audit.Suspects, auditSuspectCount(verdict))
				in.audit.Passed = auditPassed(verdict)
			}
			if auditPassed(verdict) {
				break // audited and passed as a whole — ship
			}

			common.InfoCtx(ctx, "agentic_rag: delivery gate audit failed the deliverable",
				zap.Int("pass", pass+1),
				zap.Int("suspects", auditSuspectCount(verdict)))

			suspectHist = append(suspectHist, auditSuspectCount(verdict))
			// Stall check fires BEFORE the repair turn: once three observations
			// are in, stop as soon as the newest count is no better than BOTH of
			// the two before it. Equal counts are only the special case - this
			// also catches a count climbing back (f1 < f2 < f3), where the
			// repair is making the deliverable worse and shipping what stands
			// strictly beats buying another pass. A count that is still falling
			// keeps the loop alive. The trend is read off the tail of the full
			// history.
			if n := len(suspectHist); n >= gateStallWindow {
				f1, f2, f3 := suspectHist[n-3], suspectHist[n-2], suspectHist[n-1]
				if f3 >= f1 && f3 >= f2 {
					common.InfoCtx(ctx, "agentic_rag: delivery gate short-circuit — suspect count stopped falling",
						zap.Int("pass", pass+1), zap.Ints("suspects", suspectHist))
					break
				}
			}
		}

		if directive == "" && strings.TrimSpace(final) == "" {
			// No deliverable: this repair is not a correction, it is the FIRST
			// production of the answer.
			common.InfoCtx(ctx, "agentic_rag: delivery gate skipped the audit — no deliverable yet",
				zap.Int("pass", pass+1))
			directive = ("You have produced NO deliverable yet, so there is nothing to ship to the user. " +
				"Continue the investigation in this turn and render the COMPLETE FOS FINAL message: the " +
				"`## Candidate Matrix` blocks, then `## Reasoning Chain`, and the LAST line " +
				"`Final Answer: **<value>**` (or `Guessed Answer: **<value>** (assumption: ...)`). " +
				"If a retrieval call errored or came back empty, rephrase the query and try again BEFORE " +
				"answering: an answer drawn from memory instead of the corpus is not acceptable, and " +
				"ending the turn on narration or on another tool call leaves the user with nothing.")
		} else {
			directive = fmt.Sprintf("Your FINAL message failed the pipeline's mandatory answer auditor "+
				"(%d suspect item(s)):\n%s\nFix every item whose audit opinion is not `pass` before shipping: "+
				"keep retrieving or list_chunks-read the flagged line, repair or drop it, re-derive the answer "+
				"from the supported steps, and re-render the COMPLETE FOS FINAL message (## Candidate Matrix, "+
				"## Reasoning Chain, and the Final/Guessed Answer line). If the audit reported schema integrity "+
				"(missing candidate_matrix / reasoning_chain / answer), render the full deliverable now. "+
				"If the audit reported `retrieval breadth insufficient`, do NOT re-render the same matrix: "+
				"decompose the question into its rarest distinctive terms (proper nouns, unique titles, exact "+
				"dates) and run one SHORT keyword query per term - long multi-clue paraphrases do not match "+
				"the corpus's wording. "+
				// Lever 3: a repair that only re-renders the matrix is the
				// observed failure shape - the suspect count sits flat or
				// climbs because the anchor never moved. Once a repair has
				// already failed, require NEW evidence before any re-render.
				anchorDemand(len(suspectHist), suspectHist)+
				"Citation-field repairs: call list_chunks for the cited chunk_id and OVERWRITE the line's "+
				"fields with what the tool actually returns - never guess an identifier from memory; a value "+
				"you cannot look up must be dropped, not invented. Never leave an item whose opinion is not "+
				"`pass` in the deliverable.", auditSuspectCount(verdict), verdict)
		}
		// Repair loop: retry IN PLACE until the deliverable ADVANCES. An
		// attempt that aborted, went silent, or ended in narration leaves the
		// deliverable unchanged, so re-auditing it would only reproduce the
		// verdict in hand — the retry is cheaper and carries the reason the
		// previous attempt was discarded.
		giveUp := false
		for {
			attempt := directive
			if repairFeedback != "" {
				attempt += "\n\n" + repairFeedback
			}
			if err := ctx.Err(); err != nil {
				// Budget gone: a repair turn would only abort mid-flight, and
				// the in-place retry budget would burn on instant failures.
				common.WarnCtx(ctx, "agentic_rag: delivery gate out of time before repair",
					zap.Int("pass", pass+1), zap.Error(err))
				giveUp = true
				break
			}
			gateFinal, gateErr := runRepairAttempt(ctx, in, in.sess.explorer, final, attempt)
			if gateErr == nil {
				// The turn stays in the session even when it will be
				// rejected: a narrating attempt may still have retrieved
				// useful evidence, and the session replays it — reordering
				// parallel tool results and nothing else — provider-valid.
				trimmed := strings.TrimSpace(gateFinal)
				// Adoption rule (format presence, not correctness): a
				// continuation must carry BOTH the FOS answer line and the
				// FOS sections. The answer line alone is not enough — a bare
				// "Guessed Answer: I do not have sufficient evidence" line
				// satisfies it while dropping every cited chunk, so adopting
				// it throws away a structured (if failing) deliverable and
				// buys a re-audit that can only repeat the coverage defect
				// (q1093 burned 16 audit passes that way).
				if finalAnswerValue(trimmed) != "" && hasFOSStructure(trimmed) {
					final = trimmed // deliverable-shaped continuation → adopt
					noProgress = 0
					repairFeedback = ""
					pass++ // the final changed: the next audit is worth running
					break
				}
				repairFeedback = ("REJECTION NOTICE: your previous repair continuation was DISCARDED - it " +
					"did not render the COMPLETE deliverable (it was missing the FOS answer line, or the " +
					"`## Candidate Matrix` / `## Reasoning Chain` sections, or both). Your next reply must " +
					"render it in the SAME turn: the `## Candidate Matrix` blocks, then " +
					"`## Reasoning Chain`, and the LAST line `Final Answer: **<value>**` or " +
					"`Guessed Answer: **<value>** (assumption: ...)` - a bare answer line or any narration " +
					"instead of the deliverable is discarded again.")
				common.InfoCtx(ctx, "agentic_rag: delivery gate rejected answer-less continuation",
					zap.Int("pass", pass+1),
					// Preview of what was discarded: without it there is no way
					// to tell a quota-truncated repair from one that simply never
					// rendered the FOS deliverable (q798-class diagnosis).
					zap.String("preview", truncateForLog(trimmed, 200)))
			} else {
				repairFeedback = fmt.Sprintf(("TOOL FAILURE NOTICE: your previous repair turn aborted with " +
					"a tool error (%v) and produced nothing usable. The audit findings above still stand - " +
					"retry the same repair, but avoid the failing call: if a retrieval tool errored, use a " +
					"different tool or a different query."), gateErr)
				common.WarnCtx(ctx, "agentic_rag: delivery gate repair turn aborted",
					zap.Int("pass", pass+1), zap.Error(gateErr))
			}

			// A repair attempt that aborted because the shared budget expired
			// will fail identically on every retry: stop burning the
			// no-progress allowance and ship.
			if err := ctx.Err(); err != nil {
				common.WarnCtx(ctx, "agentic_rag: delivery gate out of time after repair",
					zap.Int("pass", pass+1), zap.Error(err))
				giveUp = true
				break
			}

			noProgress++
			if noProgress >= gateNoProgressLimit {
				common.InfoCtx(ctx, "agentic_rag: delivery gate short-circuit — no progress across attempts",
					zap.Int("pass", pass+1), zap.Int("attempts", noProgress))
				giveUp = true
				break
			}
		}
		if giveUp {
			break
		}
		// Next pass re-audits whatever final now stands.
	}

	return final, auditedFinal
}

// runRepairAttempt performs ONE explorer continuation for a repair directive,
// returning the continuation's FINAL message. Retry policy lives in the gate
// loop, which retries in place while the deliverable has not advanced.
//
// The turn carries NO history: the managed session replays it, so the only
// thing handed over is the directive — plus the caller's chat history when an
// earlier failure forced the conversation to restart (see conversation.turnMessages).
func runRepairAttempt(
	ctx context.Context,
	in deliveryGateInput,
	conv *conversation,
	final, directive string,
) (string, error) {
	gateMsgs := conv.turnMessages(in.baseMessages, final, directive)
	// Deltas stay off for gate passes: clients already hold the first
	// answer, and only the corrected final is returned upward.
	noStream := func(contentDelta, thinkingDelta string) {}
	head := conv.head(ctx)
	iter := conv.runner(ctx, in.explorer, false).Run(ctx, gateMsgs)
	// Tally repair-turn tool calls into the run's shared ledger: the audit
	// directive routinely asks for list_chunks re-reads, and dropping those
	// from the counts made tool_<name> disagree with tool_<name>_ms.
	out, _, err := consumeAgentEvents(ctx, iter, noStream, in.toolCallCounts, nil, nil)
	if err != nil {
		// A failure must not leak a half-finished turn into the next one:
		// an unanswered tool call replays as a provider error.
		conv.discardFailedTurn(ctx, head)
	}
	return out, err
}

// anchorDemand returns the "change the anchor before re-rendering" clause of a
// repair directive. It binds only once a repair has ALREADY failed (two or more
// suspect observations): re-rendering the same matrix is the shape the gate
// keeps seeing when the answer is wrong - the audit count sits flat or climbs
// because the retrieval anchor never moved, so the repair buys audit rounds
// without buying evidence. From the second failed audit on, a repair turn must
// open with new retrieval, and a turn with no new `Searched` line is rejected.
func anchorDemand(failures int, hist []int) string {
	if failures < 2 {
		return ""
	}
	trend := ""
	if n := len(hist); n >= 2 {
		trend = fmt.Sprintf(" The suspect count moved %s over your failed repairs, which is the signature of re-rendering the same anchor.",
			joinInts(hist))
	}
	return ("ANCHOR CHANGE REQUIRED: this is failed audit #" + fmt.Sprint(failures) + "." + trend +
		" Your repair MUST open with at least one NEW retrieval call on a DIFFERENT anchor than the ones already in the matrix " +
		"(the rarest proper noun queried alone, or a grep_chunks co-occurrence regex over two clue terms) and log it as a new " +
		"`Searched` line. A repair turn that re-renders the matrix with no new `Searched` line is rejected outright.")
}

func joinInts(values []int) string {
	parts := make([]string, 0, len(values))
	for _, v := range values {
		parts = append(parts, fmt.Sprint(v))
	}
	return strings.Join(parts, " -> ")
}

// gateRunAudit runs one audit pass of the answer auditor on the current
// deliverable, extending the auditor's continuous conversation: the managed
// session replays every earlier audit pass, so this turn hands over nothing
// but ONE new user message carrying the JSON payload, then produces the
// auditor's ReAct turns (list_chunks deep-reads + the echoed deliverable with
// audit opinions). It returns the verdict text.
//
// A failed audit leaves nothing behind: its events are rolled out of the
// session, so the next pass replays exactly what the last successful one saw.
func gateRunAudit(
	ctx context.Context,
	auditor adk.Agent,
	conv *conversation,
	final string,
	toolCallCounts map[string]int,
) (string, error) {
	if auditor == nil {
		return "", fmt.Errorf("agentic_rag: auditor unavailable")
	}
	payload := buildAuditPayload(final)
	common.WarnCtx(ctx, "agentic_rag: gate-run audit start",
		zap.String("payload", truncateForLog(payload, 2000)))
	head := conv.head(ctx)
	iter := conv.runner(ctx, auditor, false).Run(ctx, []adk.Message{schema.UserMessage(payload)})
	verdict, _, err := consumeAgentEvents(ctx, iter, func(string, string) {}, toolCallCounts, nil, nil)
	if err != nil {
		conv.discardFailedTurn(ctx, head)
		return "", err
	}
	common.InfoCtx(ctx, "agentic_rag: gate-run audit verdict",
		zap.String("verdict", truncateForLog(verdict, 2000)))
	return verdict, nil
}

// consumeAgentEvents drains the agent event iterator, streaming assistant
// deltas through onDelta (falling back to runtime.EmitAgentMessage when nil)
// and returning the agent's FINAL message plus an evidence digest. A ReAct
// loop produces one assistant message per turn — tool-call turns interleaved
// with the last, tool-free answer turn — and the FOS deliverable is exactly
// that LAST message, so `final` REPLACES the previous content on every
// completed assistant message instead of concatenating across turns.
//
// The transcript this used to return is gone: the conversation is the session's
// business now (see run_session.go). `evidence` still accumulates tool output
// because the last-resort synthesis needs a digest, not a message list.
// When toolCallCounts is non-nil, each tool the model requested is tallied by
// name (one tally per requested call, which matches one actual tool
// invocation).
func consumeAgentEvents(
	ctx context.Context,
	iter *adk.AsyncIterator[*adk.AgentEvent],
	onDelta func(contentDelta, thinkingDelta string),
	toolCallCounts map[string]int,
	toolCallErrors map[string]int,
	toolErrorSamples map[string]string,
) (string, string, error) {
	var final string
	var evidence strings.Builder
	// toolNames maps tool_call_id -> tool name. Tool-result events carry only
	// the call id, so the name is captured here when the model issues the call
	// and joined back when the result arrives.
	toolNames := make(map[string]string)
	// Everything the agent says goes out LIVE, but on the THINKING channel.
	// A ReAct turn emits many messages and any of them can already be a
	// complete deliverable (Candidate Matrix + Reasoning Chain + Final Answer)
	// — the loop often renders one, keeps retrieving, and renders a better one.
	// Routing those to the answer channel is what put several competing "Final
	// Answer" blocks in front of the user. On the thinking channel they stay
	// readable as the work unfolds without competing with the answer, which is
	// emitted exactly once, at the end of the run, by Run itself (the only
	// place that knows which deliverable is actually being shipped — the gate
	// may adopt a different one).
	live := func(content, reasoning string) {
		emit(ctx, onDelta, "", content+reasoning)
	}
	for {
		ev, ok := iter.Next()
		if !ok {
			break
		}
		if ev.Err != nil {
			return final, evidence.String(), ev.Err
		}
		if ev.Output == nil || ev.Output.MessageOutput == nil {
			continue
		}
		mo := ev.Output.MessageOutput
		if mo.Role != schema.Assistant {
			// Tool-result events and non-assistant messages carry the tool's
			// returned content; log it and accumulate a truncated summary for
			// the last-resort synthesis.
			if mo.Message != nil {
				content := mo.Message.Content
				common.DebugCtx(ctx, "agentic_rag: tool result",
					zap.String("tool", toolNames[mo.Message.ToolCallID]),
					zap.String("tool_call_id", mo.Message.ToolCallID),
					zap.Int("content_bytes", len(content)),
					zap.String("content_head", truncateForLog(content, 2000)),
				)
				// Failure accounting: tools report failures as canonical
				// <tool_error> results so the loop keeps running; the
				// severity="error" ones are tallies here for the benchmark's
				// run stats (severity="warn" partials stay visible to the
				// model but don't count as outages). One sample per tool is
				// kept — the first root cause seen, truncated.
				if strings.Contains(content, toolErrorMarker) && strings.Contains(content, `severity="error"`) {
					if name := toolNames[mo.Message.ToolCallID]; name != "" && toolCallErrors != nil {
						toolCallErrors[name]++
						if toolErrorSamples != nil {
							if _, ok := toolErrorSamples[name]; !ok {
								toolErrorSamples[name] = truncateForLog(content, 300)
							}
						}
					}
				}
				if trimmed := strings.TrimSpace(content); trimmed != "" {
					if evidence.Len() > 0 {
						evidence.WriteString("\n\n")
					}
					evidence.WriteString(truncateForLog(trimmed, 4000))
				}
			}
			continue
		}
		// Log the tool calls the model decided to make this round, and tally
		// them by name into the caller's counter (when provided) so per-question
		// tool invocation counts can be aggregated.
		if mo.Message != nil && len(mo.Message.ToolCalls) > 0 {
			for i := range mo.Message.ToolCalls {
				tc := &mo.Message.ToolCalls[i]
				name := tc.Function.Name
				if tc.ID != "" {
					toolNames[tc.ID] = name
				}
				common.DebugCtx(ctx, "agentic_rag: tool call",
					zap.String("tool", name),
					zap.String("tool_call_id", tc.ID),
					zap.String("args", truncateForLog(tc.Function.Arguments, 2000)),
				)
				if toolCallCounts != nil && name != "" {
					toolCallCounts[name]++
				}
			}
		}
		if mo.IsStreaming {
			if mo.MessageStream == nil {
				continue
			}
			// One MessageOutput event carries one assistant message; its
			// completion REPLACES the running final (last-message semantics).
			// Chunks are merged back into one message — content join plus
			// tool-call delta assembly by Index — so a parallel search round
			// keeps every call it issued.
			var chunks []*schema.Message
			for {
				chunk, recvErr := mo.MessageStream.Recv()
				if recvErr != nil {
					if !errors.Is(recvErr, io.EOF) {
						// A real stream error: close the reader and surface it
						// instead of returning silently-truncated content.
						mo.MessageStream.Close()
						return final, evidence.String(), recvErr
					}
					break // io.EOF marks normal end of the stream.
				}
				live(chunk.Content, chunk.ReasoningContent)
				chunks = append(chunks, chunk)
			}
			mo.MessageStream.Close()
			final = mergeStreamedAssistant(chunks).Content
			continue
		}
		if mo.Message != nil {
			live(mo.Message.Content, mo.Message.ReasoningContent)
			final = mo.Message.Content
		}
	}
	return final, evidence.String(), nil
}

// mergeStreamedAssistant assembles one assistant message from its stream
// chunks: content is joined verbatim, and tool-call deltas are merged by
// their Index (name and arguments stream incrementally). A nil chunk or an
// all-empty stream yields an empty assistant message.
func mergeStreamedAssistant(chunks []*schema.Message) *schema.Message {
	out := &schema.Message{Role: schema.Assistant}
	callPos := make(map[int]int)
	for _, c := range chunks {
		if c == nil {
			continue
		}
		out.Content += c.Content
		for _, tc := range c.ToolCalls {
			// Index is a pointer in this eino version; treat nil as 0.
			idx := 0
			if tc.Index != nil {
				idx = *tc.Index
			}
			pos, ok := callPos[idx]
			if !ok {
				clone := tc
				out.ToolCalls = append(out.ToolCalls, clone)
				callPos[idx] = len(out.ToolCalls) - 1
				continue
			}
			out.ToolCalls[pos].Function.Name += tc.Function.Name
			out.ToolCalls[pos].Function.Arguments += tc.Function.Arguments
			if tc.ID != "" {
				out.ToolCalls[pos].ID = tc.ID
			}
		}
	}
	return out
}

// emit streams a (content, reasoning) delta through onDelta, falling back to
// runtime.EmitAgentMessage when onDelta is nil.
func emit(
	ctx context.Context,
	onDelta func(contentDelta, thinkingDelta string),
	content, reasoning string,
) {
	if onDelta != nil {
		onDelta(content, reasoning)
		return
	}
	runtime.EmitAgentMessage(ctx, content, reasoning)
}

// finalizeTimeout is the synthesis fallback's own budget. It runs on a context
// detached from the run's (see context.WithoutCancel at the call site): the
// fallback exists precisely for the case where the run's wall-clock deadline
// expired mid-investigation, so inheriting that deadline would guarantee its
// failure — q268 shipped a 60-char narration sentence because the final
// synthesis call was itself killed by the budget that had just run out.
const finalizeTimeout = 90 * time.Second

// finalizeEvidenceMaxRunes caps the evidence block handed to the synthesis
// call. Tool outputs accumulate 4KB each; without a cap a long investigation
// turns the prompt into a mega-message that makes thinking models deliberate
// for minutes and blow the client timeout (observed: exactly 300s on q55).
const finalizeEvidenceMaxRunes = 16000

// finalizeAnswer issues a single deterministic synthesis call so a turn that
// ended abnormally (empty final, or an error aborted the loop mid-investigation)
// still returns a grounded answer instead of a blank. It feeds the model the
// original user question, whatever partial assistant output was produced, and a
// truncated summary of the evidence retrieved so far, and instructs it to commit
// to the best answer it can — using the closest available data with an explicit
// assumption when exact figures are missing.
func finalizeAnswer(
	ctx context.Context,
	m *models.EinoChatModel,
	messages []*schema.Message,
	partial, evidence string,
) (string, error) {
	question := lastUserQuestion(messages)
	body := new(strings.Builder)
	fmt.Fprintf(body, "You were asked:\n%s\n\n", question)
	structured := hasFOSStructure(partial)
	if strings.TrimSpace(partial) != "" {
		fmt.Fprintf(body, "The research so far produced this partial output:\n%s\n\n", partial)
	}
	if ev := strings.TrimSpace(evidence); ev != "" {
		fmt.Fprintf(body, "Evidence gathered during the search:\n%s\n\n", truncateRunes(ev, finalizeEvidenceMaxRunes))
	}
	if structured {
		// The partial output IS the audited deliverable (Candidate Matrix /
		// Reasoning Chain) minus the FOS answer line. Re-render it whole:
		// collapsing it to a bare answer throws away the corpus-grounded
		// reasoning chain and chunk citations (q71 shipped a 3-char "EF5"
		// that way — correct, but with no trace of how it was derived).
		body.WriteString("The partial output above already contains the deliverable's structure " +
			"(`## Candidate Matrix` and/or `## Reasoning Chain`). Re-render the COMPLETE final " +
			"deliverable now: keep every section and chunk citation the evidence supports, " +
			"repair or drop the lines it does not, and end with the LAST line " +
			"`Final Answer: **<value>**` (or `Guessed Answer: **<value>** (assumption: ...)`). " +
			"Do not request any tools and do not hedge with \"I would need\" or \"I could not find\".")
	} else {
		body.WriteString("You have reached the end of your investigation. Based strictly on the evidence " +
			"above, produce your best final answer to the original question now, as plain text. " +
			"Reply with the final answer ONLY — no reasoning process, no plan, no narration. " +
			"If the exact data is not available, use the closest available figure, state the " +
			"assumption explicitly, and still provide the computed answer. Do not request any tools " +
			"and do not hedge with \"I would need\" or \"I could not find\".")
	}

	resp, err := m.Generate(ctx, []*schema.Message{{
		Role:    schema.User,
		Content: body.String(),
	}})
	if err != nil {
		return "", err
	}
	if resp == nil || resp.Content == "" {
		return "", errors.New("agentic_rag: finalizeAnswer produced empty output")
	}
	synth := strings.TrimSpace(resp.Content)
	if structured && !hasFOSStructure(synth) {
		// Safety net: the re-render dropped the structure. Graft the bare
		// answer onto the audited partial instead of shipping it alone, so
		// the reasoning chain survives even a non-compliant synthesis call.
		if finalAnswerValue(synth) != "" {
			return strings.TrimSpace(partial) + "\n\n" + synth + "\n", nil
		}
		return strings.TrimSpace(partial) + "\n\nFinal Answer: **" + synth + "**\n", nil
	}
	return synth, nil
}

// lastUserQuestion returns the content of the last user message, used as the
// question for the final-answer fallback and for per-question usage logs.
func lastUserQuestion(messages []*schema.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i] != nil && messages[i].Role == schema.User {
			return messages[i].Content
		}
	}
	return ""
}

// truncateForLog caps a string for log lines so a huge tool output or argument
// payload cannot blow up the log volume.
func truncateForLog(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// truncateRunes caps a string to max runes (not bytes) so CJK content is never
// cut mid-rune into invalid UTF-8.
func truncateRunes(s string, max int) string {
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max]) + "..."
}
