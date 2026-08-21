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
	"os"
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
	// AuditModel, when non-nil, is the model the delivery gate's auditor runs
	// on. It must be a SEPARATE instance over the same failover chain as Model
	// — and, unlike Model, one whose sampling is pinned (see
	// AuditTemperature) — because the auditor is not a second producer: its
	// verdict is machine-parsed and decides whether the deliverable ships, so
	// it must not inherit the producer's sampling noise, and a failover
	// instance's cached last error must not flow between the two. Nil falls
	// back to Model.
	AuditModel *models.EinoChatModel
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
	// Rejections counts the deliverables the gate refused BEFORE an audit ran —
	// an ungrounded value, a list-only name, a value-less answer line, an
	// answer-less continuation. Without it, a run whose gate never reached the
	// auditor (both q350 and q784 short-circuited this way) carried an EMPTY
	// Suspects on a never-PASSed deliverable, which read as "nothing to report"
	// and let a `Final Answer` label survive unaudited.
	Rejections int `json:"rejections,omitempty"`
	// AuditFailures counts audit passes the auditor itself could not complete
	// (LLM timeout, tool outage). It is the third way "the gate never concluded
	// PASS" becomes true, and the only one that records nothing else: the gate
	// returns the standing deliverable on an audit error, so without this count
	// the record stayed EMPTY on a never-audited deliverable and the label
	// governance read it as "no audit was needed" — shipping an unverified
	// `Final Answer`. It doubles as an operator signal: a per-question audit
	// outage is a provider problem, not a reasoning one.
	AuditFailures int `json:"audit_failures,omitempty"`
	// CitationGroundings counts the deliverables the gate rejected because their
	// candidate lines rested on a BIBLIOGRAPHIC ENTRY (a citation names a work and
	// states nothing about it). It is the mechanical half of the sibling discipline:
	// the auditor is ASKED to widen its read, but whether a round does it is up to
	// the model, and 10 of 17 measured q221 runs shipped the wrong member of a
	// two-book family — every one of them "grounded" by a citation line. Recorded
	// separately from Rejections so a benchmark can tell a citation-only matrix from
	// a missing-value one.
	CitationGroundings int `json:"citation_groundings,omitempty"`
	// AuditVerdicts holds an excerpt of EVERY audit verdict, oldest first, in
	// step with Suspects. The counts alone say a curve moved 5→3→1→0 but never
	// WHAT was contested, so a failure could only be explained by re-reading the
	// shipped deliverable and guessing — which is how the #221 analysis had to
	// proceed, and how it managed to infer the wrong cause twice. One excerpt
	// per round makes the audit's own words the record.
	AuditVerdicts []string `json:"audit_verdicts,omitempty"`
}

// defaultMaxIterations caps the ReAct loop before the agent must answer. It is
// raised beyond the original 50 because multi-step benchmark questions (multiple
// hops, off-by-one arithmetic) routinely need more than 50 model turns, and the
// duplicate-retrieval guard (todo 3) prevents the extra budget from being wasted
// on repeated identical searches.
const defaultMaxIterations = 120

var errNilModel = errors.New("agentic_rag: model is required")

// auditModelFor is the model the auditor runs on: its own instance when the
// caller built one (pinned sampling, separate failover state), else the
// producer's. A named rule rather than an inline nil check, so the fallback
// stays testable.
func auditModelFor(in Input) *models.EinoChatModel {
	if in.AuditModel != nil {
		return in.AuditModel
	}
	return in.Model
}

// llmRetryMax bounds retry attempts per model call on top of the initial one
// (adk semantics: MaxRetries=1 → up to 2 calls). With the backoff below the
// worst-case added latency per call is ~2s.
//
// Lowered from 3 to 1 on 2026-09-17 for the slow-plan experiment: a slow
// endpoint that fails at the send phase fails the same way on every attempt,
// so the extra attempts only bought latency (and, with the old 300s per-call
// budget, ran whole questions into the wall-clock deadline).
const llmRetryMax = 1

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
	// deepIDs / shallowIDs record WHICH chunks the two counts refer to.
	//
	// The counts alone cannot answer the question a failed run actually poses:
	// did the passage that holds the answer ever reach the model? Measured on
	// the browsecomp retry batch, 13 of the 20 evidence_in_hand failures had a
	// gold-bearing document served to the run, only 6 of those cited it and 0
	// shipped it — the difference between "read and silently dropped" and
	// "never read", which a total (q1021: 458 chunks) cannot make. The
	// identifiers are the raw fact; whether an uncited read chunk MATTERS is
	// the auditor's judgement, not this ledger's.
	deepIDs    map[string]struct{}
	shallowIDs map[string]struct{}
}

// NewChunkReadLedger returns an empty chunk-read ledger for one run.
func NewChunkReadLedger() *chunkReadLedger {
	return &chunkReadLedger{
		deepIDs:    make(map[string]struct{}),
		shallowIDs: make(map[string]struct{}),
	}
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

// AddDeepIDs records the identifiers of deep-read chunks. Empty values are
// dropped and repeats collapse (the same chunk read twice is one read), so
// callers can pass raw tool-output matches as-is.
func (l *chunkReadLedger) AddDeepIDs(ids ...string) {
	l.addIDs(true, ids)
}

// AddShallowIDs records the identifiers of shallow-read chunks, same contract
// as AddDeepIDs.
func (l *chunkReadLedger) AddShallowIDs(ids ...string) {
	l.addIDs(false, ids)
}

// addIDs records one tool result's chunk identifiers under the lock, creating
// the id set on first use — so a bare &chunkReadLedger{} records ids too, not
// only one built by the constructor. deep selects which of the two sets.
func (l *chunkReadLedger) addIDs(deep bool, ids []string) {
	if l == nil || len(ids) == 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	set := &l.shallowIDs
	if deep {
		set = &l.deepIDs
	}
	if *set == nil {
		*set = make(map[string]struct{})
	}
	for _, id := range ids {
		if id == "" {
			continue
		}
		(*set)[id] = struct{}{}
	}
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

// ChunkIDs returns the identifiers of the chunks the run read, split by read
// depth and sorted so the payload a benchmark archives is byte-stable across
// runs (same rule as docIDLedger.Snapshot).
//
// A deep-read id means the chunk's FULL content was put in front of the model;
// a shallow one means it appeared as a <match_snippet> window during triage.
// The split mirrors the counts, and neither list is filtered against what the
// deliverable ended up citing: the difference between the two is exactly the
// measurement this exists for, and it is the caller's to draw.
func (l *chunkReadLedger) ChunkIDs() (deep, shallow []string) {
	if l == nil {
		return nil, nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return sortedIDSet(l.deepIDs), sortedIDSet(l.shallowIDs)
}

// sortedIDSet flattens an id set into a sorted slice.
func sortedIDSet(ids map[string]struct{}) []string {
	if len(ids) == 0 {
		return nil
	}
	out := make([]string, 0, len(ids))
	for id := range ids {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
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
	"search_chunks":          {},
	"search_semantic_chunks": {},
	"list_chunks":            {},
}

// deepReadChunkTools render FULL chunk content — their <chunk> elements are
// counted as deep reads. shallowReadChunkTools render only a
// <match_snippet> window per chunk — counted as shallow reads.
var deepReadChunkTools = map[string]struct{}{
	"search_chunks":          {},
	"search_semantic_chunks": {},
	"list_chunks":            {},
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
	// watch is the run's locate watchdog (see locate_watch.go): it counts
	// locate calls and, when the meaning-based leg is held but never called,
	// appends one reminder to the tool result the model is about to read.
	watch *locateWatch
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

// chunkIDRe pulls the chunk identifiers out of the XML every retrieval tool
// renders. Each `<chunk ...>` element carries exactly one `chunk_id` and the
// `<chunks ...>` wrapper carries none, so matching the attribute is equivalent
// to walking the elements countChunkElements counts — one wrapper covers the
// whole toolset, no element parsing, no per-tool implementation touched.
var chunkIDRe = regexp.MustCompile(`chunk_id="([^"]+)"`)

// recordedChunkIDs returns the chunk identifiers a tool's output put in front
// of the model, in the tool's own order. Duplicates are left in — the ledger
// collapses them, so raw matches pass through as-is.
func recordedChunkIDs(out string) []string {
	if out == "" {
		return nil
	}
	ids := make([]string, 0, 8)
	for _, match := range chunkIDRe.FindAllStringSubmatch(out, -1) {
		if id := strings.TrimSpace(match[1]); id != "" {
			ids = append(ids, id)
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
		// <chunk> elements; web_search renders url/title instead. The ids of
		// those chunks ride along with the counts: the count says how much
		// was read, the ids say WHICH — and only the ids can answer whether a
		// passage the run needed was ever in front of the model.
		if n := countChunkElements(out); n > 0 {
			ids := recordedChunkIDs(out)
			if _, deep := deepReadChunkTools[name]; deep {
				t.chunks.AddDeep(n)
				t.chunks.AddDeepIDs(ids...)
			} else if _, shallow := shallowReadChunkTools[name]; shallow {
				t.chunks.AddShallow(n)
				t.chunks.AddShallowIDs(ids...)
			}
		}
	}
	fields := []zap.Field{
		zap.String("tool", name),
		zap.Float64("cost_ms", float64(cost.Milliseconds())),
		zap.String("args", truncateForLog(args, 200)),
	}
	if err == nil && t.watch != nil {
		// The locate watchdog rides the result, not the prompt: a rule read
		// once at the top of the system prompt competes with everything else
		// there, while this line appears in the tool result the model is
		// looking at when it decides what to query next.
		if nudge := t.watch.observe(name); nudge != "" {
			out += "\n" + nudge
			fields = append(fields, zap.Int("locate_watchdog", 1))
		}
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
			inner, aerr := NewAnswerAuditorAgent(ctx, auditModelFor(in),
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
	// The locate watchdog may only advertise the meaning-based leg if THIS run
	// holds it, so availability is established while the tools are wrapped —
	// before the model can call anything (see locate_watch.go).
	watch := newLocateWatch()
	wrapped := make([]tool.BaseTool, len(tools))
	for i, t := range tools {
		if it, ok := t.(tool.InvokableTool); ok {
			if info, ierr := it.Info(ctx); ierr == nil && info != nil && info.Name == searchSemanticChunksToolName {
				watch.markSemanticAvailable()
			}
			wrapped[i] = &instrumentedTool{
				InvokableTool: it,
				acc:           in.ToolCallDurations,
				docs:          in.RetrievedDocIDs,
				chunks:        in.ChunkReads,
				watch:         watch,
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
	if !hasAnswerLine(final) {
		// Recovery ladder, cheapest and most faithful first. A run that ends
		// on a narration tail or on a bare tool call still holds an answer
		// somewhere in its own turns, and that answer is the model's own
		// deliverable — complete with the Candidate Matrix and the citations
		// the auditor checks. Shipping it beats a synthesis call, which
		// discards that structure and, once a failover cooldown has set in,
		// may not even reach a provider (it would be short-circuited with the
		// stale error that tripped the cooldown).
		carried := sess.explorer.lastAssistant(ctx, func(m *schema.Message) bool {
			return strings.TrimSpace(m.Content) != "" && hasAnswerLine(m.Content)
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
	// Label governance is deliberately NOT applied here: it runs ONCE, at the
	// end, on whatever text actually ships. Governing before the synthesis let
	// a synthesized deliverable escape it entirely - w3 shipped `Final Answer`
	// from the synthesizer on a run whose gate never audited at all (4 pre-audit
	// rejections, no verdict), while the log showed a demotion on the
	// pre-synthesis draft the synthesizer then replaced.

	// Terminating action: the gate loop has spent its budget (or every
	// continuation failed) and the answer still carries no machine-parseable
	// FOS deliverable. Synthesize one deterministically from the question,
	// whatever partial output exists, and the evidence gathered so far — the
	// turn must never end on narration or a blank.
	if final == "" || runErr != nil || !hasAnswerLine(final) {
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
		case synthErr == nil && synth != "" && answerLineCount(synth) == 1:
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
				zap.Int("answer_lines", answerLineCount(synth)))
			if final != "" {
				runErr = nil
			}
		case runErr == nil:
			// Normal termination but synthesis failed — surface the reason.
			runErr = synthErr
		}
	}

	// The label is the MODEL's claim and the AUDITOR's judgement, and the gate does not
	// touch it. `Final Answer` asserts that every discriminating constraint is
	// corpus-verified; the auditor's contract fails that assertion when the record does
	// not establish them (`Final Answer claims corpus-verification the record does not
	// establish`), and its repair directive tells the producer to state a label it can
	// support. The pipeline used to do that edit itself — `demoteFinalAnswerLabel`
	// spliced the word with a regex and appended a canned note — which is the pipeline
	// rewriting the deliverable rather than the deliverable's author correcting it. What
	// the gate keeps is the VERDICT, so a `Final Answer` that ships without a PASS is at
	// least visible in the log and in the archived row (GateAudit.Passed).
	observeUnverifiedFinalLabel(ctx, final, in.GateAudit)

	// Diagnosis, not enforcement: a locate tool the run called but the matrix
	// never credits leaves nothing in the archived row to show it was used. On
	// #71 the ONE call that surfaced the answer's own document was a
	// search_semantic_chunks query the matrix omitted, so only the server log
	// revealed that the pure-vector leg had done the work. Two of three runs
	// trip this (see unrecordedLocateTools), which is why it warns instead of
	// rejecting: the deliverable's job is the answer, and a bookkeeping
	// omission that the schema does not enforce is not worth a repair pass.
	if missing := unrecordedLocateTools(final, in.ToolCallCounts); len(missing) > 0 {
		common.WarnCtx(ctx, "agentic_rag: deliverable omits locate calls it used",
			zap.Strings("tools", missing))
	}

	// The answer channel carries exactly one thing: the deliverable this run
	// is shipping, emitted once, now that the gate and the fallbacks have
	// settled on it. Everything the loop said on the way there already went
	// out live as thinking, so the user watched the work without being handed
	// several competing "Final Answer" blocks.
	emit(ctx, in.OnDelta, final, "")
	return final, runErr
}

// countGateAuditFailure records an audit pass the auditor could not complete, so
// "the gate never concluded PASS" survives into the label governance even when
// the gate exits on the error instead of on a verdict. See
// GateAuditRecord.AuditFailures.
// countCitationGrounding records a deliverable refused for grounding candidates on
// citations only. See GateAuditRecord.CitationGroundings.

// adoptableContinuation reports whether a repair continuation may replace the
// standing deliverable: it must carry the FOS SECTIONS. An answer line alone may
// not - a bare "Guessed Answer: I do not have sufficient evidence" satisfies that
// while dropping every cited chunk, and adopting it threw away a structured
// deliverable to buy a re-audit that could only repeat the coverage defect
// (q1093 burned 16 passes that way).
//
// The answer VALUE is deliberately not required. A matrix without its answer line
// is auditable, and the auditor's `schema integrity: answer is missing` names the
// defect precisely; requiring the value here instead rejected the continuation
// three times and then gave up - q490, q784 and q872 all ended that way, never
// audited, on a deliverable that only needed its last line restored.
func adoptableContinuation(text string) bool {
	return hasFOSStructure(text)
}

func countCitationGrounding(audit *GateAuditRecord) {
	if audit != nil {
		audit.CitationGroundings++
	}
}

func countGateAuditFailure(audit *GateAuditRecord) {
	if audit != nil {
		audit.AuditFailures++
	}
}

// countGateRejection records a deliverable the gate refused before any audit
// ran, so "the gate never concluded PASS" survives into the label governance
// and the benchmark's accounting even when the auditor never got a turn.
func countGateRejection(audit *GateAuditRecord) {
	if audit != nil {
		audit.Rejections++
	}
}

// observeUnverifiedFinalLabel records a `Final Answer` that ships without a passing
// audit. It is an OBSERVATION, never a rewrite: the label is the deliverable author's
// claim and the auditor's judgement, so the pipeline does not splice it (it used to —
// see the removed `demoteFinalAnswerLabel`, a regex that rewrote the word and appended
// a canned English note). What the gate knows deterministically is its own verdict, and
// this line puts it in the record: a run that shipped a `Final Answer` the gate never
// PASSed is visible instead of silent.
func observeUnverifiedFinalLabel(ctx context.Context, final string, audit *GateAuditRecord) {
	if answerLabel(final) != "final" {
		return
	}
	if audit != nil && audit.Passed {
		return
	}
	rounds, rejections, failures := 0, 0, 0
	if audit != nil {
		rounds, rejections, failures = len(audit.Suspects), audit.Rejections, audit.AuditFailures
	}
	common.WarnCtx(ctx, "agentic_rag: shipping a Final Answer label the gate never PASSed",
		zap.Int("audit_rounds", rounds), zap.Int("rejections", rejections),
		zap.Int("audit_failures", failures))
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
		// The gate reads the answer LABEL, not the answer: `hasAnswerLine` is what it
		// knows without guessing at content, and the auditor owns the rest (see
		// answerLabel). The label rides into the payload as evidence for the
		// label-consistency rules.
		answerLbl := answerLabel(final)
		if strings.TrimSpace(final) != "" {
			// The gate audits the deliverable it actually holds — audit-target
			// freshness is structural, not tracked. The question lives in the
			// auditor's system prompt, so the payload carries the deliverable
			// plus the gate's own reading of it.
			auditedFinal = final
			// The hold log: the full deliverable at DEBUG, plus ONE capped INFO line
			// carrying the gate's own label reading and the deliverable's last line.
			// The question it answers is "did the gate hold the text that was
			// archived?", and only the text itself can — the audit payload is
			// truncated at 2000 chars FROM THE START, so the answer line (always the
			// LAST line) never reaches the log, and a round whose archived delivery is
			// well formed was indistinguishable from a stale-read bug (measured on
			// q283: two such rounds, and the archived text does carry a well-formed
			// answer line). It used to fire only when the gate raised a precheck;
			// prechecks are gone, and this record is the part of them worth keeping.
			common.DebugCtx(ctx, "agentic_rag: delivery gate hold (full)",
				zap.Int("pass", pass+1), zap.String("answer_label", answerLbl),
				zap.String("deliverable", final))
			common.InfoCtx(ctx, "agentic_rag: delivery gate hold",
				zap.Int("pass", pass+1),
				zap.String("gate_answer_label", answerLbl),
				zap.String("deliverable_tail", truncateForLog(lastNonBlankLine(final), 200)))
			var err error
			verdict, err = gateRunAudit(ctx, in.auditor, in.sess.auditor, final, in.toolCallCounts)
			if err != nil {
				// The auditor itself failed (LLM timeout, tool outage).
				// Retrying inside this request rarely helps; fall through to
				// finalizeAnswer, which still catches answer-less finals.
				// The failure is RECORDED before returning: this is the one exit
				// that leaves no verdict behind, and an unrecorded exit silently
				// upgrades an unverified deliverable to an audited one.
				countGateAuditFailure(in.audit)
				common.WarnCtx(ctx, "agentic_rag: delivery gate audit failed",
					zap.Int("pass", pass+1), zap.Error(err))
				return final, auditedFinal
			}
			// Per-round suspect accounting for benchmarks (Input.GateAudit):
			// every audit is recorded, PASS rounds included — a PASS verdict
			// reports 0 suspects, so the list doubles as the round count. The
			// verdict's own words ride along, so a curve can be explained from
			// the record instead of reverse-engineered from the deliverable.
			if in.audit != nil {
				in.audit.Suspects = append(in.audit.Suspects, auditSuspectCount(verdict))
				in.audit.Passed = auditPassed(verdict)
			}
			recordAuditVerdict(in.audit, verdict)
			if auditPassed(verdict) {
				break // audited and passed as a whole — ship
			}

			// The finding, not the verdict string: the auditor echoes the whole
			// deliverable back, so the raw verdict spends its first 300 chars on
			// the ## Candidate Matrix and cuts the reason off entirely.
			common.InfoCtx(ctx, "agentic_rag: delivery gate audit failed the deliverable",
				zap.Int("pass", pass+1),
				zap.Int("suspects", auditSuspectCount(verdict)),
				zap.String("verdict", truncateForLog(auditFindings(verdict), 300)))

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

		if strings.TrimSpace(final) == "" {
			// No deliverable: this repair is not a correction, it is the FIRST
			// production of the answer.
			common.InfoCtx(ctx, "agentic_rag: delivery gate skipped the audit — no deliverable yet",
				zap.Int("pass", pass+1))
			countGateRejection(in.audit)
			directive = ("You have produced NO deliverable yet, so there is nothing to ship to the user. " +
				"Continue the investigation in this turn and render the COMPLETE FOS FINAL message: the " +
				"`## Candidate Matrix` blocks, then `## Reasoning Chain`, and the LAST line " +
				"`Final Answer: **<value>**` (or `Guessed Answer: **<value>** (assumption: ...)`). " +
				"If a retrieval call errored or came back empty, rephrase the query and try again BEFORE " +
				"answering: an answer drawn from memory instead of the corpus is not acceptable, and " +
				"ending the turn on narration or on another tool call leaves the user with nothing.")
		} else {
			directive = auditRepairDirective(auditSuspectCount(verdict), verdict, suspectHist)
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
				if adoptableContinuation(trimmed) {
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
				countGateRejection(in.audit)
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
	// What the producer was actually TOLD, at full length, debug only: the repair
	// directive was previously logged nowhere, so "the model ignored the anchor
	// demand" and "the demand was never in front of it" were indistinguishable.
	common.DebugCtx(ctx, "agentic_rag: delivery gate repair directive (full)",
		zap.String("directive", directive))
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
// auditRepairDirective is the instruction the gate hands back to the producer after an
// audit FAIL: what to fix, in what order, and - since q221 - what NOT to touch.
//
// It is a function rather than an inline literal because the rules it carries are
// behavioural contracts (repair the record and not the conclusion; a stalled suspect
// count needs NEW evidence rather than a re-render) and are pinned by tests, the same
// way anchorDemand is.
func auditRepairDirective(suspects int, verdict string, hist []int) string {
	directive := fmt.Sprintf("Your FINAL message failed the pipeline's mandatory answer auditor "+
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
		anchorDemand(len(hist), hist)+

		// (b2) The two classes a failing item can belong to, because the
		// anti-swap rule below was read as a blanket ban: q283's producer quoted
		// it FOUR times across nine audits to justify keeping "The Hatchling"
		// while the auditor correctly said the value does not fill the asked
		// Z-name slot (the gold, `Zimri Elder`, is in two corpus docs). The rule
		// bans the swap that silences a complaint with no evidence behind it; it
		// was never a licence to re-render a value the auditor has already shown
		// to fail a constraint, and the cost of that misreading was eight extra
		// audits and a wrong answer.
		"TWO KINDS OF FAILING ITEM, TWO DIFFERENT REPAIRS. (i) A RECORD item - a missing or malformed field, an unsupported "+
		"step, an untested rival, a line that only NAMES the candidate - is repaired IN THE RECORD, and never by changing which "+
		"value ships. (ii) An item about the VALUE ITSELF - `the answer does not fill the asked slot`, `answer is not grounded "+
		"in the candidate matrix`, a cited chunk that contradicts it, a constraint the value FAILS - IS evidence against that "+
		"value, so the repair is to CHANGE THE VALUE: re-derive it from a candidate that satisfies the slot, or, when the corpus "+
		"holds no such candidate, ship the corpus's negative result under `Guessed Answer` with the searches that showed it. "+
		"Class (ii) is not what the rule below forbids: what is forbidden is a swap with no evidence behind it, and what is "+
		"equally forbidden is re-rendering the same value a third time when the auditor has shown a constraint it fails. "+
		// (c) Repair the RECORD, not the conclusion. The cheapest repair
		// observed is a SWAP: q221 shipped "Opium: A Portrait of the
		// Heavenly Demon" under `Final Answer` right after the auditor pushed
		// the producer off a weakness-based elimination of the gold - a PASS
		// bought by weakening the answer, strictly worse than the FAIL it
		// replaced. A grounded rival that cannot be refuted is a declared
		// TIE, never a promotion by default.
		"REPAIR THE RECORD, NOT THE CONCLUSION: an audit opinion is a claim about the RECORD - a field, "+
		"a snippet, an unsupported step, an untested rival, a constraint that is not established - and it is "+
		"NEVER on its own a reason to change which candidate you retain or which value you ship. Do NOT "+
		"swap the retained candidate or the answer value to make a complaint disappear: a deliverable "+
		"re-decided against its own evidence can pass the same auditor and then ships the WEAKER answer "+
		"under the SAME label, which is worse than the FAIL it replaces. Change the value ONLY when an "+
		"opinion names evidence AGAINST the current candidate (a chunk that contradicts it, a constraint "+
		"it fails), and carry that evidence on the new line. When a GROUNDED rival cannot be refuted, the "+
		"lawful outcome is a DECLARED TIE (a `(tie: ...)` clause per rival on the answer line), never a "+
		"silent swap; a rival that is ungrounded is eliminated for ABSENCE with the named search that "+
		"showed it, never promoted. "+
		// (c2) The label is the producer's to write and the auditor's to judge; the
		// pipeline no longer relabels a deliverable itself (it used to splice the word
		// and append a note). Without this clause the anti-swap rule above reads as a
		// ban on touching the answer line at all, and a run whose record does not
		// establish its `Final Answer` would keep claiming it.
		"THE LABEL IS NOT THE VALUE: when a failing item says the record does not establish what `Final Answer` claims "+
		"(an untested constraint, a rival not refuted, a declared tie, an unverified step), the repair is your OWN answer "+
		"line - state `Guessed Answer` with the assumption note, keep the value you can support, and leave the rest of the "+
		"record intact. `REPAIR THE RECORD, NOT THE CONCLUSION` governs the VALUE, not this claim, and nothing downstream "+
		"will relabel the deliverable for you. "+
		// (d) Fix the EVIDENCE, not the wording. v2 on q221 spent three
		// passes re-arguing ONE naming-level inference ("the collection is
		// called the Opium Collection") and flatlined at 5,5,5 while the
		// deliverable kept "advancing" in prose: a rejected argument
		// restated is not a repair, and a flat count makes the gate ship
		// the weaker deliverable.
		"FIX THE EVIDENCE, NOT THE WORDING: when an opinion says a line only NAMES the candidate (a title, "+
		"a citation, a bibliography or listing entry, a collection name), restating the same evidence in "+
		"different words is not a repair and will be rejected again - the only admissible responses are to "+
		"cite a chunk that STATES the candidate's fitness for the asked slot, or to drop the line and "+
		"eliminate the candidate for ABSENCE with the named search that showed no content about it. The same "+
		"holds for any argument the auditor has already rejected (a weakness-based elimination, a "+
		"naming-level inference): change the EVIDENCE or drop the CLAIM, never re-wrap the claim. An attempt "+
		"that advances only the prose leaves the suspect count where it was. "+
		"Citation-field repairs: call list_chunks for the cited chunk_id and OVERWRITE the line's "+
		"fields with what the tool actually returns - never guess an identifier from memory; a value "+
		"you cannot look up must be dropped, not invented. Never leave an item whose opinion is not "+
		"`pass` in the deliverable.", suspects, verdict)
	return directive
}

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
		"`Searched` line. A repair turn that re-renders the matrix with no new `Searched` line is rejected outright. " +
		// The anchor that keeps a flat count flat is an anchor about the CANDIDATE
		// the run already holds. q283 is the measured shape: 123 retrievals over
		// nine audits, every one of them about Outer Wilds or the Z-name, and the
		// evidence set (nine docs, including the developer's own page wording the
		// black hole) was NEVER served - zero of nine. Another query about the
		// candidate cannot produce a value that fills the slot it fails.
		"When the failing items are about the VALUE (an asked slot it does not fill, a chunk that contradicts it, a constraint " +
		"it fails), the new anchor must be one that can surface a RIVAL candidate for the SAME constraint: run the broad " +
		"listing/overview query the constraint makes enumerable (the game/show/book list, the roster, the year list) and test " +
		"each hit against the slot. An anchor that can only return more about the candidate you already hold keeps this count " +
		"where it is, and the gate ships what stands.")
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
	// The auditor's INPUT and OUTPUT at full length, debug only: the tail of both is
	// the answer-line region (the deliverable it echoed back plus its opinions), and
	// the capped lines above can never reach it. Log volume is the cost and it is
	// deliberate - these two are the only record of what the auditor was shown.
	common.DebugCtx(ctx, "agentic_rag: gate-run audit payload (full)", zap.String("payload", payload))
	head := conv.head(ctx)
	iter := conv.runner(ctx, auditor, false).Run(ctx, []adk.Message{schema.UserMessage(payload)})
	verdict, _, err := consumeAgentEvents(ctx, iter, func(string, string) {}, toolCallCounts, nil, nil)
	if err != nil {
		conv.discardFailedTurn(ctx, head)
		return "", err
	}
	common.InfoCtx(ctx, "agentic_rag: gate-run audit verdict",
		zap.String("verdict", truncateForLog(verdict, 2000)))
	common.DebugCtx(ctx, "agentic_rag: gate-run audit verdict (full)", zap.String("verdict", verdict))
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
					zap.String("content_head", loggable(content, 2000)),
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
					zap.String("args", loggable(tc.Function.Arguments, 2000)),
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
		if hasAnswerLine(synth) {
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

// logFullToolResults keeps the WHOLE tool output and tool arguments in the log
// instead of their 2000-char heads. Read once from the environment
// (RAGFLOW_LOG_FULL_TOOL_RESULTS=1) because the volume is real: one question can
// read ~1700 chunks and a single semantic fan-out returns ~60KB, which is ~100MB of
// log for ONE question. The head plus `content_bytes` shows the shape, and the
// content is re-fetchable; a run that is being diagnosed is where the whole text
// earns its keep.
//
// The gate's own full-text lines (deliverable, audit payload, verdict, repair
// directive) are deliberately NOT gated by this switch: they are small, they are few
// per run, and they are exactly what a "did the gate hold the archived text?"
// question needs.
var logFullToolResults = os.Getenv("RAGFLOW_LOG_FULL_TOOL_RESULTS") != ""

// loggable returns s whole when the full-tool-result switch is on, and capped to max
// otherwise.
func loggable(s string, max int) string {
	if logFullToolResults {
		return s
	}
	return truncateForLog(s, max)
}

// truncateForLog caps a string for log lines so a huge tool output or argument
// payload cannot blow up the log volume.
// lastNonBlankLine returns the deliverable's last non-empty line - the answer
// line, when the deliverable is well formed. It exists to make the gate's own
// reading auditable: the audit payload is logged truncated FROM THE START, so the
// answer line never reaches the log, and a round whose report says a value is
// missing cannot be checked against the text the run shipped.
func lastNonBlankLine(s string) string {
	lines := strings.Split(s, "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if t := strings.TrimSpace(lines[i]); t != "" {
			return t
		}
	}
	return ""
}

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
