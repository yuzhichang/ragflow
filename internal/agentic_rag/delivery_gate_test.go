package agentic_rag

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/schema"
)

// testAuditMaxPass is the audit budget the gate tests hand to
// runDeliveryGate. Production reads the same knob from the template
// (Template.AuditMaxPass, `audit_max_pass` in conf/agentic_rag.yaml); the
// tests pin it so a budget edit in the yaml cannot silently change what the
// gate tests assert.
const testAuditMaxPass = 15

func TestAuditVerdictParsing(t *testing.T) {
	cases := []struct {
		verdict  string
		passed   bool
		suspects int
	}{
		{verdict: "Audit Result: PASS", passed: true, suspects: 0},
		{verdict: "line above\nAudit Result: FAIL (2 suspects)", passed: false, suspects: 2},
		{verdict: "Audit Result: FAIL (12 suspects)", passed: false, suspects: 12},
		{verdict: "Audit Result: FAIL (0 suspects)", passed: false, suspects: 0},
		{verdict: "", passed: false, suspects: 0},
		{verdict: "audit result: fail (1 suspect)", passed: false, suspects: 1},
	}
	for _, tc := range cases {
		if got := auditPassed(tc.verdict); got != tc.passed {
			t.Errorf("auditPassed(%q) = %v, want %v", tc.verdict, got, tc.passed)
		}
		if got := auditSuspectCount(tc.verdict); got != tc.suspects {
			t.Errorf("auditSuspectCount(%q) = %d, want %d", tc.verdict, got, tc.suspects)
		}
	}
}

func TestFinalAnswerValue(t *testing.T) {
	cases := []struct {
		final string
		want  string
	}{
		{final: "## Final Answer: **1939**", want: "1939"},
		{final: "#### Final Answer: **Chattapadhyay**", want: "Chattapadhyay"},
		{final: "Guessed Answer: **1941** (assumption: the 1940 record is stale)", want: "1941"},
		{final: "The answer is 1939, as the documents show.", want: ""},
		{final: "", want: ""},
		// Trailing sentence punctuation after the closing bold is tolerated:
		// models routinely write `**...**。`, and the strict whitespace-only
		// tail pushed a good deliverable into finalizeAnswer, whose output
		// shipped TWO Final Answer lines (关羽 run, 2026-09-05).
		{final: "Final Answer: **关羽共斩杀有名有姓之人 15 人（含\"过五关斩六将\"6人）**。", want: "关羽共斩杀有名有姓之人 15 人（含\"过五关斩六将\"6人）"},
		{final: "Final Answer: **1897.**", want: "1897."},
		{final: "Final Answer: **1897**\n\n", want: "1897"},
	}
	for _, tc := range cases {
		if got := finalAnswerValue(tc.final); got != tc.want {
			t.Errorf("finalAnswerValue(%q) = %q, want %q", tc.final, got, tc.want)
		}
	}
}

// The finalize synthesis must produce exactly ONE FOS answer line. Zero (the
// answer line lost its shape) or two-plus (the model re-rendered the whole
// deliverable inside a `Final Answer: **## ...**` mega-value, shipping two
// Final Answer lines) marks the synthesis degenerate.
func TestFinalAnswerLineCount(t *testing.T) {
	sane := "## Candidate Matrix\n...\nFinal Answer: **1897**"
	if got := finalAnswerLineCount(sane); got != 1 {
		t.Errorf("sane synthesis answer lines = %d, want 1", got)
	}
	doubled := "Final Answer: **## A whole document\n\nFinal Answer: **1897**。**"
	if got := finalAnswerLineCount(doubled); got != 0 {
		t.Errorf("mega-value blob answer lines = %d, want 0 (the inner value spans lines so the line-anchored pattern cannot match it)", got)
	}
	lineBoundedDouble := "Final Answer: **1897**\nFinal Answer: **1900**"
	if got := finalAnswerLineCount(lineBoundedDouble); got != 2 {
		t.Errorf("two answer lines = %d, want 2", got)
	}
	if got := finalAnswerLineCount(""); got != 0 {
		t.Errorf("empty = %d, want 0", got)
	}
}

// The FOS-structure sentinel distinguishes an audit-failed DELIVERABLE
// (worth shipping verbatim) from pure narration (only synthesis can help).
func TestHasFOSStructure(t *testing.T) {
	cases := []struct {
		final string
		want  bool
	}{
		{final: "## Candidate Matrix\n### Sub-question 1: x\n- Retained: Bob\n## Reasoning Chain\n- Clue: x\n", want: true},
		{final: "### Candidate Matrix only, no chain\n", want: true},
		{final: "prose about the search, then:\n## Reasoning Chain heading\n", want: true},
		{final: "Let me re-render the deliverable. Please audit it.\n", want: false},
		{final: "", want: false},
	}
	for _, tc := range cases {
		if got := hasFOSStructure(tc.final); got != tc.want {
			t.Errorf("hasFOSStructure(%q) = %v, want %v", tc.final, got, tc.want)
		}
	}
}

func TestBuildAuditPayload(t *testing.T) {
	final := "## Candidate Matrix\n" +
		"### Sub-question 1: who signed\n" +
		"- Retained: Bob\n" +
		"## Reasoning Chain\n" +
		"- Clue: Bob signed in 1897 (doc: a.md, doc_id: d1, chunk_id: c1, snippet: \"...\")\n" +
		"Final Answer: **1897**\n"

	p := buildAuditPayload(final)
	var decoded auditPayload
	if err := json.Unmarshal([]byte(p), &decoded); err != nil {
		t.Fatalf("buildAuditPayload produced invalid JSON: %v\n%s", err, p)
	}
	// Verbatim passthrough: the auditor reads and echoes the FINAL message's
	// own md structure, so the gate must not reshape it.
	if decoded.FinalMessage != final {
		t.Errorf("final_message not verbatim:\n got %q\nwant %q", decoded.FinalMessage, final)
	}
}

func TestBuildAuditPayloadEmptyFinal(t *testing.T) {
	p := buildAuditPayload("")
	var decoded auditPayload
	if err := json.Unmarshal([]byte(p), &decoded); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	// A narration/empty final passes through verbatim; the auditor (not the
	// gate) flags it as schema integrity.
	if decoded.FinalMessage != "" {
		t.Errorf("unexpected payload %+v", decoded)
	}
}

// newTestConversation returns the unit under test: one managed conversation
// over a fresh in-memory session log.
func newTestConversation() *conversation {
	return newRunSession().auditor
}

// fakeAuditorAgent stands in for the answer_auditor agent: it records the
// message list each Run received and emits one assistant verdict. Satisfying
// adk.Agent lets the gate drive it through a real Runner, so these exercises
// cover the session machinery too — including the history the Runner replays.
type fakeAuditorAgent struct {
	gotMsgs  []*schema.Message
	priorLen int
	runs     int
	verdict  string
	// verdictSeq, when set, supplies the verdict per run (1-based, cycling) so
	// a test can script a suspect count that moves between passes. Empty means
	// the fixed verdict is used.
	verdictSeq []string
	// emitted records every verdict handed out, for diagnosing a loop that
	// stopped on an unexpected pass.
	emitted    []string
	toolResult string
}

func (f *fakeAuditorAgent) Name(context.Context) string { return "answer_auditor" }
func (f *fakeAuditorAgent) Description(context.Context) string {
	return "scripted stand-in for the auditor"
}

func (f *fakeAuditorAgent) Run(_ context.Context, input *adk.AgentInput, _ ...adk.AgentRunOption) *adk.AsyncIterator[*adk.AgentEvent] {
	iter, gen := adk.NewAsyncIteratorPair[*adk.AgentEvent]()
	f.runs++
	f.gotMsgs = input.Messages
	f.priorLen = len(input.Messages) - 1 // minus the payload user message
	// Resolved on the calling goroutine: reading f.runs from inside the
	// goroutine below races with the NEXT Run's increment, which made a
	// scripted suspect sequence skip entries and stall detection fire at
	// inconsistent passes (observed flip-flopping 19 ↔ 14 audits).
	verdict := f.verdictAt(f.runs)
	f.emitted = append(f.emitted, verdict)
	toolResult := f.toolResult
	go func() {
		if toolResult != "" {
			gen.Send(assistantMsgEvent("deep-read", []schema.ToolCall{
				{ID: "c1", Function: schema.FunctionCall{Name: "list_chunks", Arguments: "{}"}},
			}))
			gen.Send(&adk.AgentEvent{Output: &adk.AgentOutput{
				MessageOutput: &adk.TypedMessageVariant[*schema.Message]{
					Role: schema.Tool,
					Message: &schema.Message{
						Role: schema.Tool, ToolCallID: "c1", Content: toolResult,
					},
				},
			}})
		}
		gen.Send(assistantMsgEvent(verdict, nil))
		gen.Close()
	}()
	return iter
}

// verdictAt returns the verdict for the n-th run (1-based). A non-empty
// verdictSeq CYCLES, so a test can script a repeating suspect count without
// spelling out a ceiling-length slice.
func (f *fakeAuditorAgent) verdictAt(n int) string {
	if len(f.verdictSeq) == 0 {
		return f.verdict
	}
	return f.verdictSeq[(n-1)%len(f.verdictSeq)]
}

func TestGateRunAuditReturnsVerdict(t *testing.T) {
	fake := &fakeAuditorAgent{verdict: "## Reasoning Chain\n- Clue: x\n  - audit: pass\nFinal Answer: **1**\n  - audit: pass\nAudit Result: PASS"}

	verdict, err := gateRunAudit(context.Background(), fake, newTestConversation(),
		"Final Answer: **1897**", nil)
	if err != nil {
		t.Fatalf("gateRunAudit error: %v", err)
	}
	if len(fake.gotMsgs) != 1 || fake.gotMsgs[0].Role != schema.User {
		t.Fatalf("first audit must hand over nothing but its payload, got %v", fake.gotMsgs)
	}
	if !strings.Contains(fake.gotMsgs[0].Content, `"final_message":"Final Answer: **1897**"`) {
		t.Errorf("payload missing the deliverable: %s", fake.gotMsgs[0].Content)
	}
	if !auditPassed(verdict) {
		t.Errorf("verdict = %q, want a PASS verdict", verdict)
	}
}

// flakyExplorer fails its first `failures` Runs with a tool error, then
// answers normally — the shape of an Elasticsearch outage mid-repair.
type flakyExplorer struct {
	failures int
	runs     int
	msgs     []*schema.Message
}

func (f *flakyExplorer) Name(context.Context) string { return "flaky-explorer" }
func (f *flakyExplorer) Description(context.Context) string {
	return "scripted stand-in for the explorer"
}

func (f *flakyExplorer) Run(_ context.Context, input *adk.AgentInput, _ ...adk.AgentRunOption) *adk.AsyncIterator[*adk.AgentEvent] {
	iter, gen := adk.NewAsyncIteratorPair[*adk.AgentEvent]()
	f.runs++
	f.msgs = input.Messages
	failing := f.runs <= f.failures
	go func() {
		if failing {
			gen.Send(&adk.AgentEvent{Err: errors.New("Elasticsearch query failed: connection refused")})
			gen.Close()
			return
		}
		gen.Send(assistantMsgEvent("## Reasoning Chain\n- Clue: x\nFinal Answer: **1897**", nil))
		gen.Close()
	}()
	return iter
}

// A repair turn that keeps aborting on a tool error is retried IN PLACE — the
// audit is NOT re-run, because the deliverable under repair never changed and
// re-auditing it would only reproduce the verdict already in hand.
func TestRunDeliveryGateRetriesToolFailureWithoutReAuditing(t *testing.T) {
	auditor := &fakeAuditorAgent{verdict: "Audit Result: FAIL (1 suspects)"}
	explorer := &flakyExplorer{failures: 99} // every repair turn aborts
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: testAuditMaxPass,
	}

	final, audited := runDeliveryGate(context.Background(), in)
	if auditor.runs != 1 {
		t.Errorf("audits = %d, want 1 - a failed repair must not trigger a re-audit", auditor.runs)
	}
	if explorer.runs != gateNoProgressLimit {
		t.Errorf("repair attempts = %d, want gateNoProgressLimit=%d", explorer.runs, gateNoProgressLimit)
	}
	if final != "old final" || audited != "old final" {
		t.Errorf("final = %q / audited = %q, want the standing final preserved", final, audited)
	}
	// The retry directive must name the failure so the model can avoid the call.
	if !strings.Contains(explorer.msgs[len(explorer.msgs)-1].Content, "TOOL FAILURE NOTICE") {
		t.Errorf("retry directive does not name the failure: %s", explorer.msgs[len(explorer.msgs)-1].Content)
	}
}

// A template with no audit budget ships unaudited: the gate is not entered at
// all, so neither the auditor nor a repair turn ever runs. This is the shape
// `audit_max_pass` absent (or 0) takes, and it is what keeps the auditor
// template itself from auditing its own audits.
func TestRunDeliveryGateSkipsAuditWithoutBudget(t *testing.T) {
	auditor := &fakeAuditorAgent{verdict: "Audit Result: FAIL (1 suspects)"}
	explorer := &flakyExplorer{}
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: 0,
	}

	final, audited := runDeliveryGate(context.Background(), in)
	if auditor.runs != 0 || explorer.runs != 0 {
		t.Errorf("audits = %d / repair turns = %d, want 0 each - no budget means no gate", auditor.runs, explorer.runs)
	}
	if final != "old final" {
		t.Errorf("final = %q, want the standing final untouched", final)
	}
	if audited != "" {
		t.Errorf("auditedFinal = %q, want empty (nothing was audited, so there is no fallback candidate)", audited)
	}
}

// The budget is the ceiling: a template that names 3 gets 3 audits, however
// promising the repair curve still looks. Same falling-count script as the
// full-budget test, so only the ceiling differs.
func TestRunDeliveryGateStopsAtTemplateBudget(t *testing.T) {
	falling := make([]string, 0, testAuditMaxPass+1)
	for n := 20; n >= 1; n-- {
		falling = append(falling, fmt.Sprintf("Audit Result: FAIL (%d suspects)", n))
	}
	auditor := &fakeAuditorAgent{verdictSeq: falling}
	in := deliveryGateInput{
		explorer:     &flakyExplorer{},
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: 3,
	}

	runDeliveryGate(context.Background(), in)
	if auditor.runs != 3 {
		t.Errorf("audits = %d, want 3 - the template's budget, not a hardcoded ceiling", auditor.runs)
	}
}

// A repair that DOES advance the deliverable is re-audited — that is the only
// case where a fresh audit says something new. The count falls on every pass
// here, so the stall check never fires and the ceiling is what stops the loop.
func TestRunDeliveryGateReAuditsAfterAdoptingRepair(t *testing.T) {
	// A strictly falling count (20, 19, 18, ...): the newest verdict is always
	// below both of its predecessors.
	falling := make([]string, 0, testAuditMaxPass+1)
	for n := 20; n >= 1; n-- {
		falling = append(falling, fmt.Sprintf("Audit Result: FAIL (%d suspects)", n))
	}
	auditor := &fakeAuditorAgent{verdictSeq: falling}
	explorer := &flakyExplorer{} // every repair returns a deliverable-shaped final
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: testAuditMaxPass,
	}

	final, _ := runDeliveryGate(context.Background(), in)
	if auditor.runs != testAuditMaxPass {
		t.Errorf("audits = %d, want auditMaxPass=%d (each adopted repair deserves a re-audit)",
			auditor.runs, testAuditMaxPass)
	}
	if finalAnswerValue(final) != "1897" {
		t.Errorf("final = %q, want the adopted repair deliverable", final)
	}
}

// A count that never falls is the plain stall: the newest verdict matches both
// of its predecessors, so the third observation ends the loop. q99 sat on 4
// suspects for 16 passes, q481 on 5 for 12, q371 on 9 for 9 — the ceiling used
// to let each spend 20 passes to establish what three had already shown.
func TestRunDeliveryGateStopsWhenSuspectsStall(t *testing.T) {
	auditor := &fakeAuditorAgent{verdict: "Audit Result: FAIL (5 suspects)"}
	explorer := &flakyExplorer{} // every repair is adopted; the count never moves
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: testAuditMaxPass,
	}

	final, _ := runDeliveryGate(context.Background(), in)
	if auditor.runs != gateStallWindow {
		t.Errorf("audits = %d, want gateStallWindow=%d - identical verdicts must not buy more passes",
			auditor.runs, gateStallWindow)
	}
	// The stalling audit exits before buying its repair turn.
	if explorer.runs != gateStallWindow-1 {
		t.Errorf("repair turns = %d, want %d", explorer.runs, gateStallWindow-1)
	}
	// A stall ships the deliverable in hand - the same one the loop would have
	// ended on after burning the full ceiling.
	if finalAnswerValue(final) != "1897" {
		t.Errorf("final = %q, want the repaired deliverable shipped", final)
	}
}

// A low suspect count gets NO extra slack. An earlier rule gave counts of two
// or fewer 15 passes, earned by q667 (a 13-pass stall at 2 suspects that then
// cleared and scored 4) - but once the candidate-matrix audit was removed, runs
// began freezing at 1 suspect instead, and that slack only bought 17-20 passes
// of a repair going nowhere. The trend test is flat: three observations, same
// rule whatever the count.
func TestRunDeliveryGateGivesLowSuspectsNoExtraSlack(t *testing.T) {
	auditor := &fakeAuditorAgent{verdict: "Audit Result: FAIL (2 suspects)"}
	explorer := &flakyExplorer{}
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: testAuditMaxPass,
	}

	final, _ := runDeliveryGate(context.Background(), in)
	if auditor.runs != gateStallWindow {
		t.Errorf("audits = %d, want gateStallWindow=%d - low counts are cut as fast as high ones",
			auditor.runs, gateStallWindow)
	}
	if finalAnswerValue(final) != "1897" {
		t.Errorf("final = %q, want the repaired deliverable shipped", final)
	}
}

// Regression against the observed frames runs that motivated the split. These
// are the real per-pass suspect counts: q99 and q481 froze and never cleared
// (both scored 0 and 4 respectively on the deliverable they already held).
// q667 is the one the flat rule gives up: it broke a 13-pass stall at 2
// suspects and passed, and this cut takes it at pass 7. That is the accepted
// price of the change — five stalls of that shape were observed and only one
// ever broke, while the slack that protected them cost 112 passes over 13
// questions once runs started freezing at 1 suspect.
func TestRunDeliveryGateStallOnObservedFramesSequences(t *testing.T) {
	cases := []struct {
		name string
		// suspects holds the observed per-pass count; 0 marks a pass that PASSED.
		suspects []int
		wantRuns int
	}{
		{
			name:     "q99 bm25: 15→11→6→5 then frozen at 4 (ended FAIL, scored 0)",
			suspects: []int{15, 11, 6, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
			wantRuns: 7, // cut at the third identical 4-suspect verdict
		},
		{
			name:     "q481 reasoning: 16→5 then frozen at 5 (ended FAIL, scored 4)",
			suspects: []int{16, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
			wantRuns: 4, // 16 then three identical 5s
		},
		{
			name:     "q667 bm25: 2-suspect stall of 13, then cleared (scored 4)",
			suspects: []int{5, 6, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0},
			wantRuns: 7, // cut at the third 2-suspect verdict — this run is the
			// flat rule's known cost: left alone it cleared on pass 18.
		},
		{
			name:     "q617 reasoning: frozen at 1 suspect (ended FAIL, scored 0)",
			suspects: []int{4, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			wantRuns: 7, // falls to 1 and freezes there; the third 1 ends it
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			seq := make([]string, 0, len(tc.suspects))
			for _, n := range tc.suspects {
				if n == 0 {
					seq = append(seq, "Audit Result: PASS")
					continue
				}
				seq = append(seq, fmt.Sprintf("Audit Result: FAIL (%d suspects)", n))
			}
			auditor := &fakeAuditorAgent{verdictSeq: seq}
			explorer := &flakyExplorer{}
			in := deliveryGateInput{
				explorer:     explorer,
				auditor:      auditor,
				baseMessages: []*schema.Message{schema.UserMessage("q?")},
				final:        "old final",
				auditMaxPass: testAuditMaxPass,
			}

			runDeliveryGate(context.Background(), in)
			if auditor.runs != tc.wantRuns {
				t.Errorf("audits = %d, want %d (repair turns = %d)\nverdicts: %v",
					auditor.runs, tc.wantRuns, explorer.runs, auditor.emitted)
			}
		})
	}
}

// The window holds the last three observations, and a count that climbs back
// counts as stalled even though no two adjacent verdicts agree: f1 < f2 < f3
// means the repair is making the deliverable worse. An earlier rule that
// needed three EQUAL counts let this run on.
func TestRunDeliveryGateStopsWhenSuspectsClimbBack(t *testing.T) {
	auditor := &fakeAuditorAgent{verdictSeq: []string{
		"Audit Result: FAIL (9 suspects)", // f1
		"Audit Result: FAIL (7 suspects)", // f2 — falling, keep going
		"Audit Result: FAIL (6 suspects)", // f3 — still falling, keep going
		"Audit Result: FAIL (7 suspects)", // window [7,6,7]: 7 >= 7 and >= 6 → stop
	}}
	explorer := &flakyExplorer{}
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: testAuditMaxPass,
	}

	final, _ := runDeliveryGate(context.Background(), in)
	if auditor.runs != 4 {
		t.Errorf("audits = %d, want 4 - a count climbing back is not progress", auditor.runs)
	}
	if explorer.runs != 3 {
		t.Errorf("repair turns = %d, want 3 (all but the stalling pass)", explorer.runs)
	}
	if finalAnswerValue(final) != "1897" {
		t.Errorf("final = %q, want the repaired deliverable shipped", final)
	}
}

// Narration continuations are discarded in place too: an unchanged deliverable
// must not buy itself another audit.
func TestRunDeliveryGateRetriesNarrationWithoutReAuditing(t *testing.T) {
	auditor := &fakeAuditorAgent{verdict: "Audit Result: FAIL (1 suspects)"}
	explorer := &narratingExplorer{}
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: testAuditMaxPass,
	}

	final, _ := runDeliveryGate(context.Background(), in)
	if auditor.runs != 1 {
		t.Errorf("audits = %d, want 1 - a rejected continuation must not trigger a re-audit", auditor.runs)
	}
	if explorer.runs != gateNoProgressLimit {
		t.Errorf("repair attempts = %d, want gateNoProgressLimit=%d", explorer.runs, gateNoProgressLimit)
	}
	if final != "old final" {
		t.Errorf("final = %q, want narration never clobbering the standing final", final)
	}
	if !strings.Contains(explorer.msgs[len(explorer.msgs)-1].Content, "REJECTION NOTICE") {
		t.Errorf("retry directive does not carry the rejection reason")
	}
}

// narratingExplorer always answers with process talk and no FOS answer line.
type narratingExplorer struct {
	runs int
	msgs []*schema.Message
}

func (f *narratingExplorer) Name(context.Context) string { return "narrating-explorer" }
func (f *narratingExplorer) Description(context.Context) string {
	return "scripted stand-in for the explorer"
}

func (f *narratingExplorer) Run(_ context.Context, input *adk.AgentInput, _ ...adk.AgentRunOption) *adk.AsyncIterator[*adk.AgentEvent] {
	iter, gen := adk.NewAsyncIteratorPair[*adk.AgentEvent]()
	f.runs++
	f.msgs = input.Messages
	go func() {
		gen.Send(assistantMsgEvent("Let me re-render the deliverable now.", nil))
		gen.Close()
	}()
	return iter
}

// A run whose shared wall-clock budget has expired must stop auditing and
// repairing immediately: on a dead context every attempt can only fail, and
// the in-place retry allowance would burn on instant failures (q268).
func TestRunDeliveryGateStopsOnExpiredBudget(t *testing.T) {
	auditor := &fakeAuditorAgent{verdict: "Audit Result: FAIL (1 suspects)"}
	explorer := &flakyExplorer{failures: 99}
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "old final",
		auditMaxPass: testAuditMaxPass,
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // budget gone

	final, audited := runDeliveryGate(ctx, in)
	if auditor.runs != 0 {
		t.Errorf("audits = %d, want 0 - an expired budget must skip auditing", auditor.runs)
	}
	if explorer.runs != 0 {
		t.Errorf("repair attempts = %d, want 0 - an expired budget must skip repairs", explorer.runs)
	}
	if final != "old final" || audited != "old final" {
		t.Errorf("final = %q / audited = %q, want the standing final shipped as-is", final, audited)
	}
}

// The synthesis fallback must not inherit the run's expired deadline — it
// exists precisely for the case where that deadline ran out.
func TestFinalizeAnswerOutlivesExpiredRunContext(t *testing.T) {
	dead, cancel := context.WithCancel(context.Background())
	cancel()

	// Must survive the run's expired budget — that is the whole point.
	detached, detachedCancel := context.WithTimeout(context.WithoutCancel(dead), finalizeTimeout)
	defer detachedCancel()
	if err := detached.Err(); err != nil {
		t.Fatalf("finalize context should be alive after the run context expired: %v", err)
	}
	if _, ok := detached.Deadline(); !ok {
		t.Error("finalize context must carry its own bounded budget")
	}
	// Cancellation is deliberately NOT propagated (a deadline expiry and a
	// client hang-up are indistinguishable downstream), so the bounded budget
	// is what prevents unbounded work.
	parent, parentCancel := context.WithCancel(context.Background())
	detached2, cancel2 := context.WithTimeout(context.WithoutCancel(parent), finalizeTimeout)
	defer cancel2()
	parentCancel()
	if err := detached2.Err(); err != nil {
		t.Errorf("finalize context must not inherit cancellation: %v", err)
	}
	// Values must still reach the synthesis call.
	key := struct{}{}
	typed := context.WithValue(context.Background(), key, "v")
	if got := context.WithoutCancel(typed).Value(key); got != "v" {
		t.Errorf("value lost through WithoutCancel: %v", got)
	}
}

// A bare answer line is not a deliverable: adopting it would discard the
// structured (if failing) final and buy a re-audit that only repeats the
// coverage defect. It must be rejected like narration (q1093).
type bareAnswerExplorer struct {
	runs int
	msgs []*schema.Message
}

func (f *bareAnswerExplorer) Name(context.Context) string { return "bare-answer-explorer" }
func (f *bareAnswerExplorer) Description(context.Context) string {
	return "scripted stand-in for the explorer"
}

func (f *bareAnswerExplorer) Run(_ context.Context, input *adk.AgentInput, _ ...adk.AgentRunOption) *adk.AsyncIterator[*adk.AgentEvent] {
	iter, gen := adk.NewAsyncIteratorPair[*adk.AgentEvent]()
	f.runs++
	f.msgs = input.Messages
	go func() {
		gen.Send(assistantMsgEvent("Guessed Answer: **I do not have sufficient evidence.**", nil))
		gen.Close()
	}()
	return iter
}

func TestRunDeliveryGateRejectsBareAnswerLine(t *testing.T) {
	auditor := &fakeAuditorAgent{verdict: "Audit Result: FAIL (1 suspects)"}
	explorer := &bareAnswerExplorer{}
	in := deliveryGateInput{
		explorer:     explorer,
		auditor:      auditor,
		baseMessages: []*schema.Message{schema.UserMessage("q?")},
		final:        "## Candidate Matrix\n- Retained: Bob\nFinal Answer: **Bob**",
		auditMaxPass: testAuditMaxPass,
	}

	final, _ := runDeliveryGate(context.Background(), in)
	if auditor.runs != 1 {
		t.Errorf("audits = %d, want 1 - a bare answer line advances nothing", auditor.runs)
	}
	if explorer.runs != gateNoProgressLimit {
		t.Errorf("repair attempts = %d, want gateNoProgressLimit=%d", explorer.runs, gateNoProgressLimit)
	}
	if finalAnswerValue(final) != "Bob" || !hasFOSStructure(final) {
		t.Errorf("final = %q, want the structured standing final preserved", final)
	}
	if !strings.Contains(explorer.msgs[len(explorer.msgs)-1].Content, "REJECTION NOTICE") {
		t.Error("retry directive does not carry the rejection reason")
	}
}

func TestGateRunAuditNilAuditorErrors(t *testing.T) {
	// Must return an error (not panic) when the auditor is unavailable.
	if _, err := gateRunAudit(context.Background(), nil, newTestConversation(),
		"Final Answer: **1897**", nil); err == nil {
		t.Fatal("gateRunAudit with a nil auditor should error")
	}
}
