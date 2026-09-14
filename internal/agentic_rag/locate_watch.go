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
	"fmt"
	"strings"
	"sync"
)

// Locate watchdog: the meaning-based leg is the one locate tool a run can hold
// and never reach for.
//
// The evidence for putting the nudge here rather than in the prompt is measured,
// not stylistic. The question of WHEN to use the semantic bridge was stated in
// four places in the system prompt (role list, retrieval step 2, convergence
// step 5, the output-format schema); it was rewritten into a countable trigger
// ("three lexical queries on one clue with no testable candidate") and
// re-measured on #221: two runs, zero bridge calls, both times. Across the whole
// history #221 used it 0 times in 6 runs, q350 1-5 times per run, and #71 once —
// after 16 lexical calls, at 46% of the run. A rule the model reads once at the
// top of a 30k-character prompt competes with everything else in it; a line
// attached to the tool result it is looking at does not.
//
// So: count locate calls per run, and when the meaning-based leg is held but has
// never been called, say so ON THE CURRENT TOOL RESULT — the channel the loop is
// actually reading at the moment it decides what to query next.
const (
	// locateWatchdogAfter is how many locate calls may pass without a single
	// semantic-leg call before the watchdog speaks. Three is the count the
	// prompt uses, and it matches where the measured runs went wrong: #221 made
	// 28 locate calls (0 bridge), #71 made 16 before its first bridge call, so
	// every one of those runs would have been nudged long before it committed.
	locateWatchdogAfter = 3
	// locateWatchdogMaxNudges bounds how often it speaks per run: a reminder
	// that repeats every round becomes noise the model learns to skip. Two
	// shots (at 3 and 6 locate calls) cover both "tried early" and "tried late".
	locateWatchdogMaxNudges = 2
)

// locateWatchToolNames are the calls that count towards the watchdog's tally:
// the three locate legs. list_chunks is a deep read of a document the run
// already found, so it neither counts nor triggers.
var locateWatchToolNames = map[string]struct{}{
	grepChunksToolName:           {},
	searchBm25ChunksToolName:     {},
	searchChunksToolName:         {},
	searchSemanticChunksToolName: {},
}

// locateWatch is the per-run tally behind the watchdog. One instance is created
// per Run and shared by every tool wrapper of that run; the tools execute
// concurrently within a round, so the counters are mutex-guarded.
type locateWatch struct {
	mu sync.Mutex
	// semanticAvailable records that the run's toolset actually holds the
	// meaning-based leg. Without it the nudge would advertise a tool the model
	// cannot call.
	semanticAvailable bool
	locateCalls       int
	semanticCalls     int
	nudges            int
}

// newLocateWatch returns an empty watchdog for one run.
func newLocateWatch() *locateWatch {
	return &locateWatch{}
}

// markSemanticAvailable is called once per tool as the run's tools are wrapped,
// so availability is known before the first call is made.
func (w *locateWatch) markSemanticAvailable() {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.semanticAvailable = true
}

// observe records one tool invocation and returns the notice to append to that
// tool's result, or "" when the watchdog stays silent.
//
// It speaks only when all three hold: the run holds the meaning-based leg, the
// leg has never been called, and the locate-call count has reached the reminder
// point. A run that used the bridge once — even unproductively — is left alone:
// the point is to get the leg tried, not to police how it was used.
func (w *locateWatch) observe(tool string) string {
	if w == nil {
		return ""
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	if tool == searchSemanticChunksToolName {
		w.semanticCalls++
		return ""
	}
	if _, locate := locateWatchToolNames[tool]; !locate {
		return ""
	}
	w.locateCalls++
	if !w.semanticAvailable || w.semanticCalls > 0 {
		return ""
	}
	if w.locateCalls%locateWatchdogAfter != 0 || w.nudges >= locateWatchdogMaxNudges {
		return ""
	}
	w.nudges++
	return locateWatchdogNotice(w.locateCalls)
}

// locateWatchdogNotice is the reminder itself. It names the tool (tool results
// already name tools: <tool_error tool="..."> and the 0-hit hint both do), states
// the one thing the count cannot say — that this leg matches by MEANING, so it
// reaches documents whose wording was never guessed — and asks for exactly one
// probe, with the category shift as the fallback when the description misses.
func locateWatchdogNotice(locateCalls int) string {
	var b strings.Builder
	b.WriteString("<hint_next>")
	b.WriteString(fmt.Sprintf("%d lexical queries so far and %s has not been tried once. ",
		locateCalls, searchSemanticChunksToolName))
	b.WriteString("It ranks by MEANING, so it can surface a document whose wording you have not guessed — which no amount of re-wording reaches. ")
	b.WriteString("Before the next re-worded query, run ONE probe there: describe the thing in plain language (what it IS and what is true about it). ")
	b.WriteString("If that comes back off-target, describe a different CATEGORY of thing (restaurant -> hotel -> winery -> manufacturer)")
	b.WriteString("</hint_next>")
	return b.String()
}
