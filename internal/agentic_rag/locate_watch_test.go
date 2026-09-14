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
	"strings"
	"testing"
)

// TestLocateWatchdogSpeaksOnTheCount pins the cadence: silence for the first two
// locate calls, one reminder on the third and one on the sixth, then nothing —
// a reminder that repeats every round is noise the model learns to skip.
func TestLocateWatchdogSpeaksOnTheCount(t *testing.T) {
	w := newLocateWatch()
	w.markSemanticAvailable()

	for i := 1; i <= 2; i++ {
		if got := w.observe(grepChunksToolName); got != "" {
			t.Fatalf("locate call %d: watchdog spoke too early: %q", i, got)
		}
	}
	got := w.observe(searchBm25ChunksToolName)
	if !strings.Contains(got, searchSemanticChunksToolName) {
		t.Fatalf("the 3rd locate call must name the meaning-based leg, got %q", got)
	}
	if !strings.Contains(got, "CATEGORY") || !strings.Contains(got, "MEANING") {
		t.Fatalf("the reminder must carry both the reason and the category fallback, got %q", got)
	}
	for i := 4; i <= 5; i++ {
		if q := w.observe(grepChunksToolName); q != "" {
			t.Errorf("locate call %d should be silent, got %q", i, q)
		}
	}
	if q := w.observe(grepChunksToolName); q == "" {
		t.Error("the 6th locate call should remind once more")
	}
	for i := 7; i <= 9; i++ {
		if q := w.observe(grepChunksToolName); q != "" {
			t.Errorf("call %d: the nudge cap must silence the watchdog, got %q", i, q)
		}
	}
}

// TestLocateWatchdogScope: it only speaks about a tool the run holds, it does not
// count deep reads, and one use of the bridge — even an unproductive one — ends
// the reminders. The point is getting the leg tried, not policing its use.
func TestLocateWatchdogScope(t *testing.T) {
	// No semantic leg in this toolset: the reminder would advertise a tool the
	// model cannot call.
	absent := newLocateWatch()
	for i := 0; i < 9; i++ {
		if got := absent.observe(grepChunksToolName); got != "" {
			t.Fatalf("must not advertise an absent tool: %q", got)
		}
	}

	// list_chunks opens a document the run already located: it is neither a
	// locate call nor a trigger.
	deepOnly := newLocateWatch()
	deepOnly.markSemanticAvailable()
	for i := 0; i < 6; i++ {
		if got := deepOnly.observe(listChunksToolName); got != "" {
			t.Fatalf("deep reads must not advance the tally: %q", got)
		}
	}
	if got := deepOnly.observe(grepChunksToolName); got != "" {
		t.Errorf("one locate call after six deep reads is still call #1, got %q", got)
	}

	used := newLocateWatch()
	used.markSemanticAvailable()
	used.observe(searchSemanticChunksToolName)
	for i := 0; i < 9; i++ {
		if got := used.observe(grepChunksToolName); got != "" {
			t.Fatalf("a used bridge must end the reminders: %q", got)
		}
	}
}
