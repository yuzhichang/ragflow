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

// The served-unread watchdog turns the lead-consumption duty from a line in
// the system prompt into a line on the tool result: documents served
// repeatedly but never deep-read are named where the model is looking when it
// decides what to query next. A deep read consumes the lead and re-arms the
// watchdog; a pile below the threshold stays silent; so does a non-search
// call.
func TestServedUnreadWatch(t *testing.T) {
	served := NewServedLedger()
	deep := NewDocIDLedger()
	w := newServedUnreadWatch(served, deep)
	if w == nil {
		t.Fatal("a watchdog over live ledgers must exist")
	}

	// A document served once is a neighbour, not a pattern: silent even
	// after a search.
	served.Add("10509")
	if got := w.observe(grepChunksToolName); got != "" {
		t.Fatalf("one serve event must stay silent, got %q", got)
	}

	// Three documents served repeatedly and never opened: the watchdog
	// speaks, naming the stems, once.
	for _, d := range []string{"13204", "16606", "10509"} {
		served.Add(d)
		served.Add(d)
		served.Add(d)
	}
	got := w.observe(searchBm25ChunksToolName)
	if got == "" {
		t.Fatal("a three-document unread pile must trigger the reminder")
	}
	for _, want := range []string{"10509", "list_chunks"} {
		if !strings.Contains(got, want) {
			t.Fatalf("reminder must name the lead and the tools, got %q", got)
		}
	}

	// One reminder per deep-read epoch: searches stay quiet until a lead is
	// consumed.
	if again := w.observe(grepChunksToolName); again != "" {
		t.Fatalf("the same epoch must not be reminded twice, got %q", again)
	}

	// Opening one lead re-arms the watchdog, but the opened document leaves
	// the pile and the remainder is below the threshold: silent.
	deep.Add("13204")
	if got := w.observe(listChunksToolName); got != "" {
		t.Fatalf("a deep read must be silent, got %q", got)
	}
	if again := w.observe(grepChunksToolName); again != "" {
		t.Fatalf("a pile below the threshold must stay silent, got %q", again)
	}

	// A fresh pile above the threshold is named again.
	for _, d := range []string{"17061", "17062", "1756"} {
		served.Add(d)
		served.Add(d)
		served.Add(d)
	}
	got = w.observe(searchSemanticChunksToolName)
	if got == "" || !strings.Contains(got, "17061") {
		t.Fatalf("a fresh pile must be named, got %q", got)
	}

	// A nil watchdog (no ledgers) is permanently silent and the observe path
	// must not panic on non-search tools.
	var nilWatch *servedUnreadWatch
	if got := nilWatch.observe(grepChunksToolName); got != "" {
		t.Fatalf("nil watchdog must stay silent, got %q", got)
	}
	if got := w.observe(thinkToolName); got != "" {
		t.Fatalf("a non-search tool must not trigger the watchdog, got %q", got)
	}
}
