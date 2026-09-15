package agentic_rag

import (
	"strings"
	"testing"
)

func TestDemoteFinalAnswerLabel(t *testing.T) {
	final := "## Candidate Matrix\n...\n## Reasoning Chain\n...\nFinal Answer: **Boston**"
	out, changed := demoteFinalAnswerLabel(final, "the gate did not conclude PASS")
	if !changed {
		t.Fatal("label was not demoted")
	}
	if !strings.Contains(out, "Guessed Answer: **Boston**") {
		t.Fatalf("demoted line missing: %q", out)
	}
	if !strings.Contains(out, "(assumption: the gate did not conclude PASS)") {
		t.Fatalf("assumption reason missing: %q", out)
	}
	if strings.Contains(out, "Final Answer") {
		t.Fatalf("Final label still present: %q", out)
	}
	// Guessed answers pass through untouched.
	g := "Guessed Answer: **Boston** (assumption: x)"
	if out, changed := demoteFinalAnswerLabel(g, "r"); changed {
		t.Fatalf("guessed answer must not be rewritten: %q", out)
	}
	// A deliverable without any answer line is returned unchanged.
	if _, changed := demoteFinalAnswerLabel("no answer line here", "r"); changed {
		t.Fatal("answer-less text must not change")
	}
}

// TestDemoteRewritesHeadingAndValueLine pins the fix for the shape observed on
// 4/46 rows of the browsecomp retry batch (q233/q521/q836/q851): the heading
// and the value line carry the label separately, so rewriting only the first
// occurrence leaves `## Final Answer` above a demoted value.
func TestDemoteRewritesHeadingAndValueLine(t *testing.T) {
	final := "## Candidate Matrix\n...\n## Final Answer\n**Guessed Answer: Braun Strowman** (assumption: from the corpus)"
	out, changed := demoteFinalAnswerLabel(final, "the delivery gate did not conclude PASS")
	if !changed {
		t.Fatal("heading label was not demoted")
	}
	if strings.Contains(out, "Final Answer") {
		t.Fatalf("heading kept the stronger label: %q", out)
	}
	if !strings.Contains(out, "## Guessed Answer") {
		t.Fatalf("heading was not rewritten: %q", out)
	}
	// The assumption note lands once, on the value line that already had one.
	if strings.Count(out, "(assumption:") != 1 {
		t.Fatalf("assumption note duplicated or missing: %q", out)
	}

	// All-Final deliverable: both occurrences demoted, note appended once, on
	// the value line (not on the bare heading).
	both := "## Final Answer\n**Final Answer: Boston**"
	out, changed = demoteFinalAnswerLabel(both, "reason")
	if !changed {
		t.Fatal("conclusive deliverable must demote")
	}
	if strings.Contains(out, "Final") {
		t.Fatalf("a Final label survived: %q", out)
	}
	if !strings.Contains(out, "**Guessed Answer: Boston** (assumption: reason)") {
		t.Fatalf("assumption must ride the value line: %q", out)
	}
}

// TestReconcileAnswerLabels pins the conservative direction: a heading claiming
// Final over a Guessed value line is rewritten to Guessed, and a deliverable
// that is already consistent passes through byte-identical.
func TestReconcileAnswerLabels(t *testing.T) {
	mixed := "## Reasoning Chain\n...\n## Final Answer\n**Guessed Answer: Braun Strowman** (assumption: x)"
	out := reconcileAnswerLabels(mixed)
	if strings.Contains(out, "Final") {
		t.Fatalf("heading kept the stronger claim: %q", out)
	}
	if !strings.Contains(out, "**Guessed Answer: Braun Strowman** (assumption: x)") {
		t.Fatalf("value line must be untouched: %q", out)
	}

	// A Final-only deliverable is left alone: the gate's PASS decides that
	// label, reconciliation only removes contradictions.
	finalOnly := "## Final Answer\n**Final Answer: Boston**"
	if got := reconcileAnswerLabels(finalOnly); got != finalOnly {
		t.Fatalf("Final-only deliverable must pass through: %q", got)
	}
	// A Guessed-only deliverable is left alone too.
	guessedOnly := "## Guessed Answer\n**Guessed Answer: Boston** (assumption: x)"
	if got := reconcileAnswerLabels(guessedOnly); got != guessedOnly {
		t.Fatalf("Guessed-only deliverable must pass through: %q", got)
	}
}

func TestCountChunkElementsStillWorks(t *testing.T) {
	if n := countChunkElements("<chunks><chunk a=\"1\">x</chunk></chunks>"); n != 1 {
		t.Fatalf("count = %d, want 1", n)
	}
}

// TestReconcileAnswerLabelsBacktickForm covers a shipped shape that slipped the
// detector: the value line wrapped in backticks, under a `## Final Answer`
// heading (observed on q350). The heading must still be reconciled.
func TestReconcileAnswerLabelsBacktickForm(t *testing.T) {
	mixed := "## Final Answer\n`Guessed Answer: **insufficient corpus evidence** (assumption: x)`"
	out := reconcileAnswerLabels(mixed)
	if strings.Contains(out, "Final Answer") {
		t.Fatalf("heading kept the stronger claim: %q", out)
	}
	if !strings.Contains(out, "## Guessed Answer") {
		t.Fatalf("heading was not rewritten: %q", out)
	}
}

// TestUnrecordedLocateTools pins the diagnostic: a locate tool the run called
// must be credited on a `Searched:` line, and the check exists because an
// omitted call is invisible in the archived row (on #71 the decisive
// pure-vector call was omitted). Tools never called are not reported.
func TestUnrecordedLocateTools(t *testing.T) {
	matrix := "### Sub-question 1\n" +
		"- `Searched: search_bm25_chunks(\"Ballast Bank Wexford\") -> top: doc 15985.md`\n" +
		"- `Tested: Ballast Bank - all clues supported`\n" +
		"Final Answer: **Ballast Bank**"
	counts := map[string]int{
		"grep_chunks":            8,
		"search_bm25_chunks":     11,
		"search_semantic_chunks": 1,
		"list_chunks":            17,
	}
	got := unrecordedLocateTools(matrix, counts)
	if len(got) != 2 || got[0] != "grep_chunks" || got[1] != "search_semantic_chunks" {
		t.Fatalf("unrecordedLocateTools = %v, want [grep_chunks search_semantic_chunks]", got)
	}
	// A deep read is not a locate call, and a tool that was never called cannot
	// be missing.
	if got := unrecordedLocateTools(matrix, map[string]int{"list_chunks": 3}); len(got) != 0 {
		t.Errorf("unrecordedLocateTools with no locate calls = %v, want none", got)
	}
	if got := unrecordedLocateTools("", map[string]int{"search_chunks": 2}); len(got) != 1 || got[0] != "search_chunks" {
		t.Errorf("a deliverable with no Searched lines = %v, want [search_chunks]", got)
	}
}

// TestRecordAuditVerdict pins the audit record: one FINDING per round, in step
// with Suspects, with the echoed deliverable dropped (it is 95% of the verdict
// string and none of it is a finding), trimmed, and safe on every path —
// including a run whose gate never produced a verdict (nil record).
func TestRecordAuditVerdict(t *testing.T) {
	// The shape the auditor actually returns: the producer's message echoed back
	// with one `- audit: <opinion>` sub-line per audited line, then the overall
	// verdict. Verbatim from the answer_auditor contract in conf/agentic_rag.yaml.
	verdict := strings.Join([]string{
		"## Candidate Matrix",
		"### Sub-question 1: who signed the treaty",
		`- Retained: Bob (doc: A Title, doc_id: d1, chunk_id: c1, snippet: "...")`,
		`  - audit: field integrity: field doc is incorrect (list_chunks doc_name says "66090.md", the line claims "A Title")`,
		"## Reasoning Chain",
		"- Clue: Bob signed in 1897 (doc: 66090.md, doc_id: d1, chunk_id: c1, snippet: \"...\")",
		"  - audit: pass",
		"Final Answer: **1897**",
		"  - audit: suspect: the supported clues never derive 1897",
		"Audit Result: FAIL (2 suspects)",
	}, "\n")

	rec := &GateAuditRecord{Suspects: []int{2}}
	recordAuditVerdict(rec, verdict)
	if len(rec.AuditVerdicts) != 1 {
		t.Fatalf("verdicts = %d, want 1", len(rec.AuditVerdicts))
	}
	got := rec.AuditVerdicts[0]
	for _, want := range []string{"field integrity: field doc is incorrect", "suspect: the supported clues never derive 1897", "Audit Result: FAIL (2 suspects)"} {
		if !strings.Contains(got, want) {
			t.Errorf("excerpt must keep %q, got %q", want, got)
		}
	}
	if strings.Contains(got, "Candidate Matrix") || strings.Contains(got, "doc_id: d1") {
		t.Errorf("the echoed deliverable must not be archived, got %q", got)
	}
	if strings.Contains(got, "audit: pass") {
		t.Errorf("a passing item is not a finding, got %q", got)
	}

	// Entry i of AuditVerdicts must belong to entry i of Suspects — including a
	// PASS round, whose findings are empty by construction.
	rec.Suspects = append(rec.Suspects, 0)
	recordAuditVerdict(rec, "## Candidate Matrix\n- Retained: Bob\n  - audit: pass\nAudit Result: PASS")
	if len(rec.AuditVerdicts) != len(rec.Suspects) {
		t.Fatalf("verdicts %d and suspects %d must stay in step", len(rec.AuditVerdicts), len(rec.Suspects))
	}
	if last := rec.AuditVerdicts[len(rec.AuditVerdicts)-1]; !strings.Contains(last, "Audit Result: PASS") {
		t.Errorf("a PASS round must still record its verdict line, got %q", last)
	}

	// An empty verdict records nothing, and a nil record is not a panic.
	before := len(rec.AuditVerdicts)
	recordAuditVerdict(rec, "   ")
	recordAuditVerdict(nil, "Audit Result: PASS")
	if len(rec.AuditVerdicts) != before {
		t.Error("a blank verdict must not add an entry")
	}

	// Truncation bound, on rune boundaries.
	recordAuditVerdict(rec, "  - audit: suspect: "+strings.Repeat("é", auditVerdictExcerptMax+50))
	last := rec.AuditVerdicts[len(rec.AuditVerdicts)-1]
	if n := len([]rune(last)); n != auditVerdictExcerptMax+1 {
		t.Errorf("excerpt is %d runes, want %d (bound plus the ellipsis)", n, auditVerdictExcerptMax+1)
	}
}

// TestAuditFindings pins the distillation BOTH consumers use (the log line and
// the archived record): the echoed deliverable never reaches either, a PASS round
// reduces to its verdict line, and an unrecognised shape says so rather than
// returning "" — which a reader would take for "no issues".
func TestAuditFindings(t *testing.T) {
	got := auditFindings("## Candidate Matrix\n- Retained: Bob\n  - audit: pass\nAudit Result: PASS")
	if strings.Contains(got, "Bob") || strings.Contains(got, "Candidate Matrix") {
		t.Errorf("the echoed deliverable must be dropped, got %q", got)
	}
	if got != "Audit Result: PASS" {
		t.Errorf("a PASS round distils to its verdict line, got %q", got)
	}
	if got := auditFindings("## Candidate Matrix\n- Retained: Bob\n"); got != "audit produced no parseable opinion" {
		t.Errorf("an unrecognised shape must say so, got %q", got)
	}
}

// TestAuditRepairDirective pins the instruction the gate hands the producer after
// an audit FAIL. Two rules ride in it, and both cost a whole pass when they are
// missing:
//
//   - Lever 3 (unchanged): a repair after TWO failures must open with new
//     retrieval on a different anchor — re-rendering the same matrix leaves the
//     suspect count flat or climbing.
//   - (c) Repair the RECORD, not the conclusion: q221 shipped "Opium: A Portrait
//     of the Heavenly Demon" under `Final Answer` right after the auditor pushed
//     it off a weakness-based elimination of the gold, i.e. a PASS bought by
//     swapping the answer. The directive must forbid that swap by name, and send
//     a grounded-but-unrefutable rival to a declared TIE instead.
func TestAuditRepairDirective(t *testing.T) {
	early := auditRepairDirective(2, "Audit Result: FAIL (2 suspects)\n  - audit: suspect: competing slot-filler never tested", []int{2})
	if strings.Contains(early, "ANCHOR CHANGE REQUIRED") {
		t.Error("the anchor demand must not fire on the first failure — it would forbid a repair before one was tried")
	}
	for _, want := range []string{
		"REPAIR THE RECORD, NOT THE CONCLUSION",
		// (d) A rejected argument restated is not a repair: v2 on q221 spent
		// three passes re-arguing one naming-level inference and flatlined.
		"FIX THE EVIDENCE, NOT THE WORDING",
		"a `(tie: ...)` clause per rival on the answer line",
		"eliminated for ABSENCE",
		"(2 suspect item(s))",
		// The verdict rides along verbatim: the producer repairs the lines the
		// auditor named, not a paraphrase of them.
		"competing slot-filler never tested",
	} {
		if !strings.Contains(early, want) {
			t.Errorf("repair directive must carry %q", want)
		}
	}

	stalled := auditRepairDirective(3, "Audit Result: FAIL (3 suspects)", []int{5, 4, 6})
	if !strings.Contains(stalled, "ANCHOR CHANGE REQUIRED") {
		t.Error("a repair after two failures must demand new evidence on a different anchor")
	}
	if !strings.Contains(stalled, "5 -> 4 -> 6") {
		t.Errorf("the directive must show the suspect trend it is reacting to, got:\n%s", stalled)
	}
}

// TestGovernAnswerLabel pins the ONE label rule — and the reason it has to run on
// the text that SHIPS: it used to be inlined before the last-resort synthesis,
// which replaces the answer with a freshly rendered deliverable carrying its own
// label, so a synthesized `Final Answer` shipped ungoverned (w3: four pre-audit
// rejections, no audit at all, `Final Answer` on the synthesizer's text).
func TestGovernAnswerLabel(t *testing.T) {
	cases := []struct {
		name       string
		final      string
		audit      *GateAuditRecord
		wantDemote bool
	}{
		{
			// "never audited" is still a reason: the gate refused this deliverable
			// four times and stopped, so nothing verified it.
			name:       "pre-audit rejections demote",
			final:      "Final Answer: **X**",
			audit:      &GateAuditRecord{Rejections: 4},
			wantDemote: true,
		},
		{
			name:       "a passing audit lets the label stand",
			final:      "Final Answer: **X**",
			audit:      &GateAuditRecord{Suspects: []int{0}, Passed: true},
			wantDemote: false,
		},
		{
			// A tie does NOT demote here any more: the gate reads labels, and whether a
			// clause is a real tie is the auditor's judgement (its contract carries
			// `Final Answer claims a tie`). A tie the auditor rejects keeps the audit
			// from PASSing, and the case below is what that looks like.
			name:       "a tie alone is not the gate's demotion reason",
			final:      `Final Answer: **X** (tie: "Y" - both fit)`,
			audit:      &GateAuditRecord{Suspects: []int{0}, Passed: true},
			wantDemote: false,
		},
		{
			name:       "no audit record at all",
			final:      "Final Answer: **X**",
			audit:      nil,
			wantDemote: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, reason := governAnswerLabel(tc.final, tc.audit)
			if demoted := strings.Contains(got, "Guessed Answer"); demoted != tc.wantDemote {
				t.Errorf("demoted = %v, want %v (%q)", demoted, tc.wantDemote, got)
			}
			if (reason != "") != tc.wantDemote {
				t.Errorf("reason = %q, want a reason only when it rewrote", reason)
			}
			// The value is never touched by label governance, and the label still
			// reads: the note rides the same line as the value without hiding it.
			if !hasAnswerLine(got) {
				t.Errorf("governance lost the answer line: %q", got)
			}
			if !strings.Contains(got, "**X**") {
				t.Errorf("governance touched the value: %q", got)
			}
			if tc.wantDemote && !strings.Contains(got, "assumption:") {
				t.Errorf("a demotion must carry its reason: %q", got)
			}
		})
	}

	// A tie already wearing the Guessed label has nothing to rewrite: returning a
	// reason would log a demotion that never happened.
	got, reason := governAnswerLabel(`Guessed Answer: **X** (tie: "Y" - both fit)`, &GateAuditRecord{Suspects: []int{0}, Passed: true})
	if reason != "" {
		t.Errorf("reason = %q, want none — the label was already Guessed", reason)
	}
	if !strings.Contains(got, "tie:") {
		t.Errorf("the clause must survive governance untouched: %q", got)
	}
}

// TestIsCitationShapedSnippet pins the three signals that separate a
// bibliographic entry from prose that merely mentions a work.
func TestIsCitationShapedSnippet(t *testing.T) {
	cases := []struct {
		name   string
		snip   string
		want   bool
		reason string
	}{
		{
			name: "finding-aid Sources entry",
			snip: "Hodgson, Barbara. Opium: A Portrait of the Heavenly Demon. San Francisco: Chronicle Books. 1999.",
			want: true,
		},
		{
			name: "entry with a comma before the year",
			snip: "Hodgson, Barbara. In the Arms of Morpheus: The Tragic History of Laudanum, Morphine, and Patent Medicines. Vancouver: Greystone Books, 2001.",
			want: true,
		},
		{
			// Prose that satisfies the opener and the closing year but has no
			// `Author. Title.` skeleton: one sentence, not an entry.
			name: "prose mentioning an author and a year",
			snip: "Hodgson, Barbara wrote Opium in 1999.",
			want: false,
		},
		{
			name: "the scope sentence that names both books",
			snip: "Many of the items were reproduced in Hodgson's non-fiction publications In the Arms of Morpheus: The Tragic History of Laudanum, Morphine and Patent Medicines (2001) and Opium: A Portrait of the Heavenly Demon (1999).",
			want: false,
		},
		{
			name: "the publisher blurb",
			snip: "In the Arms of Morpheus is the shocking story of how a simple but bewitching substance touted as a miracle drug enslaved unwitting generations, 1901 included.",
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isCitationShapedSnippet(tc.snip); got != tc.want {
				t.Errorf("isCitationShapedSnippet(%q) = %v, want %v %s", tc.snip, got, tc.want, tc.reason)
			}
		})
	}
}

// TestCitationOnlyGroundings pins the mechanical sibling check on the shape the
// passing q221 deliveries actually had (y1): both family members "supported" by
// bibliographic entries from the SAME Sources chunk, decided afterwards by title
// preference. The auditor is asked to catch this and often does — this check does
// not depend on the round.
func TestCitationOnlyGroundings(t *testing.T) {
	citationB := `Tested: "Opium: A Portrait of the Heavenly Demon" (1999) - clue [1, 2, 3] supported (doc: 59224.md, doc_id: e66d61a91653453d8ad81f9e6c65bf52, chunk_id: a8d8adae46b0e123, snippet: "Hodgson, Barbara. Opium: A Portrait of the Heavenly Demon. San Francisco: Chronicle Books. 1999.")`
	citationA := `Tested: "In the Arms of Morpheus: The Tragic History of Laudanum, Morphine and Patent Medicines" (2001) - also supported (doc: 59224.md, doc_id: e66d61a91653453d8ad81f9e6c65bf52, chunk_id: a8d8adae46b0e123, snippet: "Hodgson, Barbara. In the Arms of Morpheus: The Tragic History of Laudanum, Morphine, and Patent Medicines. Vancouver: Greystone Books, 2001.")`

	got := citationOnlyGroundings(strings.Join([]string{"## Candidate Matrix", "### Sub-question 4: the book", citationB, citationA, `Retained: "Opium: A Portrait of the Heavenly Demon"`}, "\n"))
	if len(got) != 2 {
		t.Fatalf("citationOnlyGroundings = %v, want both candidate lines", got)
	}
	for _, want := range []string{"Opium: A Portrait of the Heavenly Demon", "In the Arms of Morpheus", "a8d8adae46b0e123"} {
		joined := strings.Join(got, " | ")
		if !strings.Contains(joined, want) {
			t.Errorf("the rejection must name %q, got %v", want, got)
		}
	}
	// The directive must send the producer to a chunk that STATES the claim, and
	// must offer the sibling/tie outcomes rather than a title preference.
	dir := citationGroundingDirective(got)
	for _, want := range []string{"BIBLIOGRAPHIC ENTRY", "TEXT states the claim", "DECLARED TIE", "states a sibling's"} {
		if !strings.Contains(dir, want) {
			t.Errorf("directive must carry %q, got:\n%s", want, dir)
		}
	}

	// A properly grounded line is untouched: the blurb STATES the work's subject.
	blurb := `Tested: "In the Arms of Morpheus" - clue [1] supported (doc: 22894.md, chunk_id: f7838c11ad90d0ac, snippet: "In the Arms of Morpheus is the shocking story of how a simple but bewitching substance touted as a miracle drug enslaved unwitting generations. Extracted from opium, the sap of the poppy, Opium was welcomed into the homes of rich and poor alike under the guise of medical use in the form of laudanum and patent medicines.")`
	if got := citationOnlyGroundings(blurb); len(got) != 0 {
		t.Errorf("a chunk that states the subject must pass, got %v", got)
	}
	// A citation carried by a line about the YEAR is not this defect: the entry's
	// own date can genuinely be that line's evidence.
	yearLine := `Tested: 2001 - clue [2] publication year (doc: 59224.md, chunk_id: a8d8adae46b0e123, snippet: "Hodgson, Barbara. In the Arms of Morpheus: The Tragic History of Laudanum, Morphine, and Patent Medicines. Vancouver: Greystone Books, 2001.")`
	if got := citationOnlyGroundings(yearLine); len(got) != 0 {
		t.Errorf("a numeric candidate must be exempt from the citation check, got %v", got)
	}
	// The scope sentence names both books without stating either one's fitness: the
	// audit-side sibling rule owns that shape, not this check.
	scope := `Tested: "Opium: A Portrait of the Heavenly Demon" - book about opium (doc: 59224.md, chunk_id: e383712ea597e4ee, snippet: "Many of the items were reproduced in Hodgson's non-fiction publications In the Arms of Morpheus: The Tragic History of Laudanum, Morphine and Patent Medicines (2001) and Opium: A Portrait of the Heavenly Demon (1999).")`
	if got := citationOnlyGroundings(scope); len(got) != 0 {
		t.Errorf("a scope sentence is judged by the auditor, not by this check, got %v", got)
	}
}
