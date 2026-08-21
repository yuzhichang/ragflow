package agentic_rag

import (
	"strings"
	"testing"
)

func TestFinalAnswerValueTwoLineForm(t *testing.T) {
	// q1228's shape: the heading alone on its line, the value under it.
	final := "## Reasoning Chain\n...\n## Final Answer\n**Bhowani Junction**\n"
	if got := finalAnswerValue(final); got != "Bhowani Junction" {
		t.Fatalf("two-line value = %q, want Bhowani Junction", got)
	}
	// Single-line form keeps working.
	if got := finalAnswerValue("Final Answer: **Casey Means**"); got != "Casey Means" {
		t.Fatalf("single-line value = %q", got)
	}
	// A heading with a non-answer body under it is NOT a value.
	if got := finalAnswerValue("## Final Answer\nSome prose paragraph without bold markers"); got != "" {
		t.Fatalf("prose under the heading must not parse as a value, got %q", got)
	}
}

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

func TestAnswerValueIsGrounded(t *testing.T) {
	hay := "<chunk chunk_id=\"c1\">Boston is the capital of Massachusetts.</chunk>\n" +
		"<chunk chunk_id=\"c2\">The Wexford Ballast Bank is a landmark.</chunk>"
	cases := []struct {
		value string
		want  bool
	}{
		{"Boston", true},                   // verbatim
		{"**Boston**", true},               // emphasis folded
		{"the Wexford Ballast Bank", true}, // case/whitespace folded
		{"Joe Ricketts", false},            // synthesized name
		{"Campion School, Mumbai", false},  // wrong entity
		{"27", true},                       // numeric: exempt from the check
		{"109", true},                      // numeric: exempt
		{"", true},                         // empty: nothing to check
	}
	for _, tc := range cases {
		if got := answerValueIsGrounded(tc.value, hay); got != tc.want {
			t.Errorf("answerValueIsGrounded(%q) = %v, want %v", tc.value, got, tc.want)
		}
	}
	// An empty haystack SKIPS the check (no tool results recorded → the gate
	// cannot tell grounded from synthesized, so it must not fail the answer).
	if !answerValueIsGrounded("Boston", "") {
		t.Fatal("empty haystack must skip the grounding check")
	}
}

func TestCountChunkElementsStillWorks(t *testing.T) {
	if n := countChunkElements("<chunks><chunk a=\"1\">x</chunk></chunks>"); n != 1 {
		t.Fatalf("count = %d, want 1", n)
	}
}
