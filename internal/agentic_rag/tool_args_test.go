package agentic_rag

import (
	"encoding/json"
	"strings"
	"testing"
)

// The payloads below are the shapes real runs logged (see tool_error_samples in
// a run's leaderboard.json), replayed through the tools' own argument structs.

func TestFlexIntAcceptsNumbersAndStrings(t *testing.T) {
	var args thinkArgs
	payload := `{"thought":"compare the two candidates","next_thought_needed":true,` +
		`"thought_number":"2","total_thoughts":"3 thoughts"}`
	if err := json.Unmarshal([]byte(payload), &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if int(args.ThoughtNumber) != 2 || int(args.TotalThoughts) != 3 {
		t.Fatalf("want 2/3, got %d/%d", args.ThoughtNumber, args.TotalThoughts)
	}
	if !args.NextThoughtNeeded || args.Thought != "compare the two candidates" {
		t.Fatalf("unexpected args: %+v", args)
	}
}

func TestFlexIntEmptyAndNullAreZero(t *testing.T) {
	var args thinkArgs
	if err := json.Unmarshal([]byte(`{"thought":"x","thought_number":"","total_thoughts":null}`), &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if args.ThoughtNumber != 0 || args.TotalThoughts != 0 {
		t.Fatalf("want 0/0, got %d/%d", args.ThoughtNumber, args.TotalThoughts)
	}
}

func TestFlexStringsAcceptsListShapes(t *testing.T) {
	// Logged verbatim: cannot unmarshal object into Go struct field
	// listChunksArgs.anchor_chunk_ids of type []string
	for name, payload := range map[string]string{
		"declared array":   `{"doc_id":"abc","anchor_chunk_ids":["c1","c2"]}`,
		"index keyed":      `{"doc_id":"abc","anchor_chunk_ids":{"0":"c1","1":"c2"}}`,
		"one key wrapping": `{"doc_id":"abc","anchor_chunk_ids":{"anchor_chunk_ids":["c1","c2"]}}`,
		"single value":     `{"doc_id":"abc","anchor_chunk_ids":{"chunk_id":"c1"}}`,
		"bare string":      `{"doc_id":"abc","anchor_chunk_ids":"c1"}`,
	} {
		var args listChunksArgs
		if err := json.Unmarshal([]byte(payload), &args); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(args.AnchorChunkIDs) == 0 {
			t.Fatalf("%s: anchor ids not recovered: %+v", name, args)
		}
	}
}

func TestFlexStringsKeepsNaturalKeyOrder(t *testing.T) {
	var args listChunksArgs
	payload := `{"doc_id":"abc","anchor_chunk_ids":{"10":"c10","9":"c9","2":"c2"}}`
	if err := json.Unmarshal([]byte(payload), &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := []string{"c2", "c9", "c10"}
	for i := range want {
		if args.AnchorChunkIDs[i] != want[i] {
			t.Fatalf("order: got %v want %v", args.AnchorChunkIDs, want)
		}
	}
}

func TestFlexStringsRejectsNestedObjects(t *testing.T) {
	var args searchBm25ChunksArgs
	if err := json.Unmarshal([]byte(`{"queries":{"a":{"b":1}}}`), &args); err == nil {
		t.Fatal("an object whose values are not scalars must still fail")
	}
}

func TestFlexPlanStepsAcceptsObjectShapes(t *testing.T) {
	// Logged verbatim: cannot unmarshal object into Go struct field
	// todoWriteArgs.steps of type []agentic_rag.planStep
	for name, payload := range map[string]string{
		"declared array": `{"task":"t","steps":[{"id":"1","description":"locate the film","status":"completed"}]}`,
		"index keyed":    `{"task":"t","steps":{"1":{"description":"locate the film"},"2":{"description":"read the credits"}}}`,
		"single object":  `{"task":"t","steps":{"description":"locate the film"}}`,
		"wrapped list":   `{"task":"t","steps":{"steps":[{"description":"locate the film"}]}}`,
		"aliased fields": `{"task":"t","steps":[{"step":"locate the film","state":"done"}]}`,
		"bare string":    `{"task":"t","steps":"locate the film"}`,
		"string list":    `{"task":"t","steps":["locate the film","read the credits"]}`,
	} {
		var args todoWriteArgs
		if err := json.Unmarshal([]byte(payload), &args); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(args.Steps) == 0 || strings.TrimSpace(args.Steps[0].Description) == "" {
			t.Fatalf("%s: steps not recovered: %+v", name, args.Steps)
		}
	}
}

func TestFlexPlanStepsKeepsKeyAsID(t *testing.T) {
	var args todoWriteArgs
	if err := json.Unmarshal([]byte(`{"task":"t","steps":{"10":{"description":"tenth"},"9":{"description":"ninth"}}}`), &args); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(args.Steps) != 2 || args.Steps[0].Description != "ninth" || args.Steps[0].ID != "9" {
		t.Fatalf("want the numeric key order and the key as id, got %+v", args.Steps)
	}
}

func TestFlexPlanStepsRequiresADescription(t *testing.T) {
	var args todoWriteArgs
	if err := json.Unmarshal([]byte(`{"task":"t","steps":[{"status":"pending"}]}`), &args); err == nil {
		t.Fatal("a step without a description must still fail")
	}
}

// Tolerance is opt-in per field: fields left as plain Go types keep failing
// loudly, so the schema still means what it says.
func TestFieldsWithoutFlexibleTypesStillReject(t *testing.T) {
	var args thinkArgs
	if err := json.Unmarshal([]byte(`{"thought":{"text":"nested"}}`), &args); err == nil {
		t.Fatal("a plain string field must still reject an object")
	}
	var search searchChunksArgs
	if err := json.Unmarshal([]byte(`{"queries":["a"],"similarity_threshold":"0.7"}`), &search); err == nil {
		t.Fatal("a plain float field must still reject a string")
	}
}

func TestNormalizeRegexPatternUnicodeEscapes(t *testing.T) {
	// Logged verbatim from a failing grep_chunks call.
	got := normalizeRegexPattern(`1[\u2013\-]3 feet|1 to 3 feet`)
	want := "1[–\\-]3 feet|1 to 3 feet"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
	if _, err := json.Marshal(got); err != nil {
		t.Fatalf("pattern must stay JSON-encodable: %v", err)
	}
	if plain := normalizeRegexPattern(`alpha|beta`); plain != "alpha|beta" {
		t.Fatalf("a pattern without escapes must be untouched, got %q", plain)
	}
}
