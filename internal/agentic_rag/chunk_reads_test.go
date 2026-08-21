package agentic_rag

import "testing"

func TestCountChunkElements(t *testing.T) {
	cases := map[string]int{
		// list_chunks deep read: wrapper + two chunks
		`<chunks doc_id="d1" fetched="2"><chunk chunk_id="c1" doc_id="d1">x</chunk><chunk chunk_id="c2" doc_id="d1">y</chunk></chunks>`: 2,
		// search_bm25 triage view
		`<search_results count="2" query="q"><chunk rank="1" chunk_id="c1" doc_id="d1"><match_snippet>s</match_snippet></chunk><chunk rank="2" chunk_id="c2" doc_id="d2"><match_snippet>t</match_snippet></chunk></search_results>`: 2,
		// empty result set: only the wrapper
		`<chunks doc_id="d1" fetched="0"></chunks>`:             0,
		`<search_results count="0" query="q"></search_results>`: 0,
		``: 0,
		`no chunks here, just <other tag="x"> text`: 0,
	}
	for out, want := range cases {
		if got := countChunkElements(out); got != want {
			t.Fatalf("countChunkElements(%q) = %d, want %d", out, got, want)
		}
	}
}

func TestChunkReadLedger(t *testing.T) {
	var nilLedger *chunkReadLedger
	nilLedger.AddDeep(5) // nil-safe
	nilLedger.AddShallow(5)
	deep, shallow := nilLedger.Snapshot()
	if deep != 0 || shallow != 0 {
		t.Fatalf("nil ledger snapshot = (%d, %d)", deep, shallow)
	}

	l := NewChunkReadLedger()
	l.AddDeep(2)
	l.AddDeep(3)
	l.AddShallow(7)
	l.AddShallow(0) // ignored
	l.AddDeep(-1)   // ignored
	deep, shallow = l.Snapshot()
	if deep != 5 || shallow != 7 {
		t.Fatalf("ledger = (deep=%d, shallow=%d), want (5, 7)", deep, shallow)
	}
}
