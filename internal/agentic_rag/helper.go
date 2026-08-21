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
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/common"

	"go.uber.org/zap"
)

// toolErrorMarker opens the failure notice a tool returns when its
// invocation failed. The model reads the notice (with the root cause and
// severity) and adapts — fix the arguments, switch tools, or finish
// gracefully — while consumeAgentEvents sniffs the same marker to tally
// per-tool failures into the benchmark's run stats.
const toolErrorMarker = "<tool_error"

// toolErrorNotice formats a tool failure as a model-readable result.
func toolErrorNotice(tool string, err error) string {
	msg := err.Error()
	if r := []rune(msg); len(r) > 400 {
		msg = string(r[:400]) + "…"
	}
	return toolErrorXML(tool, "error", msg)
}

// toolErrorXML renders the CANONICAL failure element every tool uses:
//
//	<tool_error tool="NAME" severity="error|warn" key="value">reason</tool_error>
//
// The model reads it and adapts (fix the arguments, switch tools, stop
// retrying a dead backend); consumeAgentEvents tallies the severity="error"
// ones into the benchmark's per-tool failure accounting. severity="warn"
// marks a partial failure the tool already recovered from (e.g. one query
// of several died) — visible, but not an outage. Extra attributes (query,
// failed_queries, ...) carry the failure's scope. The reason is
// XML-escaped; the caller truncates it to keep the result bounded.
func toolErrorXML(tool, severity, msg string, attrs ...[2]string) string {
	var b strings.Builder
	b.WriteString(toolErrorMarker)
	b.WriteString(` tool="`)
	b.WriteString(xmlEscape(tool))
	b.WriteString(`" severity="`)
	b.WriteString(xmlEscape(severity))
	b.WriteString(`"`)
	for _, a := range attrs {
		b.WriteString(` `)
		b.WriteString(a[0])
		b.WriteString(`="`)
		b.WriteString(xmlEscape(a[1]))
		b.WriteString(`"`)
	}
	b.WriteString(">")
	b.WriteString(xmlEscape(msg))
	b.WriteString("</tool_error>")
	return b.String()
}

// guardedToolRun wraps a tool's invokable run. A Go error returned from a
// tool does not reach the model as a result — it aborts the whole ReAct
// loop, so one bad regex or a backend outage ends the turn with nothing
// and the model never learns why. Failures are converted into a
// model-readable <tool_error> result (the loop continues) while the
// operator still gets a warn-level log line with the full error.
func guardedToolRun(ctx context.Context, tool string, run func(context.Context, string) (string, error), args string) (string, error) {
	res, err := run(ctx, args)
	if err == nil {
		return res, nil
	}
	common.WarnCtx(ctx, "agentic_rag: tool failed - converting to model-readable result",
		zap.String("tool", tool), zap.Error(err))
	return toolErrorNotice(tool, err), nil
}

// resolveDatasetScope keeps model-provided dataset ids within the
// conversation's server-bound scope. An omitted request uses the full bound
// scope; an explicit request must be a subset of it.
func resolveDatasetScope(bound, requested []string) ([]string, error) {
	if len(requested) == 0 {
		return bound, nil
	}

	allowed := make(map[string]struct{}, len(bound))
	for _, id := range bound {
		allowed[id] = struct{}{}
	}
	for _, id := range requested {
		if _, ok := allowed[id]; !ok {
			return nil, fmt.Errorf("dataset_id %q is outside the conversation's bound scope", id)
		}
	}
	return requested, nil
}

// clampFloat01 clamps a float into [0, 1], used for similarity weights the model
// may supply out of range.
func clampFloat01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// grepChunksSelectFields is the ES _source fields the retrieval tools request.
// doc_id, page_num_int and chunk_order_int are the mandated reading-order
// identifiers; content_with_weight and the other fields feed the tool output.
// The regexp query matches content_with_weight, so it must stay in the list.
var grepChunksSelectFields = []string{
	"content_with_weight",
	"doc_id",
	"docnm_kwd",
	"kb_id",
	"page_num_int",
	"chunk_order_int",
}

// grepChunksSortFields is the reading-order sort used by grep_chunks and
// list_chunks: by document, then page, then chunk within a page.
var grepChunksSortFields = []string{"doc_id", "page_num_int", "chunk_order_int"}

// readingOrderLess orders chunks by document, then page, then chunk index —
// matching the doc_id / page_num_int / chunk_order_int sort the engine applies.
func readingOrderLess(a, b runtime.RetrievalChunk) bool {
	if a.DocumentID != b.DocumentID {
		return a.DocumentID < b.DocumentID
	}
	if a.PageNum != b.PageNum {
		return a.PageNum < b.PageNum
	}
	return a.ChunkIndex < b.ChunkIndex
}

// snippetContextRunes is THE shared half-window (in runes) that every locate
// tool — grep_chunks, search_bm25_chunks, search_chunks — extends beyond its
// match span when rendering snippets: the window starts snippetContextRunes
// runes before the earliest match and ends snippetContextRunes runes after the
// latest match. One knob governs all three.
const snippetContextRunes = 120

// collapseSpaces normalises a chunk body to single-line form so snippets never
// carry raw newlines regardless of the source formatting.
func collapseSpaces(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	return strings.TrimSpace(s)
}

// clampRuneBounds widens [firstByte,lastByte) by snippetContextRunes on each
// side, converts both byte offsets to rune indexes (snapping inward onto rune
// boundaries), and clamps to the content length. Returns -1,-1 when content is
// empty or the bounds are inverted after clamping.
func clampRuneBounds(content string, firstByte, lastByte int) (int, int) {
	if content == "" || firstByte < 0 || lastByte < firstByte || lastByte > len(content) {
		if content != "" && lastByte > len(content) && firstByte >= 0 && firstByte <= len(content) {
			lastByte = len(content)
		} else if !(content != "" && firstByte >= 0 && firstByte <= lastByte && lastByte <= len(content)) {
			return -1, -1
		}
	}
	runes := []rune(content)
	start := utf8.RuneCountInString(content[:firstByte]) - snippetContextRunes
	end := utf8.RuneCountInString(content[:lastByte]) + snippetContextRunes
	if start < 0 {
		start = 0
	}
	if end > len(runes) {
		end = len(runes)
	}
	if start >= end {
		return -1, -1
	}
	return start, end
}

// sliceSnippet renders the shared snippet shape: whitespace-collapsed text of
// content from (earliest match − N) to (latest match + N) runes, with leading /
// trailing "..." marking any truncation. Callers locate their own span first
// (regex or literal terms) and hand in its byte bounds.
func sliceSnippet(content string, firstByte, lastByte int) string {
	if content == "" {
		return ""
	}
	start, end := clampRuneBounds(content, firstByte, lastByte)
	if start < 0 {
		return ""
	}
	runes := []rune(content)
	prefix, suffix := "", ""
	if start > 0 {
		prefix = "..."
	}
	if end < len(runes) {
		suffix = "..."
	}
	return prefix + collapseSpaces(string(runes[start:end])) + suffix
}

// regexMatchSpan returns the earliest match start / latest match end byte
// offsets of re over ALL occurrences in content.
func regexMatchSpan(re *regexp.Regexp, content string) (int, int, bool) {
	locs := re.FindAllStringIndex(content, -1)
	if len(locs) == 0 {
		return 0, 0, false
	}
	first, last := locs[0][0], locs[0][1]
	for _, l := range locs[1:] {
		if l[0] < first {
			first = l[0]
		}
		if l[1] > last {
			last = l[1]
		}
	}
	return first, last, true
}

// termMatchSpan is regexMatchSpan's literal-term counterpart: over every
// case-insensitive occurrence of every term it takes the earliest start and the
// latest end, so multi-keyword queries centre the window on the whole hit set.
func termMatchSpan(terms []string, content string) (int, int, bool) {
	lower := strings.ToLower(content)
	first, last := -1, -1
	for _, t := range terms {
		if t == "" {
			continue
		}
		from := 0
		for {
			i := strings.Index(lower[from:], t)
			if i < 0 {
				break
			}
			lo, hi := from+i, from+i+len(t)
			if first < 0 || lo < first {
				first = lo
			}
			if hi > last {
				last = hi
			}
			from = hi
		}
	}
	if first < 0 {
		return 0, 0, false
	}
	return first, last, true
}

// snippetHit couples a located chunk with its pre-rendered snippet line.
type snippetHit struct {
	chunk   runtime.RetrievalChunk
	snippet string
}

// formatLocateResultsXML renders the UNIFIED payload shared by every locate
// tool: one compact vocabulary (<search_results> root with a query echo, chunk
// attributes incl. rank/score, one <match_snippet> per chunk). Roots differ
// from list_chunks' <chunks> by design: these are triage views, not deep reads.
func formatLocateResultsXML(query string, hits []snippetHit) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("<search_results count=\"%d\" query=\"%s\">\n",
		len(hits), xmlEscape(query)))
	for i, h := range hits {
		c := h.chunk
		b.WriteString(fmt.Sprintf(
			"<chunk rank=\"%d\" chunk_id=\"%s\" doc_id=\"%s\" page_num=\"%d\" chunk_index=\"%d\" dataset_id=\"%s\" doc_name=\"%s\" score=\"%.3f\">\n",
			i+1,
			xmlEscape(c.ID), xmlEscape(c.DocumentID), c.PageNum, c.ChunkIndex,
			xmlEscape(c.DatasetID), xmlEscape(c.DocumentName), c.Score,
		))
		if h.snippet != "" {
			b.WriteString(fmt.Sprintf("<match_snippet>%s</match_snippet>\n", xmlEscape(h.snippet)))
		}
		b.WriteString("</chunk>\n")
	}
	if len(hits) == 0 {
		// Zero hits usually means a WORD-FORM mismatch (the corpus says
		// "mineralization" while the query says "mineralizer"), not corpus
		// absence. Surface the retry discipline here so the model sees it at
		// the exact moment it decides what to query next.
		b.WriteString("<hint>0 hits — the corpus may use a different WORD FORM of your terms. Retry with: (1) derivational variants of the rarest term (mineralizer -> mineralization -> mineralize), singular/plural forms; (2) for grep_chunks, the stem plus a trailing wildcard (mineralizer -> mineraliz.*); (3) the single rarest term ALONE instead of a multi-word phrase (a proper noun, procedure name, or unique date). An exact-phrase 0-hit is not evidence that the corpus lacks the answer.</hint>\n")
	}
	b.WriteString("</search_results>")
	return b.String()
}

// dedupStrings returns s in first-occurrence order without duplicates.
func dedupStrings(s []string) []string {
	seen := make(map[string]struct{}, len(s))
	out := make([]string, 0, len(s))
	for _, v := range s {
		if _, dup := seen[v]; dup {
			continue
		}
		seen[v] = struct{}{}
		out = append(out, v)
	}
	return out
}

// formatChunksXML serialises the ordered chunks of a single document into the
// compact XML shape consumed by the model. anchorMeta (may be empty) appends
// anchored-read annotations to the root element. Terminology is uniform with
// the other retrieval tools: dataset_id (dataset), doc_id (document).
// formatChunksXML renders one deep-read result set. notice, when non-empty, is
// emitted as a <notice> element right after the opening tag: it carries
// findings that must NOT abort the caller's turn (e.g. anchors that did not
// resolve), so the model can see and judge them instead of losing the whole
// ReAct turn to an error.
func formatChunksXML(docID string, chunks []runtime.RetrievalChunk, anchorMeta, notice string) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("<chunks doc_id=\"%s\" fetched=\"%d\"%s>\n",
		xmlEscape(docID), len(chunks), anchorMeta))
	if notice != "" {
		b.WriteString(fmt.Sprintf("<notice>%s</notice>\n", xmlEscape(notice)))
	}
	for _, c := range chunks {
		b.WriteString(fmt.Sprintf(
			"<chunk chunk_id=\"%s\" doc_id=\"%s\" page_num=\"%d\" chunk_index=\"%d\" dataset_id=\"%s\" doc_name=\"%s\">\n",
			xmlEscape(c.ID), xmlEscape(c.DocumentID), c.PageNum, c.ChunkIndex,
			xmlEscape(c.DatasetID), xmlEscape(c.DocumentName)))
		if c.Content != "" {
			b.WriteString(fmt.Sprintf("<content>%s</content>\n", xmlEscape(c.Content)))
		}
		b.WriteString("</chunk>\n")
	}
	b.WriteString("</chunks>")
	return b.String()
}
