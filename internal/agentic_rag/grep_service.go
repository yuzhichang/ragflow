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
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"ragflow/internal/agent/runtime"
	"ragflow/internal/dao"
	"ragflow/internal/engine"
	enginetypes "ragflow/internal/engine/types"
)

// Elasticsearch it pushes the regex down to a native `regexp` query; on engines
// without native regex support it falls back to a broad recall + in-memory RE2
// filter. It is stateless and safe to share across goroutines.
type GrepAdapter struct {
	docEngine engine.DocEngine
}

// NewGrepAdapter wraps a doc engine behind the GrepService interface.
func NewGrepAdapter(docEngine engine.DocEngine) *GrepAdapter {
	return &GrepAdapter{docEngine: docEngine}
}

// regexpSearchable is the narrow engine capability for native regex search.
// *elasticsearch.Engine implements it; other engines do not.
type regexpSearchable interface {
	SearchByRegexp(ctx context.Context, req *enginetypes.RegexpSearchRequest) (*enginetypes.SearchResult, error)
}

// ListByDocIDs reads the full original chunks belonging to the given document
// ids, skipping graph (relation/entity/location) chunks so callers get the
// actual document text. It backs the list_chunks deep-read runtime.
// Because content_with_weight is now a searchable keyword field, it reads the
// docs' chunks directly via the regexp pushdown (match-all "." scoped by
// doc_id), no longer needing a content_ltks-based general recall.
func (g *GrepAdapter) ListByDocIDs(ctx context.Context, req runtime.GrepRequest) ([]runtime.RetrievalChunk, error) {
	if g == nil || g.docEngine == nil {
		return nil, runtime.ErrGrepServiceMissing
	}
	docIDs := nonEmptyStrings(req.DocScope)
	if len(docIDs) == 0 {
		return nil, nil
	}
	limit := req.Limit
	if limit <= 0 {
		limit = grepChunksDefaultLimit
	}
	offset := max(req.Offset, 0)

	// Use a derived engine context with no per-query timeout, same rationale
	// as Grep.
	ectx, ecancel := engineCallContext(ctx)
	defer ecancel()

	se, ok := g.docEngine.(regexpSearchable)
	if !ok {
		return nil, runtime.ErrRegexpNotSupported
	}
	// Match-all on the (now keyword) content_with_weight, scoped to the docs,
	// restricted to ordinary text chunks (available_int=1, no compile_kwd). When
	// the caller requests a sort (e.g. reading order), push it down so ES applies
	// offset/limit over the deterministically-ordered set — no over-fetching.
	res, err := se.SearchByRegexp(ectx, &enginetypes.RegexpSearchRequest{
		TenantID:     req.TenantID,
		KbIDs:        req.DatasetIDs,
		Offset:       offset,
		Limit:        limit,
		Pattern:      ".*",
		Sort:         sortExprFromFields(req.Sort), // reading order: doc_id, page_num_int, chunk_order_int
		SelectFields: req.SelectFields,
		Filter: map[string]interface{}{
			"available_int": 1,
			"doc_id":        docIDs,
			"must_not":      map[string]interface{}{"exists": "compile_kwd"},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("list_chunks: deep-read recall: %w", err)
	}

	out := make([]runtime.RetrievalChunk, 0, limit)
	for _, raw := range res.Chunks {
		content := contentWithWeightFromRaw(raw)
		if content == "" || isGraphChunkContent(content) {
			continue
		}
		out = append(out, runtime.RetrievalChunk{
			ID:           runtime.FirstStringFromMap(raw, "id", "_id"),
			Content:      content,
			DocumentID:   runtime.StringFromMap(raw, "doc_id"),
			DocumentName: runtime.StringFromMap(raw, "docnm_kwd"),
			DatasetID:    runtime.StringFromMap(raw, "kb_id"),
			ChunkIndex:   runtime.IntFromMap(raw, "chunk_order_int"),
			PageNum:      runtime.IntFromMap(raw, "page_num_int"),
		})
		if len(out) >= limit {
			break
		}
	}
	return out, nil
}

// resolveDocumentOwner looks up a document row by id via MySQL. It is a
// package-level variable so unit tests can stub ownership without a database;
// the production default reads through the DAO singleton.
var resolveDocumentOwner = func(ctx context.Context, docID string) (string, error) {
	if dao.DB == nil {
		return "", fmt.Errorf("resolveDocDatasetID: database not initialised")
	}
	doc, err := dao.NewDocumentDAO().GetByID(ctx, dao.DB, docID)
	if err != nil {
		return "", err
	}
	if doc == nil {
		return "", fmt.Errorf("document %q not found", docID)
	}
	return doc.KbID, nil
}

// ResolveDocDatasetID returns the knowledge-base id that owns the document in
// req.DocScope, restricted to req.DatasetIDs. It resolves via MySQL (the document
// table stores the authoritative kb_id) rather than an engine search, and backs
// list_chunks so the deep-read is scoped to the document's own dataset instead of
// the whole bound set. It returns an error when the document does not exist or
// belongs to a dataset outside the bound scope.
func (g *GrepAdapter) ResolveDocDatasetID(ctx context.Context, req runtime.GrepRequest) (string, error) {
	docIDs := nonEmptyStrings(req.DocScope)
	if len(docIDs) == 0 {
		return "", fmt.Errorf("resolveDocDatasetID: no doc_id given")
	}
	if len(docIDs) != 1 {
		return "", fmt.Errorf("resolveDocDatasetID: expected exactly one doc_id, got %d", len(docIDs))
	}
	kbID, err := resolveDocumentOwner(ctx, docIDs[0])
	if err != nil {
		return "", fmt.Errorf("list_chunks: resolve owning dataset: %w", err)
	}
	if kbID == "" {
		return "", fmt.Errorf("list_chunks: document %q has no kb_id", docIDs[0])
	}
	// Enforce the bound scope: a doc whose owning dataset is not in the bound set
	// must not be deep-read, even if the model guessed a valid-looking doc_id.
	if !containsString(req.DatasetIDs, kbID) {
		return "", fmt.Errorf("list_chunks: document %q belongs to dataset %q outside the bound dataset scope", docIDs[0], kbID)
	}
	return kbID, nil
}

// ListDocumentChunkIndex returns a lightweight reading-order index of ALL
// ordinary-text chunks in req.DocScope: each entry carries only the chunk id
// and its chunk_order_int (Content stays empty) — Pass 1 of list_chunks'
// two-phase read. No limit is applied: neighbor math needs the complete set.
func (g *GrepAdapter) ListDocumentChunkIndex(ctx context.Context, req runtime.GrepRequest) ([]runtime.RetrievalChunk, error) {
	return g.fetchScoped(ctx, req, []string{"id", "chunk_order_int"}, false)
}

// FetchChunksByID returns the full original text of the chunks named in
// req.ChunkScope, in reading order, skipping knowledge-compiled products and
// graph relation/entity payloads — Pass 2 of list_chunks' two-phase read.
func (g *GrepAdapter) FetchChunksByID(ctx context.Context, req runtime.GrepRequest) ([]runtime.RetrievalChunk, error) {
	return g.fetchScoped(ctx, req, grepChunksSelectFields, true)
}

// scopedFetchWindow caps a single match-all scoped recall so pathological
// documents cannot stream unbounded rows; ordinary texts stay far below this.
const scopedFetchWindow = 2000

// fetchScoped issues one regexp-pushdown match-all query over the scoped docs,
// narrowed by req.ChunkScope term filter when non-empty. selectFields controls
// the ES _source payload; withFullContent=false leaves RetrievalChunk.Content
// empty so the index pass stays lightweight. Results come back in reading
// order (doc_id, page_num_int, chunk_order_int).
func (g *GrepAdapter) fetchScoped(
	ctx context.Context, req runtime.GrepRequest,
	selectFields []string, withFullContent bool,
) ([]runtime.RetrievalChunk, error) {
	if g == nil || g.docEngine == nil {
		return nil, runtime.ErrGrepServiceMissing
	}
	docIDs := nonEmptyStrings(req.DocScope)
	chunkIDs := nonEmptyStrings(req.ChunkScope)
	if len(docIDs) == 0 && len(chunkIDs) == 0 {
		return nil, fmt.Errorf("fetchScoped: need a doc_id or at least one chunk_id to scope the recall")
	}

	ectx, ecancel := engineCallContext(ctx)
	defer ecancel()

	se, ok := g.docEngine.(regexpSearchable)
	if !ok {
		return nil, runtime.ErrRegexpNotSupported
	}
	filter := map[string]interface{}{
		"available_int": 1,
		"doc_id":        docIDs,
		"must_not":      map[string]interface{}{"exists": "compile_kwd"},
	}
	if ids := nonEmptyStrings(req.ChunkScope); len(ids) > 0 {
		filter["id"] = ids
	}

	res, err := se.SearchByRegexp(ectx, &enginetypes.RegexpSearchRequest{
		TenantID:     req.TenantID,
		KbIDs:        req.DatasetIDs,
		Offset:       0,
		Limit:        scopedFetchWindow, // index/full reads must see the whole scoped set
		Pattern:      ".*",
		Sort:         sortExprFromFields(grepChunksSortFields),
		SelectFields: selectFields,
		Filter:       filter,
	})
	if err != nil {
		return nil, fmt.Errorf("list_chunks: scoped recall: %w", err)
	}

	out := make([]runtime.RetrievalChunk, 0, len(res.Chunks))
	for _, raw := range res.Chunks {
		content := ""
		if withFullContent {
			content = contentWithWeightFromRaw(raw)
			if content == "" || isGraphChunkContent(content) {
				continue
			}
		}
		out = append(out, runtime.RetrievalChunk{
			ID:           runtime.FirstStringFromMap(raw, "id", "_id"),
			Content:      content,
			DocumentID:   runtime.StringFromMap(raw, "doc_id"),
			DocumentName: runtime.StringFromMap(raw, "docnm_kwd"),
			DatasetID:    runtime.StringFromMap(raw, "kb_id"),
			ChunkIndex:   runtime.IntFromMap(raw, "chunk_order_int"),
			PageNum:      runtime.IntFromMap(raw, "page_num_int"),
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return readingOrderLess(out[i], out[j]) })
	return out, nil
}

// Grep regex-matches chunk content within the given scope.
func (g *GrepAdapter) Grep(ctx context.Context, req runtime.GrepRequest) ([]runtime.RetrievalChunk, error) {
	if g == nil || g.docEngine == nil {
		return nil, runtime.ErrGrepServiceMissing
	}
	if strings.TrimSpace(req.Pattern) == "" {
		return nil, fmt.Errorf("grep: pattern cannot be empty")
	}
	// Validate the regex on the Go side regardless of pushdown path.
	if _, err := regexp.Compile("(?i)" + req.Pattern); err != nil {
		return nil, fmt.Errorf("grep: invalid regex: %w", err)
	}

	limit := req.Limit
	if limit <= 0 {
		limit = grepChunksDefaultLimit
	}

	// Only search ordinary document text chunks: available_int=1 and no
	// compile_kwd (which marks knowledge-compiled products like wiki_page,
	// hypergraph, mindmap, list, timeline). Knowledge products are derived
	// content, not the original document prose the user asked about, and several
	// of them exceed ES's keyword term length limit.
	filter := map[string]interface{}{
		"available_int": 1,
		"must_not":      map[string]interface{}{"exists": "compile_kwd"},
	}
	if len(req.DocScope) > 0 {
		filter["doc_id"] = req.DocScope
	}

	// Use a derived engine context with no per-query timeout (the run-level
	// budget caps the whole run). eino's streaming ReAct can hand the tool an
	// already-canceled context right after the model emits tool_calls; a canceled
	// parent must not kill the engine query, so we fall back to a fresh context
	// in that case.
	ectx, ecancel := engineCallContext(ctx)
	defer ecancel()

	// Path 1: native regex pushdown (Elasticsearch). With content_with_weight
	// mapped as a searchable keyword field, ES regexp matching is the sole path;
	// there is intentionally no in-memory RE2 fallback (a pushdown failure is a
	// real error the caller should surface, not silently "no matches").
	if se, ok := g.docEngine.(regexpSearchable); ok {
		res, err := se.SearchByRegexp(ectx, &enginetypes.RegexpSearchRequest{
			TenantID:     req.TenantID,
			KbIDs:        req.DatasetIDs,
			Limit:        limit,
			Pattern:      req.Pattern,
			Sort:         sortExprFromFields(req.Sort), // same reading order as list_chunks
			SelectFields: req.SelectFields,
			Filter:       filter,
		})
		if err != nil {
			return nil, fmt.Errorf("grep: regexp pushdown failed: %w.%s", err, regexpPushdownHint(req.Pattern))
		}
		return translateGrepChunks(res.Chunks), nil
	}

	// Non-ES engines do not implement regex matching on chunk content; surface
	// an explicit error rather than returning (empty) regex "matches".
	return nil, runtime.ErrRegexpNotSupported
}

// engineCallContext returns a context for a single engine query. It carries no
// per-query timeout — the run-level smartReasoningTimeout in
// internal/service/chat_pipeline.go caps the whole agent run. The only concern
// here is a defensive one: eino's streaming ReAct can hand the tool an
// already-canceled context right after the model emits tool_calls; deriving
// from context.Background() in that case keeps the query alive instead of being
// killed by that streaming artifact. Live parents are honored as-is so genuine
// run-level cancellation still reaches the engine.
func engineCallContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if ctx.Err() == nil {
		return context.WithCancel(ctx)
	}
	return context.WithCancel(context.Background())
}

// translateGrepChunks converts ES regexp-search result maps into RetrievalChunk.
// regexpPushdownHint turns an engine-side regexp rejection into something the
// caller can act on.
//
// The failure is almost never a syntax error: Elasticsearch compiles the
// pattern into an automaton bounded by max_determinized_states, and every
// pattern is already wrapped in `.*(...).*`, so a bounded repetition over CJK
// text determinizes into far more states than the budget allows —
// `Determinizing .*(关羽.{0,20}杀).* would require more than 10000 effort.`
//
// The wording matters more than it looks: a ReAct loop treats a tool error as
// "that query is done" and moves on, and a raw query_shard_exception gives the
// model nothing to correct. In one observed run the pattern was rejected, the
// model retried once with a pattern that matched nothing, and then ended the
// turn having produced no deliverable at all. The hint has to say what to write
// INSTEAD.
func regexpPushdownHint(pattern string) string {
	if m := boundedRepetitionRe.FindString(pattern); m != "" {
		return " Retry WITHOUT the bounded repetition `" + m + "`: the engine compiles a pattern " +
			"into an automaton with a fixed state budget (it already wraps every pattern in " +
			"`.*(...).*`), and repetition over CJK text exhausts it. Join the terms with `.*` " +
			"(e.g. `关羽.*杀`) or list them as alternatives (`关羽.*杀|关公.*斩`)."
	}
	if backreferenceRe.MatchString(pattern) {
		return " Retry WITHOUT backreferences: the engine does not support them — repeat the " +
			"alternation instead (e.g. `(关羽|关公).*斩`)."
	}
	if lookaroundRe.MatchString(pattern) {
		return " Retry WITHOUT lookaround (`(?=`, `(?<!`): the engine does not support it — " +
			"express the constraint as plain terms."
	}
	return " Retry with a CHEAPER pattern: literal terms joined by `.*`, at most 2-3 alternation " +
		"branches, no bounded repetition — this pattern exceeded the engine's automaton state budget."
}

var (
	// bounded repetition: {n}, {n,}, {n,m} — legal in PCRE, rejected by ES.
	boundedRepetitionRe = regexp.MustCompile(`\{\s*\d+\s*(,\s*\d*\s*)?\}`)
	backreferenceRe     = regexp.MustCompile(`\\[1-9]`)
	lookaroundRe        = regexp.MustCompile(`\(\?[=!<]`)
)

func translateGrepChunks(chunks []map[string]interface{}) []runtime.RetrievalChunk {
	out := make([]runtime.RetrievalChunk, 0, len(chunks))
	for _, raw := range chunks {
		out = append(out, runtime.RetrievalChunk{
			ID:           runtime.FirstStringFromMap(raw, "id", "_id"),
			Content:      contentWithWeightFromRaw(raw),
			DocumentID:   runtime.StringFromMap(raw, "doc_id"),
			DocumentName: runtime.StringFromMap(raw, "docnm_kwd"),
			DatasetID:    runtime.StringFromMap(raw, "kb_id"),
			ChunkIndex:   runtime.IntFromMap(raw, "chunk_order_int"),
			PageNum:      runtime.IntFromMap(raw, "page_num_int"),
		})
	}
	return out
}

// contentWithWeightFromRaw returns the chunk's content string, preferring
// content_with_weight and falling back to content.
func contentWithWeightFromRaw(raw map[string]interface{}) string {
	if v := runtime.StringFromMap(raw, "content_with_weight"); v != "" {
		return v
	}
	return runtime.StringFromMap(raw, "content")
}

// sortExprFromFields builds an ascending OrderByExpr from an ordered list of
// field names. It is shared by Grep (grep_chunks) and ListByDocIDs (list_chunks)
// so both push the same reading-order sort down to ES.
func sortExprFromFields(fields []string) *enginetypes.OrderByExpr {
	var expr *enginetypes.OrderByExpr
	for _, field := range nonEmptyStrings(fields) {
		if expr == nil {
			expr = &enginetypes.OrderByExpr{}
		}
		expr.Asc(field)
	}
	return expr
}

// nonEmptyStrings drops empty/whitespace-only strings, preserving order.
func nonEmptyStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if strings.TrimSpace(s) != "" {
			out = append(out, s)
		}
	}
	return out
}

// containsString reports whether s is present in in.
func containsString(in []string, s string) bool {
	for _, v := range in {
		if v == s {
			return true
		}
	}
	return false
}

// isGraphChunkContent reports whether a chunk's content looks like a knowledge
// graph relation/entity/location JSON payload (e.g.
// {"head":"何进","relation_type":"鸩杀","tail":"董太后","type":"relation"}) rather
// than original document text. Deep-read and retrieval tools skip such chunks so
// the model reads actual document prose, not extracted graph triples.
func isGraphChunkContent(content string) bool {
	trimmed := strings.TrimSpace(content)
	if len(trimmed) < 2 || trimmed[0] != '{' || trimmed[len(trimmed)-1] != '}' {
		return false
	}
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(trimmed), &m); err != nil {
		return false
	}
	if t, _ := m["type"].(string); t == "relation" || t == "entity" || t == "location" || t == "dataset_graph" {
		return true
	}
	_, hasHead := m["head"]
	_, hasTail := m["tail"]
	if hasHead || hasTail {
		return true
	}
	_, hasName := m["name"]
	_, hasSubject := m["subject"]
	_, hasPredicate := m["predicate"]
	_, hasObject := m["object"]
	return hasName || hasSubject || hasPredicate || hasObject
}
