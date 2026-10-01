package main

// Structure metrics (port of _block_headers, _delivery_checks, structure_metrics,
// _candidate_keys, _prompt_signatures, _leakage_counts and friends): what the
// deliverable DECLARED, counted from its own markdown. The counts are
// reported, never scored.

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	earlyStopDocs = 10
	docTextLimit  = 400000
	// _DOC_TEXT_LIMIT caps how much of one corpus document is read for the
	// intermediate-node grounding check; a value's distinctive tokens are
	// almost always in the document's opening half.
)

var structureFields = []string{
	"blocks", "titles_anchored", "titles_use_vars", "slot", "kind", "op",
	"target_binds", "binds", "binds_unique", "from_lines", "constraints",
	"satisfied", "from_pointers", "searched", "tested", "eliminated",
	"retained", "derived", "failure", "reasoning_chain", "answer_label",
}

var deliveryTotalsFields = []string{
	"from_entries", "from_pointers", "from_entries_without_pointer",
	"from_pointer_mismatch", "from_unbound", "inlined_candidate",
	"depends_on_written", "searched_lines", "constraints_defined",
	"constraints_duplicated", "blocks_without_constraint",
	"blocks_without_searched", "blocks_with_by_name_query",
	"pivot_missing_blocks", "single_term_searches", "constraint_number_reused",
	"blocks_parsed", "blocks_total", "decomposition_detached_blocks",
	"decomposition_orphan_roots", "decomposition_from_none_after_first",
	"decomposition_prose_block_refs", "leaked_docnames", "leaked_values",
	"example_value_shipped",
}

// The whitespace runs are [^\S\n], never \s: \s* would swallow the newline
// before the heading too, so the slice of a block would begin with blank
// lines and its title would read as empty.
var (
	blockHeadRE    = regexp.MustCompile(`(?m)^[^\S\n]*#{2,4}[^\S\n]*Sub-question[^\S\n]+(\d+)[^\S\n]*:`)
	fromEntryRE    = regexp.MustCompile(`(\?[A-Za-z_][A-Za-z0-9_]*)\s*(?:\(block\s*(\d+)\))?`)
	candidateRE    = regexp.MustCompile(`(?m)^\s*-\s*(?:Tested|Eliminated|Retained):\s*(.*)$`)
	claimLineRE    = regexp.MustCompile(`^\s*-\s*(?:Tested|Eliminated|Retained|Failure):`)
	deemphasisRE   = regexp.MustCompile(`\*+`)
	proseBlockRE   = regexp.MustCompile(`(?i)\bblock\s*\d+\b|the above|previous sub-question|sub-question above`)
	dependsTailRE  = regexp.MustCompile(`(?i)[—\s-]+\s*depends_on\s*:.*$`)
	patternRE      = regexp.MustCompile("[(\"`]([^\"`)]+)")
	titleRE        = regexp.MustCompile(`(?m)^\s*#{2,4}\s*Sub-question\s+\d+\s*:(.*)$`)
	retainedLineRE = regexp.MustCompile(`(?m)^\s*-\s*Retained:\s*(.+)$`)
	derivedLineRE  = regexp.MustCompile(`(?m)^\s*-\s*Derived:\s*(.+)$`)
	failureTypeRE  = regexp.MustCompile(`(?m)^\s*-\s*Failure:\s*([A-Za-z_-]+)`)
	wordRE         = regexp.MustCompile(`[A-Za-z0-9'’-]+`)
	docNameRE      = regexp.MustCompile(`\b\d{3,}\.md\b`)
	exampleValueRE = regexp.MustCompile(`(?:Tested|Retained|Derived|Evidence):\s*([^—\n` + "`" + `/(]{4,80})`)
	valueTokenRE   = regexp.MustCompile(`[a-z0-9]+`)
	bindsNameRE    = regexp.MustCompile(`(?m)^\s*-\s*Binds:\s*([^\s(]+)`)
	slotTagRE      = regexp.MustCompile(`\s*[-–—]?\s*(?:slot|kind)\s*:.*$`)
	titlePrefixRE  = regexp.MustCompile(`^\s*#{2,4}\s*Sub-question\s+[0-9]+\s*:\s*`)
	answerWordRE   = regexp.MustCompile(`[A-Za-z][A-Za-z''’-]+`)
)

// fromEntry is one `From:` entry: its variable and its block pointer.
type fromEntry struct{ name, pointer string }

func splitBlocks(text string) []map[string]any {
	marks := blockHeadRE.FindAllStringSubmatchIndex(text, -1)
	var blocks []map[string]any
	for i, mark := range marks {
		end := len(text)
		if i+1 < len(marks) {
			end = marks[i+1][0]
		}
		body := text[mark[0]:end]
		lines := strings.Split(body, "\n")
		op, binds := "", ""
		hasFrom := false
		assertsCorpus := false
		var searched, constraints, constraintTexts []string
		var constraintDefs [][2]string
		var fromEntries []fromEntry
		for _, line := range lines[1:] {
			stripped := strings.TrimSpace(deemphasisRE.ReplaceAllString(line, ""))
			switch {
			case op == "" && strings.HasPrefix(stripped, "- Op:"):
				op = strings.TrimSpace(stripped[len("- Op:"):])
			case binds == "" && strings.HasPrefix(stripped, "- Binds:"):
				binds = strings.TrimSpace(stripped[len("- Binds:"):])
			case strings.HasPrefix(stripped, "- From:"):
				hasFrom = true
				for _, m := range fromEntryRE.FindAllStringSubmatch(stripped[len("- From:"):], -1) {
					fromEntries = append(fromEntries, fromEntry{m[1], m[2]})
				}
			case strings.HasPrefix(stripped, "- Constraints:"):
				constraintLine := dependsTailRE.ReplaceAllString(stripped, "")
				for _, m := range regexp.MustCompile(`\b(c\d+)\s*=\s*([^;]*)`).FindAllStringSubmatch(constraintLine, -1) {
					constraintDefs = append(constraintDefs, [2]string{m[1], strings.TrimSpace(regexp.MustCompile(`\s+`).ReplaceAllString(m[2], " "))})
				}
				for _, def := range constraintDefs {
					constraints = append(constraints, def[0])
				}
				constraintTexts = append(constraintTexts, constraintLine)
			case strings.HasPrefix(stripped, "- Searched:"):
				searched = append(searched, strings.TrimSpace(stripped[len("- Searched:"):]))
			case claimLineRE.MatchString(stripped):
				assertsCorpus = true
			}
		}
		var candidates []string
		for _, m := range candidateRE.FindAllStringSubmatch(deemphasisRE.ReplaceAllString(body, ""), -1) {
			if head := lineHead(m[1]); head != "" {
				candidates = append(candidates, head)
			}
		}
		title := ""
		if len(lines) > 0 {
			title = strings.TrimSpace(deemphasisRE.ReplaceAllString(lines[0], ""))
		}
		number := mark[1]
		blocks = append(blocks, map[string]any{
			"number": number, "title": title, "op": op, "binds": binds,
			"has_from_line": hasFrom, "from_entries": fromEntries,
			"candidates": candidates, "searched": searched,
			"constraints": constraints, "constraint_defs": constraintDefs,
			"constraint_texts": constraintTexts, "asserts_corpus": assertsCorpus,
		})
	}
	return blocks
}

// candidateKeys ports _candidate_keys: keys under which one candidate counts
// as searched by name. The head is reduced to its own tokens: the whole
// string when it is short, plus every two-word run inside it.
func candidateKeys(head string) []string {
	words := wordRE.FindAllString(strings.ToLower(head), -1)
	var keys []string
	if len(words) >= 2 && len(words) <= 4 {
		keys = append(keys, strings.Join(words, " "))
	}
	for i := 0; i+1 < len(words); i++ {
		keys = append(keys, words[i]+" "+words[i+1])
	}
	var out []string
	for _, key := range keys {
		if len(key) >= 6 {
			out = append(out, key)
		}
	}
	return out
}

// lineHead ports _line_head: the candidate name at the head of a matrix
// line, before its fields.
func lineHead(line string) string {
	cut := len(line)
	lowered := strings.ToLower(line)
	for _, separator := range []string{"—", " - ", "satisfied:", "(doc:", "doc:", "(source:"} {
		if index := strings.Index(lowered, strings.ToLower(separator)); index >= 0 && index < cut {
			cut = index
		}
	}
	head := strings.TrimSpace(line[:cut])
	return strings.Trim(head, "`*\" ")
}

var promptPathEnv = "BENCHMARK_PROMPT_PATH"

// promptSignatures ports _prompt_signatures: the names and document ids the
// PROMPT itself prints in its examples. A canary, not a judgement.
func promptSignatures(path string) map[string]any {
	docnames := map[string]bool{}
	values := map[string]bool{}
	loaded := false
	if path == "" {
		path = os.Getenv(promptPathEnv)
	}
	if path == "" {
		// Walk up from the working directory looking for conf/agentic_rag.yaml
		// (the Python original resolved it from the script's own location).
		wd, _ := os.Getwd()
		dir := wd
		for i := 0; i < 6; i++ {
			candidate := filepath.Join(dir, "conf", "agentic_rag.yaml")
			if fileExists(candidate) {
				path = candidate
				break
			}
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
			dir = parent
		}
	}
	if path != "" {
		if raw, err := os.ReadFile(path); err == nil {
			loaded = true
			for _, name := range docNameRE.FindAllString(string(raw), -1) {
				docnames[name] = true
			}
			for _, match := range exampleValueRE.FindAllStringSubmatch(string(raw), -1) {
				value := strings.Trim(strings.TrimSpace(match[1]), "\"")
				// Only names can leak: a lowercase phrase is schema vocabulary
				// a delivery is SUPPOSED to repeat.
				uppercase := regexp.MustCompile(`[A-Z]`).FindAllString(value, -1)
				if len(value) >= 6 && !strings.HasPrefix(value, "<") && len(uppercase) >= 2 {
					values[value] = true
				}
			}
		}
	}
	return map[string]any{"docnames": docnames, "values": values, "loaded": loaded}
}

// leakageCounts ports _leakage_counts: whether the deliverable reused
// anything the PROMPT prints in its examples.
func leakageCounts(text string, signatures map[string]any) map[string]int {
	if signatures == nil {
		signatures = promptSignatures("")
	}
	docnames := signatures["docnames"].(map[string]bool)
	values := signatures["values"].(map[string]bool)
	answer := regexp.MustCompile(`(?im)^.*(?:Final|Guessed)\s+Answer.*$`).FindAllString(text, -1)
	answerLower := strings.ToLower(strings.Join(answer, "\n"))
	lowered := strings.ToLower(text)
	leakedValues, shipped := 0, 0
	for value := range values {
		needle := strings.ToLower(value)
		if count := strings.Count(lowered, needle); count > 0 {
			leakedValues += count
			if strings.Contains(answerLower, needle) {
				shipped = 1
			}
		}
	}
	leakedDocnames := 0
	for docname := range docnames {
		if strings.Contains(lowered, strings.ToLower(docname)) {
			leakedDocnames++
		}
	}
	return map[string]int{
		"leaked_docnames":       leakedDocnames,
		"leaked_values":         leakedValues,
		"example_value_shipped": shipped,
	}
}

// titleNamesAnchor ports _title_names_anchor: whether a sub-question title
// names an anchor of its own (a number, a capitalised name inside the
// sentence, or a distinctive term). A bare slot tag has none.
func titleNamesAnchor(title string) bool {
	text := titlePrefixRE.ReplaceAllString(title, "")
	text = strings.TrimSpace(slotTagRE.ReplaceAllString(text, ""))
	if regexp.MustCompile(`[0-9]`).MatchString(text) {
		return true
	}
	words := answerWordRE.FindAllString(text, -1)
	if len(words) < 3 {
		return false
	}
	for _, word := range words[1:] {
		if len(word) > 0 && word[:1] != strings.ToLower(word[:1]) {
			return true
		}
	}
	return false
}

// intermediateValues ports _intermediate_values: the entity values a run
// committed to mid-chain (retained candidates and the first clause of every
// Derived line), deduplicated.
func intermediateValues(text string) []string {
	var values []string
	for _, line := range retainedLineRE.FindAllStringSubmatch(text, -1) {
		candidate := lineHead(line[1])
		if candidate != "" && !strings.EqualFold(candidate, "none") {
			values = append(values, candidate)
		}
	}
	for _, line := range derivedLineRE.FindAllStringSubmatch(text, -1) {
		head := strings.TrimSpace(strings.Split(line[1], ";")[0])
		if len(head) >= 3 && len(head) <= 60 {
			values = append(values, head)
		}
	}
	seen := map[string]bool{}
	var unique []string
	for _, value := range values {
		key := strings.ToLower(value)
		if !seen[key] {
			seen[key] = true
			unique = append(unique, value)
		}
	}
	return unique
}

// valueTokens ports _value_tokens: the distinctive tokens of a value (>=4
// chars, not a pure number).
func valueTokens(value string) []string {
	var tokens []string
	for _, token := range valueTokenRE.FindAllString(strings.ToLower(value), -1) {
		if len(token) >= 4 && !isAllDigits(token) {
			tokens = append(tokens, token)
		}
	}
	return tokens
}

func isAllDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return len(s) > 0
}

// docText ports _doc_text: one corpus document's lower-cased text, cached
// across questions.
func docText(docID, corpusDir string, cache map[string]string) string {
	if text, ok := cache[docID]; ok {
		return text
	}
	name := docID
	if !strings.HasSuffix(name, ".md") {
		name += ".md"
	}
	text := ""
	if data, err := os.ReadFile(filepath.Join(corpusDir, name)); err == nil {
		if len(data) > docTextLimit {
			data = data[:docTextLimit]
		}
		text = strings.ToLower(string(data))
	}
	cache[docID] = text
	return text
}

// groundValues ports _ground_values: count how many committed values occur in
// the documents the run retrieved. Returns (grounded, checked).
func groundValues(values []string, docIDs []string, corpusDir string, cache map[string]string) (int, int) {
	if corpusDir == "" || len(values) == 0 || len(docIDs) == 0 {
		return 0, 0
	}
	var union strings.Builder
	for _, docID := range docIDs {
		union.WriteString(docText(docID, corpusDir, cache))
	}
	if union.Len() == 0 {
		return 0, 0
	}
	text := union.String()
	grounded, checked := 0, 0
	for _, value := range values {
		tokens := valueTokens(value)
		if len(tokens) == 0 {
			continue
		}
		checked++
		all := true
		for _, token := range tokens {
			if !strings.Contains(text, token) {
				all = false
				break
			}
		}
		if all {
			grounded++
		}
	}
	return grounded, checked
}

var _ = sort.Strings

// structureMetrics ports structure_metrics: count the structure a
// deliverable DECLARED, from its own text. Every counter is a regex over the
// markdown the producer shipped, so the result is reproducible from
// answers.jsonl alone.
func structureMetrics(answer, questionText string, signatures map[string]any) map[string]any {
	text := thinkRE.ReplaceAllString(answer, "")
	// Field prefixes are matched on a de-emphasised copy: the bolded-header
	// delivery of 2026-09-23 read as zero adopted fields otherwise.
	plain := deemphasisRE.ReplaceAllString(text, "")
	count := func(pattern string) int { return len(regexp.MustCompile(`(?m)`+pattern).FindAllString(plain, -1)) }
	titles := titleRE.FindAllStringSubmatch(text, -1)
	bindsNames := bindsNameRE.FindAllStringSubmatch(plain, -1)
	uniqueBinds := map[string]bool{}
	for _, m := range bindsNames {
		uniqueBinds[strings.ToLower(m[1])] = true
	}
	metrics := map[string]any{
		"blocks":             len(titles),
		"slot":               countPrefix(titles, `slot:\s*[A-Za-z_\[\]]`),
		"titles_anchored":    countTitles(titles, titleNamesAnchor),
		"titles_use_vars":    countTitles(titles, func(t string) bool { return strings.Contains(t, "?") }),
		"kind":               countPrefix(titles, `kind:\s*\S`),
		"op":                 count(`^\s*-\s*Op:\s*[A-Za-z_]`),
		"target_binds":       count(`^\s*-\s*Binds:\s*\?answer\b`),
		"binds":              count(`^\s*-\s*Binds:\s*`),
		"binds_unique":       boolToInt(len(uniqueBinds) == len(bindsNames) && len(bindsNames) > 0),
		"from_lines":         count(`^\s*-\s*From:\s*`),
		"constraints":        count(`^\s*-\s*Constraints:\s*`),
		"satisfied":          count(`satisfied:\s*\[`),
		"depends_on_written": count(`depends_on\s*:`),
		"searched":           count(`^\s*-\s*Searched:\s*`),
		"tested":             count(`^\s*-\s*Tested:\s*`),
		"eliminated":         count(`^\s*-\s*Eliminated:\s*`),
		"retained":           count(`^\s*-\s*Retained:\s*`),
		"derived":            count(`^\s*-\s*Derived:\s*`),
		"failure":            count(`^\s*-\s*Failure:\s*`),
		"reasoning_chain":    boolToInt(regexp.MustCompile(`(?m)^\s*#{1,4}\s*Reasoning Chain\b`).MatchString(text)),
		"answer_label":       boolToInt(regexp.MustCompile(`(?i)\b(?:Final|Guessed)\s+Answer\b`).MatchString(text)),
	}
	var failureTypes []string
	seenTypes := map[string]bool{}
	for _, m := range failureTypeRE.FindAllStringSubmatch(text, -1) {
		t := strings.ToLower(m[1])
		if !seenTypes[t] {
			seenTypes[t] = true
			failureTypes = append(failureTypes, t)
		}
	}
	sort.Strings(failureTypes)
	metrics["failure_types"] = failureTypes
	metrics["values"] = intermediateValues(text)
	for k, v := range deliveryChecks(text, questionText, signatures) {
		metrics[k] = v
	}
	return metrics
}

func countPrefix(titles [][]string, pattern string) int {
	re := regexp.MustCompile(pattern)
	n := 0
	for _, m := range titles {
		if re.MatchString(m[1]) {
			n++
		}
	}
	return n
}

func countTitles(titles [][]string, predicate func(string) bool) int {
	n := 0
	for _, m := range titles {
		if predicate(m[1]) {
			n++
		}
	}
	return n
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

// deliveryChecks ports _delivery_checks: counts for the delivery questions
// that CAN be answered mechanically. Each is membership, not phrasing - and
// they are counted HERE, not enforced in the gate. Read blocks_parsed beside
// every count: a count taken over an unparsed block is not evidence of
// compliance.
func deliveryChecks(text, questionText string, signatures map[string]any) map[string]int {
	blocks := splitBlocks(text)
	bindings := map[string]int{}
	for _, block := range blocks {
		name := strings.ToLower(strings.TrimSpace(strAny(block, "binds", "")))
		if strings.HasPrefix(name, "?") {
			if _, seen := bindings[name]; !seen {
				bindings[name] = strToInt(block["number"])
			}
		}
	}
	// The decomposition is ONE chain: union the blocks each edge joins, then
	// read whether every block belongs to the component that carries ?answer.
	union := map[int]int{}
	for _, block := range blocks {
		union[strToInt(block["number"])] = strToInt(block["number"])
	}
	var unionRoot func(x int) int
	unionRoot = func(x int) int {
		for union[x] != x {
			union[x] = union[union[x]]
			x = union[x]
		}
		return x
	}
	pointers, withoutPointer, mismatch, unbound, inlined := 0, 0, 0, 0, 0
	parsed, searchedLines, withoutSearched, withByName := 0, 0, 0, 0
	pivotMissing, singleTerm, withoutConstraint, fromNoneAfterFirst := 0, 0, 0, 0
	proseRefs := 0
	defined := map[string]int{}
	definitions := map[string]map[string]bool{}
	questionLower := strings.ToLower(questionText)
	for _, block := range blocks {
		title := strAny(block, "title", "")
		number := strToInt(block["number"])
		op := strAny(block, "op", "")
		binds := strAny(block, "binds", "")
		hasFrom := boolOf(block["has_from_line"])
		searched := block["searched"].([]string)
		if op != "" && binds != "" && hasFrom && regexp.MustCompile(`slot:\s*\S`).MatchString(title) {
			parsed++
		}
		if number > 1 && len(block["from_entries"].([]fromEntry)) == 0 {
			// `From` names what a block consumes; the contract allows `none`
			// in the FIRST block only.
			fromNoneAfterFirst++
		}
		proseRefs += len(proseBlockRE.FindAllString(title, -1))
		for _, line := range block["constraint_texts"].([]string) {
			proseRefs += len(proseBlockRE.FindAllString(line, -1))
		}
		for _, entry := range block["from_entries"].([]fromEntry) {
			producer, bound := bindings[strings.ToLower(entry.name)]
			if bound {
				union[unionRoot(number)] = unionRoot(producer)
			}
			if entry.pointer == "" {
				withoutPointer++
				continue
			}
			pointers++
			if !bound {
				unbound++
			} else if producer != parseIntOrDefault(entry.pointer, -1) {
				mismatch++
			}
		}
		for _, def := range block["constraint_defs"].([][2]string) {
			defined[def[0]]++
			if definitions[def[0]] == nil {
				definitions[def[0]] = map[string]bool{}
			}
			definitions[def[0]][strings.ToLower(def[1])] = true
		}
		if len(block["constraints"].([]string)) == 0 {
			withoutConstraint++
		}
		searchedLines += len(searched)
		if boolOf(block["asserts_corpus"]) && len(searched) == 0 {
			withoutSearched++
		}
		recorded := strings.ToLower(strings.Join(searched, " || "))
		candidates := block["candidates"].([]string)
		hasByName := false
		for _, candidate := range candidates {
			for _, key := range candidateKeys(candidate) {
				if strings.Contains(recorded, key) {
					hasByName = true
				}
			}
		}
		switch {
		case hasByName:
			withByName++
		case len(candidates) > 0:
			// A block that tests or retains a candidate owes a query on that
			// candidate's own name; without it, it verified none of them.
			pivotMissing++
		}
		for _, raw := range searched {
			if m := patternRE.FindStringSubmatch(raw); m != nil {
				if words := regexp.MustCompile(`[A-Za-z0-9']+`).FindAllString(m[1], -1); len(words) > 0 && len(words) <= 3 {
					singleTerm++
				}
			}
		}
		for _, candidate := range candidates {
			lowered := strings.ToLower(candidate)
			// A name the QUESTION itself supplies is not an inlined finding.
			if len(lowered) >= 3 && strings.Contains(strings.ToLower(title), lowered) && !strings.Contains(questionLower, lowered) {
				inlined++
			}
		}
	}
	answerBlock, hasAnswerBlock := bindings["?answer"]
	detached := 0
	if hasAnswerBlock {
		main := unionRoot(answerBlock)
		for _, block := range blocks {
			if unionRoot(strToInt(block["number"])) != main {
				detached++
			}
		}
	}
	// A root (`From: none`) is lawful; an ORPHAN ROOT whose variable no block
	// consumes is a second, disconnected question. ?answer is exempt.
	consumedNames := map[string]bool{}
	for _, block := range blocks {
		for _, entry := range block["from_entries"].([]fromEntry) {
			consumedNames[strings.ToLower(entry.name)] = true
		}
	}
	orphanRoots := 0
	for _, block := range blocks {
		binds := strings.ToLower(strings.TrimSpace(strAny(block, "binds", "")))
		if len(block["from_entries"].([]fromEntry)) == 0 && binds != "" && !consumedNames[binds] && binds != "?answer" {
			orphanRoots++
		}
	}
	definedTotal := 0
	for _, count := range defined {
		definedTotal += count
	}
	duplicated := 0
	for _, count := range defined {
		if count > 1 {
			duplicated++
		}
	}
	reused := 0
	for _, texts := range definitions {
		if len(texts) > 1 {
			reused++
		}
	}
	out := map[string]int{
		"decomposition_detached_blocks":       detached,
		"decomposition_orphan_roots":          orphanRoots,
		"decomposition_from_none_after_first": fromNoneAfterFirst,
		"decomposition_prose_block_refs":      proseRefs,
		"from_pointers":                       pointers,
		"from_entries":                        pointers + withoutPointer,
		"from_entries_without_pointer":        withoutPointer,
		"from_pointer_mismatch":               mismatch,
		"from_unbound":                        unbound,
		"inlined_candidate":                   inlined,
		"searched_lines":                      searchedLines,
		"constraints_defined":                 definedTotal,
		"constraints_duplicated":              duplicated,
		"constraint_number_reused":            reused,
		"blocks_without_constraint":           withoutConstraint,
		"blocks_without_searched":             withoutSearched,
		"blocks_with_by_name_query":           withByName,
		"pivot_missing_blocks":                pivotMissing,
		"single_term_searches":                singleTerm,
		"blocks_parsed":                       parsed,
		"blocks_total":                        len(blocks),
	}
	for k, v := range leakageCounts(text, signatures) {
		out[k] = v
	}
	return out
}

func strToInt(v any) int { return asIntDefault(v, 0) }
