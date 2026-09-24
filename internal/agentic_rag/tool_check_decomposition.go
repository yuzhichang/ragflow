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

	einotool "github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/schema"
)

// checkDecompositionToolName is the deterministic structural check of a DRAFT
// decomposition - the block plan, written BEFORE any search, so its input holds
// only the block titles and their Op / Binds / From / Constraints header lines.
// No `Searched`, `Tested`, `Eliminated`, `Retained`, `Derived` or `Failure` line
// exists at that moment and this tool deliberately checks none of them.
//
// Why it is code and not one more audit opinion: across the archived runs the
// auditor diagnosed these shape defects only partially (in one run a title that
// inlined another block's candidate and a reserved name bound by two blocks both
// went unreported while thirty field-integrity findings crowded the same
// verdict), and no run ever repaired one (q875 spent six audit rounds on a shape
// defect and shipped anyway). Every check below is a membership test over header
// text, so it holds whatever the wording - the property the gate's own note
// demands of anything that carries an opinion.
//
// The findings reuse the auditor's own fail strings verbatim, so a finding here
// is a finding there; where the auditor's string carries no locator, the locator
// is appended in parentheses so the audited prefix is still a substring of it.
const checkDecompositionToolName = "check_decomposition"

const checkDecompositionToolDescription = `Mechanically checks a DRAFT decomposition (the block plan, written before any search) and returns the structural findings it contains.

Call this IMMEDIATELY after writing the decomposition and BEFORE any retrieval: at that moment the plan is only block titles plus their Op / Binds / From / Constraints lines, and a defect in it costs a whole run - a name bound twice voids every later reference to it, and a title that inlines a candidate freezes a guess into the plan.

Input:
- question: the user's question, verbatim (the titles' anchors are checked against it).
- plan: the drafted decomposition exactly as it will appear in the matrix, with no evidence lines (no Searched/Tested/Eliminated/Retained/Derived/Failure).

It reports the same fail strings the answer auditor uses. Fix every finding and call it again; a plan that returns OK is ready for retrieval.`

type checkDecompositionArgs struct {
	Question string `json:"question"`
	Plan     string `json:"plan"`
}

// CheckDecompositionTool is a stateless, deterministic parser of the block plan.
type CheckDecompositionTool struct{}

func NewCheckDecompositionTool() *CheckDecompositionTool { return &CheckDecompositionTool{} }

// Info returns the tool's metadata for the chat model.
func (t *CheckDecompositionTool) Info(_ context.Context) (*schema.ToolInfo, error) {
	return &schema.ToolInfo{
		Name: checkDecompositionToolName,
		Desc: checkDecompositionToolDescription,
		ParamsOneOf: schema.NewParamsOneOfByParams(map[string]*schema.ParameterInfo{
			"question": {
				Type: schema.String, Required: true,
				Desc: "The user's question, verbatim.",
			},
			"plan": {
				Type: schema.String, Required: true,
				Desc: "The drafted decomposition: every `### Sub-question N:` title plus its `- Op:`, `- Binds:`, `- From:` and `- Constraints:` lines. No evidence lines.",
			},
		}),
	}, nil
}

// InvokableRun parses the plan and reports its structural findings; errors are
// converted to model-readable text by guardedToolRun, like every other tool.
func (t *CheckDecompositionTool) InvokableRun(ctx context.Context, argumentsInJSON string, _ ...einotool.Option) (string, error) {
	return guardedToolRun(ctx, checkDecompositionToolName, t.invokableRun, argumentsInJSON)
}

func (t *CheckDecompositionTool) invokableRun(_ context.Context, argumentsInJSON string) (string, error) {
	var args checkDecompositionArgs
	if err := json.Unmarshal([]byte(argumentsInJSON), &args); err != nil {
		return "", fmt.Errorf("check_decomposition: parse arguments: %w", err)
	}
	if strings.TrimSpace(args.Plan) == "" {
		return "", fmt.Errorf("check_decomposition: plan must be a non-empty string")
	}
	findings := checkDecomposition(args.Question, args.Plan)
	if len(findings) == 0 {
		return "OK - no structural finding: the plan is one DAG, every block reaches the ?answer block, every title carries an anchor of the question, and the constraint numbers are a partition. Start retrieving.", nil
	}
	if len(findings) > 24 {
		findings = append(findings[:24], fmt.Sprintf("... and %d more finding(s)", len(findings)-24))
	}
	var b strings.Builder
	b.WriteString("The plan is NOT ready for retrieval. Fix every finding below (they use the auditor's own fail strings) and call this tool again:\n")
	for i, f := range findings {
		fmt.Fprintf(&b, "%d. %s\n", i+1, f)
	}
	return b.String(), nil
}

// ── the check itself ────────────────────────────────────────────────────────

// decompositionBlock is one `### Sub-question N:` block reduced to its header
// facts. Evidence lines are ignored on purpose: the draft has none yet.
type decompositionBlock struct {
	number         string
	title          string
	op             string
	binds          string
	hasOp          bool
	slot           string
	hasBinds       bool
	hasFrom        bool
	hasConstraints bool
	trailingField  bool
	fromNone       bool
	fromEntries    []decompositionEdge
	constraintDefs []decompositionConstraint
}

// decompositionEdge is one `From:` entry: the variable it consumes, the block
// it points at (when the entry carries a pointer) and whether it carries one.
type decompositionEdge struct {
	varName    string
	block      string
	hasPointer bool
}

type decompositionConstraint struct {
	id       string // c<k>
	claim    string // the claim text as written, for the candidate-inlining test
	claimKey string // whitespace-collapsed lower-case claim, for the partition-vs-restart test
}

var (
	decompositionHeadRe    = regexp.MustCompile(`(?m)^[^\S\n]*#{2,6}[^\S\n]*Sub-question[^\S\n]*(\d+)[^\S\n]*:?[^\S\n]*(.*)$`)
	decompositionOpRe      = regexp.MustCompile(`(?m)^[^\S\n]*-[^\S\n]*Op:[^\S\n]*(.*)$`)
	decompositionBindsRe   = regexp.MustCompile(`(?m)^[^\S\n]*-[^\S\n]*Binds:[^\S\n]*(\?[A-Za-z0-9_]+)`)
	decompositionFromRe    = regexp.MustCompile(`(?m)^[^\S\n]*-[^\S\n]*From:[^\S\n]*(.*)$`)
	decompositionConRe     = regexp.MustCompile(`(?m)^[^\S\n]*-[^\S\n]*Constraints:[^\S\n]*(.*)$`)
	decompositionCdefRe    = regexp.MustCompile(`\b(c\d+)\s*=\s*([^;]*)`)
	decompositionFromEntRe = regexp.MustCompile(`(\?[A-Za-z0-9_]+)([^,;]*)`)
	decompositionPointerRe = regexp.MustCompile(`\(\s*block[^\S\n]*(\d+)\s*\)`)
	// A cross-block reference written as a block number or as prose: the entry
	// fields are exempt (`From: ?school (block 1)` legitimately carries one), so
	// only titles and constraint text are scanned.
	decompositionProseRefRe = regexp.MustCompile(`(?i)\bblock[^\S\n]*\d+\b|\bthe above\b|\bthe previous block\b|\bthat block\b|\bthe earlier block\b`)
	decompositionSlotRe     = regexp.MustCompile(`(?i)slot:[^\S\n]*([A-Za-z_]+)`)
	decompositionWordRe     = regexp.MustCompile(`[A-Za-z][A-Za-z0-9'_-]{2,}`)
	decompositionYearRe     = regexp.MustCompile(`\b(?:1[5-9]\d\d|20\d\d)\b`)
)

// checkDecomposition returns the plan's findings, sorted for a stable report.
func checkDecomposition(question, plan string) []string {
	// Markdown bold is stripped before any prefix match: a run that wrote
	// `- **Op:**` emptied every header-derived metric it had, so tolerating the
	// emphasis is a precondition of reading a plan at all.
	plain := strings.ReplaceAll(plan, "*", "")
	questionTokens := decompositionTokens(question)
	questionYears := map[string]bool{}
	for _, y := range decompositionYearRe.FindAllString(question, -1) {
		questionYears[y] = true
	}

	blocks, findings := decompositionParse(plain)
	if len(blocks) == 0 {
		return []string{"schema integrity: the plan declares no `### Sub-question N:` block - write the decomposition before retrieving"}
	}

	// (a) header shape, then the variable bindings: names are unique
	// question-wide and `?answer` is bound by exactly one block.
	byVar := map[string][]*decompositionBlock{}
	var answerBlocks []*decompositionBlock
	for _, b := range blocks {
		if !b.hasBinds || b.binds == "" {
			findings = append(findings, fmt.Sprintf("schema integrity: block header is missing (Binds, block %s)", b.number))
			continue
		}
		byVar[b.binds] = append(byVar[b.binds], b)
		if strings.EqualFold(b.binds, "?answer") {
			answerBlocks = append(answerBlocks, b)
		}
	}
	varNames := sortedKeys(byVar)
	for _, v := range varNames {
		if len(byVar[v]) < 2 {
			continue
		}
		nums := blockNumbers(byVar[v])
		findings = append(findings, fmt.Sprintf("schema integrity: variable is bound twice (%s, blocks %s)", v, strings.Join(nums, ", ")))
	}
	if len(answerBlocks) == 0 {
		findings = append(findings, "schema integrity: no block binds ?answer")
	}

	// (b) `From:` entries: the variable must be bound somewhere, the pointer must
	// be present and must name the block that binds it, and `follow` must be fed.
	for _, b := range blocks {
		if !b.hasFrom {
			continue
		}
		if b.fromNone && b.number != blocks[0].number {
			// A root after the first block is lawful only when something consumes
			// its variable - reported in (c) below.
		}
		if strings.EqualFold(b.op, "follow") && len(b.fromEntries) == 0 {
			findings = append(findings, fmt.Sprintf("schema integrity: follow op has no bound input (block %s)", b.number))
		}
		for _, e := range b.fromEntries {
			binders := byVar[strings.ToLower(e.varName)]
			if strings.HasPrefix(e.varName, "?") && len(binders) == 0 {
				findings = append(findings, fmt.Sprintf("schema integrity: consumed variable is never bound (%s, block %s)", e.varName, b.number))
				continue
			}
			if !e.hasPointer {
				findings = append(findings, fmt.Sprintf("schema integrity: from entry has no block pointer (%s, block %s)", e.varName, b.number))
				continue
			}
			matched := false
			for _, p := range binders {
				if p.number == e.block {
					matched = true
				}
			}
			if !matched && len(binders) > 0 {
				findings = append(findings, fmt.Sprintf("schema integrity: dependency pointer does not match the binding block (%s, block %s)", e.varName, b.number))
			}
		}
	}

	// (c) `From: none` marks a ROOT; a root whose variable nobody consumes is a
	// second, disconnected question, and every block must reach the ?answer block.
	for _, b := range blocks {
		if !b.hasFrom || !b.fromNone || b.binds == "" {
			continue
		}
		consumers := 0
		for _, other := range blocks {
			if other == b {
				continue
			}
			for _, e := range other.fromEntries {
				if strings.EqualFold(e.varName, b.binds) {
					consumers++
				}
			}
		}
		// The `?answer` block is exempt: a one-block plan (or a flat question
		// whose naming block comes last) is exactly that block, and it is not a
		// second question just because nothing consumes the value it names.
		if consumers == 0 && !strings.EqualFold(b.binds, "?answer") {
			findings = append(findings, fmt.Sprintf("schema integrity: root block's variable is never consumed (block %s) - connect it in its consumer's `From:` or delete the block", b.number))
		}
	}
	if len(answerBlocks) == 1 {
		findings = append(findings, decompositionReach(blocks, byVar, answerBlocks[0])...)
	}

	for _, b := range blocks {
		if b.trailingField {
			findings = append(findings, fmt.Sprintf("schema integrity: constraints line carries a trailing field - the chain is stated by From (block %s)", b.number))
		}
	}

	// (d) only a VARIABLE crosses a block boundary.
	for _, b := range blocks {
		if decompositionProseRefRe.MatchString(b.title) {
			findings = append(findings, fmt.Sprintf("schema integrity: block number or prose is used across a block boundary (title, block %s) - write the variable instead", b.number))
		}
		for _, c := range b.constraintDefs {
			if decompositionProseRefRe.MatchString(c.claim) {
				findings = append(findings, fmt.Sprintf("schema integrity: block number or prose is used across a block boundary (%s, block %s) - write the variable instead", c.id, b.number))
			}
		}
	}

	// (e) a title is a STABLE plan: an anchor the question supplies, or another
	// block's `?variable` - never a value a search returned.
	for _, b := range blocks {
		findings = append(findings, decompositionTitle(questionTokens, questionYears, b)...)
	}

	// (f) the constraint numbers are a PARTITION, and a number that carries a
	// DIFFERENT claim in a second block is a restarted numbering.
	defined := map[string][]*decompositionBlock{}
	for _, b := range blocks {
		if len(b.constraintDefs) == 0 {
			findings = append(findings, fmt.Sprintf("schema integrity: block defines no constraint (block %s)", b.number))
		}
		for _, c := range b.constraintDefs {
			defined[c.id] = append(defined[c.id], b)
		}
	}
	ids := sortedKeys(defined)
	for _, id := range ids {
		owners := defined[id]
		if len(owners) < 2 {
			continue
		}
		claims := map[string]bool{}
		for _, o := range owners {
			for _, c := range o.constraintDefs {
				if c.id == id {
					claims[c.claimKey] = true
				}
			}
		}
		if len(claims) > 1 {
			findings = append(findings, fmt.Sprintf("schema integrity: constraint number is reused for a different claim (%s)", id))
			continue
		}
		findings = append(findings, fmt.Sprintf("schema integrity: constraint is defined by two blocks (%s)", id))
	}

	// (g) a constraint is a self-contained claim about the question's own
	// anchors: another block's filler stays the variable that produced it.
	for _, b := range blocks {
		for _, c := range b.constraintDefs {
			for _, token := range decompositionNamedValues(questionTokens, c.claim) {
				findings = append(findings, fmt.Sprintf("suspect: constraint inlines another block's candidate - use its variable (%s, %s, block %s)", token, c.id, b.number))
			}
		}
	}

	sort.Strings(findings)
	return findings
}

// decompositionParse reads the block headings and their header fields.
func decompositionParse(plain string) ([]*decompositionBlock, []string) {
	heads := decompositionHeadRe.FindAllStringSubmatchIndex(plain, -1)
	blocks := make([]*decompositionBlock, 0, len(heads))
	findings := []string{}
	for i, h := range heads {
		start := h[0]
		end := len(plain)
		if i+1 < len(heads) {
			end = heads[i+1][0]
		}
		body := plain[start:end]
		b := &decompositionBlock{number: plain[h[2]:h[3]], title: strings.TrimSpace(plain[h[4]:h[5]])}
		if m := decompositionSlotRe.FindStringSubmatch(b.title); m != nil {
			b.slot = strings.ToLower(m[1])
		}
		if m := decompositionOpRe.FindStringSubmatch(body); m != nil {
			b.op = strings.TrimSpace(m[1])
			b.hasOp = b.op != ""
		}
		if m := decompositionBindsRe.FindStringSubmatch(body); m != nil {
			b.hasBinds = true
			b.binds = m[1]
		}
		if m := decompositionFromRe.FindStringSubmatch(body); m != nil {
			b.hasFrom = true
			raw := strings.TrimSpace(m[1])
			lower := strings.ToLower(raw)
			if lower == "" || strings.HasPrefix(lower, "none") {
				b.fromNone = true
			} else {
				for _, e := range decompositionFromEntRe.FindAllStringSubmatch(raw, -1) {
					entry := decompositionEdge{varName: e[1]}
					if p := decompositionPointerRe.FindStringSubmatch(e[2]); p != nil {
						entry.hasPointer = true
						entry.block = p[1]
					}
					b.fromEntries = append(b.fromEntries, entry)
				}
			}
		}
		if m := decompositionConRe.FindStringSubmatch(body); m != nil {
			b.hasConstraints = true
			// Every `;`-separated segment must be a numbered definition: text
			// trailing the last constraint is a field the line does not own.
			for _, seg := range strings.Split(m[1], ";") {
				if strings.TrimSpace(seg) != "" && !decompositionCdefRe.MatchString(seg) {
					b.trailingField = true
				}
			}
			for _, cd := range decompositionCdefRe.FindAllStringSubmatch(m[1], -1) {
				b.constraintDefs = append(b.constraintDefs, decompositionConstraint{
					id:       cd[1],
					claim:    strings.TrimSpace(cd[2]),
					claimKey: strings.ToLower(strings.Join(strings.Fields(cd[2]), " ")),
				})
			}
		}
		blocks = append(blocks, b)
	}
	for _, b := range blocks {
		if !b.hasOp {
			findings = append(findings, fmt.Sprintf("schema integrity: op is missing or unknown (block %s)", b.number))
		}
		if !b.hasConstraints {
			findings = append(findings, fmt.Sprintf("schema integrity: constraints line is missing (block %s)", b.number))
		}
		switch {
		case b.slot == "":
			findings = append(findings, fmt.Sprintf("schema integrity: slot is missing (block %s)", b.number))
		case !decompositionSlots[b.slot]:
			findings = append(findings, fmt.Sprintf("schema integrity: slot type unknown (%s, block %s)", b.slot, b.number))
		}
	}
	return blocks, findings
}

// decompositionReach reports the two graph defects: a cycle, and a block that
// does not hang off the ?answer block.
//
// The traversal is the plan's own `consumed by` relation - a `From:` entry points
// FROM the consumer TO the block that binds the variable - so the edges built
// here run producer -> consumer and the answer block is where a chain lands. The
// connectivity test is deliberately UNDIRECTED: in a chain every block flows
// toward the ?answer block, while a flat question's constraint blocks consume it
// and flow away from it, and requiring either direction alone would fail the
// other lawful shape. What both shapes share is that every block touches the
// ?answer block's component ("consumed by" / "chain" in the plan's own table),
// which is the property the tool reports.
func decompositionReach(blocks []*decompositionBlock, byVar map[string][]*decompositionBlock, answer *decompositionBlock) []string {
	findings := []string{}
	edges := map[string][]string{}      // producer -> consumer, for the cycle walk
	undirected := map[string][]string{} // both ways, for connectivity
	link := func(producer, consumer string) {
		edges[producer] = append(edges[producer], consumer)
		undirected[producer] = append(undirected[producer], consumer)
		undirected[consumer] = append(undirected[consumer], producer)
	}
	for _, c := range blocks {
		for _, e := range c.fromEntries {
			for _, p := range byVar[strings.ToLower(e.varName)] {
				if p == c {
					findings = append(findings, fmt.Sprintf("schema integrity: decomposition has a cycle (block %s consumes its own variable %s)", c.number, e.varName))
					continue
				}
				link(p.number, c.number)
			}
		}
	}
	color := map[string]int{} // 0 white, 1 grey, 2 black
	var walk func(n string, path []string)
	walk = func(n string, path []string) {
		switch color[n] {
		case 1:
			findings = append(findings, fmt.Sprintf("schema integrity: decomposition has a cycle (%s)", strings.Join(append(path, n), " -> ")))
			return
		case 2:
			return
		}
		color[n] = 1
		for _, m := range edges[n] {
			walk(m, append(path, n))
		}
		color[n] = 2
	}
	for _, b := range blocks {
		walk(b.number, nil)
	}
	seen := map[string]bool{answer.number: true}
	queue := []string{answer.number}
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		for _, m := range undirected[n] {
			if !seen[m] {
				seen[m] = true
				queue = append(queue, m)
			}
		}
	}
	for _, b := range blocks {
		if !seen[b.number] {
			findings = append(findings, fmt.Sprintf("schema integrity: block does not reach the answer block (block %s) - declare its `From` edge or delete the block", b.number))
		}
	}
	return findings
}

// decompositionTitle reports the two title defects the auditor names: a title
// with no anchor at all, and a title that names a value the question never
// supplies (which, before any search, can only be an inlined candidate).
func decompositionTitle(questionTokens, questionYears map[string]bool, b *decompositionBlock) []string {
	title := b.title
	if i := strings.Index(title, "slot:"); i > 0 {
		title = title[:i]
	}
	title = strings.TrimRight(strings.TrimSpace(title), "-— ")
	// A title carrying another block's variable satisfies the anchor rule, but it
	// is still scanned for an inlined value: "the hospital where ?person died in
	// Smithtown" names a place a search returned just as plainly as a title with
	// no variable at all.
	hasVar := strings.Contains(title, "?")
	tokens := decompositionWordRe.FindAllString(title, -1)
	years := decompositionYearRe.FindAllString(title, -1)
	if len(tokens) == 0 && len(years) == 0 {
		return []string{fmt.Sprintf("suspect: sub-question is not self-contained - it names no anchor (block %s)", b.number)}
	}
	anchored := false
	outside := []string{}
	for _, tok := range tokens {
		lower := strings.ToLower(tok)
		// A generic word the question happens to supply ("school", "hospital",
		// "the person") is shape vocabulary, not an anchor: the rule wants a
		// proper noun, a number or a distinctive term, and the auditor's own
		// failures are titled exactly that way ("which hospital", "the person").
		if decompositionGenericWord[lower] {
			continue
		}
		if questionTokens[lower] || decompositionSharesStem(questionTokens, lower) {
			anchored = true
			continue
		}
		if decompositionGlossWord[lower] {
			continue
		}
		if tok[0] >= 'A' && tok[0] <= 'Z' {
			outside = append(outside, tok)
		}
	}
	for _, y := range years {
		if questionYears[y] {
			anchored = true
		}
	}
	findings := []string{}
	if len(outside) > 0 {
		sort.Strings(outside)
		findings = append(findings, fmt.Sprintf("suspect: sub-question inlines another block's candidate - use its variable (%s, block %s)", strings.Join(outside, ", "), b.number))
	}
	if !anchored && !hasVar {
		findings = append(findings, fmt.Sprintf("suspect: sub-question is not self-contained - it names no anchor (block %s)", b.number))
	}
	return findings
}

// decompositionNamedValues returns the capitalised, non-generic values in a
// constraint that the question never supplies.
func decompositionNamedValues(questionTokens map[string]bool, text string) []string {
	out := []string{}
	for _, tok := range decompositionWordRe.FindAllString(text, -1) {
		lower := strings.ToLower(tok)
		if questionTokens[lower] || decompositionSharesStem(questionTokens, lower) ||
			decompositionGenericWord[lower] || decompositionGlossWord[lower] {
			continue
		}
		if tok[0] >= 'A' && tok[0] <= 'Z' {
			out = append(out, tok)
		}
	}
	sort.Strings(out)
	return out
}

func decompositionTokens(text string) map[string]bool {
	out := map[string]bool{}
	for _, tok := range decompositionWordRe.FindAllString(text, -1) {
		out[strings.ToLower(tok)] = true
	}
	return out
}

// decompositionSharesStem keeps the anchor test lenient: an inflected form of a
// question token still counts as the question's own anchor, because a false "no
// anchor" on a lawful title costs the run a rewrite.
func decompositionSharesStem(questionTokens map[string]bool, token string) bool {
	if len(token) < 6 {
		return false
	}
	for i := len(token) - 1; i >= 5; i-- {
		if questionTokens[token[:i]] {
			return true
		}
	}
	return false
}

// decompositionGenericWord holds the words that legitimately appear in a title
// or a constraint without being an anchor: shape vocabulary, connectives, and
// the facet tags the title carries.
var decompositionGenericWord = map[string]bool{
	"sub": true, "question": true, "the": true, "a": true, "an": true, "of": true, "and": true,
	"or": true, "who": true, "whose": true, "that": true, "this": true, "these": true, "those": true,
	"one": true, "two": true, "three": true, "four": true, "several": true, "many": true, "most": true,
	"both": true, "each": true, "other": true, "another": true, "same": true, "any": true, "all": true,
	"named": true, "name": true, "names": true, "slot": true, "kind": true, "value": true,
	"person": true, "people": true, "individual": true, "individuals": true, "figure": true, "pair": true,
	"organization": true, "organisation": true, "institution": true, "hospital": true, "school": true,
	"schools": true, "answer": true, "variable": true, "formula": true, "compound": true, "molecule": true,
	"title": true, "text": true, "year": true, "years": true, "date": true, "number": true, "count": true,
	"first": true, "last": true, "shared": true, "identity": true, "attended": true, "died": true,
	"born": true, "raised": true, "lived": true, "worked": true, "scored": true, "played": true,
	"across": true, "between": true, "before": true, "after": true, "during": true, "from": true,
	"with": true, "without": true, "where": true, "which": true, "while": true, "when": true,
	"their": true, "there": true, "then": true, "than": true, "into": true, "over": true,
	"full": true, "part": true, "set": true, "list": true, "clue": true, "clues": true,
	"property": true, "fact": true, "claim": true, "constraint": true, "constraints": true,
}

// decompositionGlossWord holds the words a constraint may legitimately INTRODUCE
// while resolving the question's own wording: the calendar, the seasons, the
// zodiac and the unit names are what "resolve a relative window against the
// question's anchor" means in practice, and an archived run that glossed "under
// the zodiac sign Cancer" into "(June 21 - July 22)" was reported as an inlined
// candidate until this list existed. Everything else a constraint introduces is
// a value the question never supplied - which, before any search, can only be a
// guess or another block's filler.
var decompositionGlossWord = map[string]bool{
	"january": true, "february": true, "march": true, "april": true, "may": true, "june": true,
	"july": true, "august": true, "september": true, "october": true, "november": true, "december": true,
	"jan": true, "feb": true, "mar": true, "apr": true, "jun": true, "jul": true, "aug": true,
	"sep": true, "sept": true, "oct": true, "nov": true, "dec": true,
	"monday": true, "tuesday": true, "wednesday": true, "thursday": true, "friday": true,
	"saturday": true, "sunday": true, "weekday": true, "weekend": true,
	"spring": true, "summer": true, "autumn": true, "fall": true, "winter": true,
	"zodiac": true, "aries": true, "taurus": true, "gemini": true, "cancer": true, "leo": true,
	"virgo": true, "libra": true, "scorpio": true, "sagittarius": true, "capricorn": true,
	"aquarius": true, "pisces": true, "horoscope": true, "sign": true,
	"century": true, "decade": true, "day": true, "week": true, "month": true, "season": true,
	"mile": true, "miles": true, "kilometre": true, "kilometer": true, "metre": true, "meter": true,
	"pound": true, "pounds": true, "kilogram": true, "gram": true, "inch": true, "inches": true,
	"foot": true, "feet": true, "degree": true, "degrees": true, "percent": true,
}

// decompositionSlots is the closed shape list a title's `slot:` tag draws from.
var decompositionSlots = map[string]bool{
	"name": true, "literal": true, "number": true, "quantity": true, "date": true, "set": true,
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool {
		if len(out[i]) != len(out[j]) {
			return len(out[i]) < len(out[j])
		}
		return out[i] < out[j]
	})
	return out
}

func blockNumbers(blocks []*decompositionBlock) []string {
	nums := make([]string, 0, len(blocks))
	for _, b := range blocks {
		nums = append(nums, b.number)
	}
	sort.Strings(nums)
	return nums
}
