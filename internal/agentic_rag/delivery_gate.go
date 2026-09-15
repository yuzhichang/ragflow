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
	"encoding/json"
	"regexp"
	"sort"
	"strings"
)

// The answer line is READ, not matched. Five shapes have shown up in real runs —
// plain (`Final Answer: **X**`), value-on-the-next-line (`## Final Answer` then
// `**X**`), whole-line bold (`**Final Answer: X**`), the label bolded on its own
// (`**Final Answer**: **X**`), and notes that CITE documents in parentheses
// (`... 2 independent chunks (64640.md, 55819.md)`) — and each fix until now was one
// more pattern for one more shape. That is a treadmill by construction: the tail
// after the value is prose the model writes freely, so any whitelist of what may
// follow a value will be wrong again. This parser keeps ONE pattern — the contract's
// label vocabulary, which does not vary — and scans everything around it
// tolerantly. The tail is not policed at all: an answer line is a line that carries
// the label and a value, and what the notes say is the auditor's business.
//
// Cost of the old approach, measured: the tail whitelist rejected `(assumption:
// ...(82489.md)...)` and the gate reported `answer_value_missing` while HOLDING a
// well-formed line, skipping every value-based check and buying a repair turn.

// answerLineLabelRe recognises ONLY the label words, at the start of a line once the
// markdown wrappers are stripped.
var answerLineLabelRe = regexp.MustCompile(`(?i)^(final|guessed)\s+answer\b`)

// answerDecorations are the markdown wrappers a line may carry before its label:
// indentation, blockquote, heading marks, emphasis, list bullets.
const answerDecorations = " \t>*#_-–—"

// answerValueDecorations is the same set WITHOUT the asterisk: on a line that carries
// only a value, a leading `**` is the value's own bold marker, not a wrapper.
const answerValueDecorations = " \t>#_-–—"

// answerColons are the separators a label may carry: the ASCII colon and the
// full-width one. Bilingual output is the norm here, and a full-width colon used to
// survive into the value (`：X`), which the grounding check would then read as a
// value that appears in no chunk.
const answerColons = " \t:："

// answerLine is the answer line as the gate READS it.
type answerLine struct {
	Label   string   // "final" or "guessed", lowercase
	Value   string   // the answer's own text
	Carrier []string // the line(s) that may carry notes and tie clauses
	Lines   int      // how many lines the answer line spans (1 or 2)
}

// readAnswerLine returns the deliverable's answer line, or ok=false when there is
// none. The FIRST line that parses wins; a deliverable that re-rendered itself ships
// two answer lines and the gate still acts on a defined one (finalAnswerLineCount is
// what flags the shape).
func readAnswerLine(final string) (answerLine, bool) {
	lines := strings.Split(final, "\n")
	for i := range lines {
		if al, ok := readAnswerLineAt(lines, i); ok {
			return al, true
		}
	}
	return answerLine{}, false
}

// readAnswerLineAt parses the answer line whose label sits on lines[i].
func readAnswerLineAt(lines []string, i int) (answerLine, bool) {
	rest, label, ok := stripAnswerLabel(lines[i])
	if !ok {
		return answerLine{}, false
	}
	al := answerLine{Label: label, Carrier: []string{lines[i]}, Lines: 1}
	if value := answerValueToken(rest); value != "" {
		al.Value = value
		return al, true
	}
	// Nothing after the label: the value-on-the-next-line shape. Blank lines are
	// skipped and the first non-blank line decides — if that is another label line,
	// this heading carried no value and the caller keeps looking.
	for j := i + 1; j < len(lines); j++ {
		if strings.TrimSpace(lines[j]) == "" {
			continue
		}
		if _, _, isLabel := stripAnswerLabel(lines[j]); isLabel {
			break
		}
		if value := answerBoldToken(lines[j]); value != "" {
			al.Value = value
			al.Carrier = append(al.Carrier, lines[j])
			al.Lines = j - i + 1
			return al, true
		}
		break
	}
	return answerLine{}, false
}

// stripAnswerLabel removes the markdown wrappers before a label and returns the text
// after it. Only the label words are matched by a pattern; the wrappers are stripped
// by character class, so `## Final Answer`, `**Guessed Answer**:` and
// `> Guessed  Answer` all land on the same code path.
func stripAnswerLabel(line string) (rest, label string, ok bool) {
	trimmed := stripAnswerDecorations(line)
	m := answerLineLabelRe.FindStringSubmatch(trimmed)
	if m == nil {
		return "", "", false
	}
	return trimmed[len(m[0]):], strings.ToLower(m[1]), true
}

// stripAnswerDecorations drops the leading markdown wrappers.
func stripAnswerDecorations(line string) string {
	return strings.TrimLeft(line, answerDecorations)
}

// answerValueToken reads the value that follows a label. Every shape lands here and
// none of them needs its own pattern:
//
//   - `: **X**` or `**: **X**` — the bolded run after the colon;
//   - `: X` — the plain text: a value the model forgot to bold is still the value,
//     and the delivery's shape is the auditor's call, not the reader's;
//   - `: X**` (whole-line bold) — the text up to the closing asterisks;
//   - `: **<a whole document>` — "" — an UNTERMINATED bold run is the mega-value blob
//     shape (a re-rendered document inside the value), never an answer.
func answerValueToken(rest string) string {
	rest = strings.TrimLeft(rest, " \t")
	// The label may be bolded on its OWN, closing before the colon
	// (`**Final Answer**: **X**`): that pair closes the label, not the value, and it
	// has to be consumed before the value's opener is looked for - otherwise the
	// opener is read as a decoration and an unterminated value slips through.
	if strings.HasPrefix(rest, "**") {
		rest = strings.TrimLeft(rest[2:], answerColons)
	}
	rest = strings.TrimLeft(rest, answerColons)
	if strings.HasPrefix(rest, "**") {
		body := rest[2:]
		end := strings.Index(body, "**")
		if end < 0 {
			return ""
		}
		return strings.TrimSpace(body[:end])
	}
	if end := strings.Index(rest, "**"); end >= 0 {
		rest = rest[:end]
	}
	return strings.TrimSpace(strings.TrimRight(rest, " \t*"))
}

// answerBoldToken reads the bolded run that stands on its OWN line — the
// value-on-the-next-line contract. A heading followed by a PARAGRAPH is a heading
// (`## Final Answer` then prose is not an answer line; the old two-line matcher
// required the bolded marker for exactly that reason), while the same-line form above
// accepts a plain value because the label has already said what that line is.
func answerBoldToken(line string) string {
	rest := strings.TrimLeft(line, answerValueDecorations)
	if !strings.HasPrefix(rest, "**") {
		return ""
	}
	body := rest[2:]
	end := strings.Index(body, "**")
	if end < 0 {
		return ""
	}
	return strings.TrimSpace(body[:end])
}

// answerNoteSafe strips the characters that would break the note they are wrapped
// in: a newline would split the line the label governance rewrites, and brackets keep
// a note from ending its own group on a reader's page.
func answerNoteSafe(s string) string {
	return strings.NewReplacer("(", "[", ")", "]", "\n", " ", "\r", " ").Replace(s)
}

// finalAnswerValue returns the value shipped on the FOS answer line, or "" when the
// reply carries none. It is FORMAT PRESENCE, not a correctness judgement — the gate
// uses it only to build the auditor payload, to decide whether a gate continuation may
// replace the standing final, and to drive the grounding/list-only suspicions.
// Whether the value is CORRECT is exclusively answer_auditor's business.
func finalAnswerValue(final string) string {
	al, ok := readAnswerLine(final)
	if !ok {
		return ""
	}
	if al.Value == "" || answerLabelRe.MatchString(al.Value) {
		// A bolded LABEL is not a value: `## Final Answer` followed by
		// `**Final Answer**` must keep reading as the value-less shape it is.
		return ""
	}
	return al.Value
}

// finalAnswerLineCount counts the deliverable's answer lines. The finalize synthesis
// must produce exactly ONE; zero means the answer line lost its shape and two-plus
// means the model re-rendered the whole deliverable (the two-answer-line failure, or
// the mega-value blob that swallows one), and either marks the synthesis degenerate.
func finalAnswerLineCount(s string) int {
	lines := strings.Split(s, "\n")
	count := 0
	for i := 0; i < len(lines); {
		al, ok := readAnswerLineAt(lines, i)
		if !ok {
			i++
			continue
		}
		count++
		i += al.Lines
	}
	return count
}

// balancedParenEnd returns the index of the `)` closing the group that opens at
// start, or -1 when the group never closes. Notes and tie clauses CITE documents
// inside their parentheses — `... 2 independent chunks (64640.md, 55819.md)` — so the
// body cannot be read with a `[^)]*` class, which is what made such an answer line
// read as value-less.
func balancedParenEnd(s string, start int) int {
	depth := 0
	for i := start; i < len(s); i++ {
		switch s[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// answerTieNotes returns the bodies of the `(tie: ...)` clauses on one line, with
// balanced nesting inside a body.
func answerTieNotes(line string) []string {
	var out []string
	for i := 0; i < len(line); i++ {
		if line[i] != '(' {
			continue
		}
		if !strings.HasPrefix(strings.ToLower(line[i+1:]), "tie:") {
			continue
		}
		end := balancedParenEnd(line, i)
		if end < 0 {
			break
		}
		out = append(out, strings.TrimSpace(line[i+len("(tie:"):end]))
		i = end
	}
	return out
}

// unnamedTieRival stands in for a `(tie: )` clause that names nothing. It keeps
// the declaration visible instead of letting a malformed clause read as "no tie
// declared" — the one case where a run could otherwise escape the Guessed label
// by writing the clause wrong.
const unnamedTieRival = "(unnamed rival)"

// finalAnswerTies returns the rivals the ANSWER LINE's tie clauses name, in the
// order written, or nil when the deliverable declares none.
//
// One clause PER RIVAL, not a list inside one clause: the candidates are titles,
// and titles carry commas — the q221 gold itself is "In the Arms of Morpheus:
// The Tragic History of Laudanum, Morphine and Patent Medicines" — so a
// comma-separated list could not be split back into candidates, while repeated
// clauses parse exactly. A tie is not limited to two candidates either: a
// question can be under-determined by the corpus with three or more grounded
// candidates, and every one of them has to be named or the deliverable is
// hiding a rival it could not rule out.
//
// It reads answer lines only, never the whole deliverable: a `(tie: ...)` in
// narrative prose is not a declared tie, and this value is ACTED on — it forces
// the `Guessed Answer` label even when the audit passed (see the label
// governance in Run), because a tie is a statement that the discriminating
// constraint is NOT corpus-verified, which is exactly what `Final Answer`
// claims. Matching mid-prose would demote honest `Final Answer` runs.
func finalAnswerTies(final string) []string {
	al, ok := readAnswerLine(final)
	if !ok {
		return nil
	}
	var rivals []string
	for _, carrier := range al.Carrier {
		for _, note := range answerTieNotes(carrier) {
			rival := strings.TrimSpace(note)
			if rival == "" {
				rival = unnamedTieRival
			}
			rivals = append(rivals, rival)
		}
	}
	return rivals
}

// fosSectionRe matches the structural headings of the FOS deliverable.
var fosSectionRe = regexp.MustCompile(`(?im)^[^\S\n]*#{1,4}[^\S\n]*(?:Candidate Matrix|Reasoning Chain)\b`)

// hasFOSStructure reports whether the reply carries the deliverable's
// structural headings. Format presence again — it decides whether an
// audit-failed deliverable is worth shipping verbatim (it still carries
// corpus-grounded structure) versus pure narration that only finalizeAnswer
// can turn into an answer (the q55-class A-gate stays intact).
func hasFOSStructure(final string) bool {
	return fosSectionRe.MatchString(final)
}

// answerLabelRe matches an answer label at the start of a line: optional
// indentation, optional markdown heading marks, optional emphasis, then
// Final/Guessed + "answer". Groups: 1 = prefix (indent/heading/emphasis),
// 2 = the label word, 3 = the "answer" tail including its colon when present.
// Matching the prefix is what lets ONE regex serve both shapes the deliverable
// mixes — the `## Final Answer` heading and the `**Final Answer: X**` value line.
// The prefix tolerates the wrappers a deliverable puts around the label —
// emphasis, backticks, straight or curly quotes — so a value line written as
// `Guessed Answer: **X**` is still recognised (RE2 has no \uXXXX escapes, hence
// the literal quotes).
var answerLabelRe = regexp.MustCompile("(?i)^([^\\S\\n]*(?:#{1,4}\\s*)?[`*\"'‘’“”\\s]*)(final|guessed)(\\s+answer\\s*:?)")

// demoteFinalAnswerLabel rewrites the deliverable's answer label from
// `Final Answer` to `Guessed Answer` when the delivery gate shipped WITHOUT a
// concluding PASS audit.
//
// `Final Answer` is a claim that every discriminating constraint is
// corpus-verified, so it may only survive an audit that passed. Measured on the
// 525-question browsecomp run, 29 questions shipped a WRONG answer under that
// label while 76% of them had never passed their last audit - the gate stopped
// on a stall, a budget or the clock, and the over-claim rode along. Demotion is
// label-only: the value (what the judge scores) is untouched, only the
// confidence claim is corrected.
//
// EVERY label occurrence is rewritten, not just the first: a deliverable
// carries the label twice (heading + value line), and demoting one left
// `## Final Answer` sitting above `**Guessed Answer: X**` — the corrected value
// under the stronger claim (4/46 rows of the browsecomp retry batch). The
// assumption note is appended once, on the line that actually carries the value.
func demoteFinalAnswerLabel(final, reason string) (string, bool) {
	lines := strings.Split(final, "\n")
	changed := false
	assumptionLine := -1
	for i, line := range lines {
		loc := answerLabelRe.FindStringSubmatchIndex(line)
		if loc == nil {
			continue
		}
		// Where the assumption belongs: the line whose label is followed by a
		// value (`Final Answer: **X**`), not the bare `## Final Answer` heading.
		// Read from the ORIGINAL line - the rewrite shifts the offsets.
		if assumptionLine < 0 && strings.TrimSpace(line[loc[7]:]) != "" {
			assumptionLine = i
		}
		if !strings.EqualFold(line[loc[4]:loc[5]], "final") {
			continue
		}
		lines[i] = line[:loc[4]] + "Guessed" + line[loc[5]:]
		changed = true
	}
	if !changed {
		return final, false
	}
	if assumptionLine < 0 {
		// Heading-only deliverable: the note rides the first label line.
		for i, line := range lines {
			if answerLabelRe.MatchString(line) {
				assumptionLine = i
				break
			}
		}
	}
	if assumptionLine >= 0 && !strings.Contains(strings.ToLower(lines[assumptionLine]), "assumption:") {
		lines[assumptionLine] = strings.TrimRight(lines[assumptionLine], " \t\r") + " (assumption: " + reason + ")"
	}
	return strings.Join(lines, "\n"), true
}

// reconcileAnswerLabels makes a deliverable agree with itself on the answer
// label. Demotion normalizes the gate's own rewrite, but the MODEL also mixes
// the shapes on its own (`## Final Answer` heading above a `**Guessed Answer:
// X**` value line), and the label can arrive from a synthesized or recovered
// deliverable that never went through demotion at all.
//
// The conservative label wins: a Guessed value under a Final heading overstates
// verification — precisely the claim lever 1 exists to police — so the heading
// is rewritten to match the value. The value itself is never touched.
func reconcileAnswerLabels(final string) string {
	lines := strings.Split(final, "\n")
	guessed := false
	for _, line := range lines {
		if loc := answerLabelRe.FindStringSubmatchIndex(line); loc != nil && strings.EqualFold(line[loc[4]:loc[5]], "guessed") {
			guessed = true
			break
		}
	}
	if !guessed {
		return final // no conservative claim to align to
	}
	changed := false
	for i, line := range lines {
		loc := answerLabelRe.FindStringSubmatchIndex(line)
		if loc == nil || !strings.EqualFold(line[loc[4]:loc[5]], "final") {
			continue
		}
		lines[i] = line[:loc[4]] + "Guessed" + line[loc[5]:]
		changed = true
	}
	if !changed {
		return final
	}
	return strings.Join(lines, "\n")
}

var (
	// namePropertyQuestionRe matches a question asking for a NAME PROPERTY — the
	// kind of fact an enumeration can never establish. CJK phrasings are
	// included because the corpus and the questions are bilingual.
	namePropertyQuestionRe = regexp.MustCompile(`(?i)\b(birth name|full birth name|full name|given name|real name|maiden name|birth surname|née|né)\b|本名|艺名|原名|真名|全名|姓名`)
	// propertyBeforeRe / propertyAfterRe match the phrasing that STATES such a
	// property, applied to a window just before / after the value.
	propertyBeforeRe = regexp.MustCompile(`(?i)(born|née|né|real name|birth ?name|given name|full name|maiden name|surname)\s*[:,\-]?\s*$|(本名|艺名|原名|真名|全名|姓名)\s*[:：,，]?\s*$`)
	propertyAfterRe  = regexp.MustCompile(`(?i)^\s*[:,\-]?\s*(was born|is born|born)\b|^\s*[，,：:]?\s*(本名|艺名|原名|真名)`)
	// nameTokenRe counts name-like tokens in a window: a cast/crew list is a
	// dense run of capitalized words, a prose sentence is not.
	nameTokenRe = regexp.MustCompile(`\b[\p{Lu}][\p{Ll}\p{L}]{1,}\b`)
)

// listOnlyNameReason reports why a name-shaped answer must not ship: the
// question asks for a name PROPERTY, and the value only ever appears as one
// entry in an enumeration (a cast/crew/name list).
//
// A list names everyone in a film; it cannot state whose BIRTH NAME something
// is, so any name in it "matches" equally — shipping one is a pick, not an
// answer. Measured on q784: the gold name reached the model 7 times, the wrong
// name 152 times, and the answer came out of a cast list.
//
// Returns "" whenever the check does not apply (not a name-property question, a
// non-name value, or a value stated in a property context), so a prose answer is
// never rejected by this lever.
func listOnlyNameReason(question, value, haystack string) string {
	if strings.TrimSpace(value) == "" || haystack == "" {
		return ""
	}
	property := namePropertyQuestionRe.FindString(question)
	if property == "" {
		return "" // the question does not ask for a name property
	}
	// Name-shaped only: digits or long phrases are not a person's name, and the
	// property argument does not apply to them.
	if strings.ContainsAny(value, "0123456789") || len(strings.Fields(value)) > 6 {
		return ""
	}
	hay := strings.ToLower(haystack)
	needle := strings.ToLower(strings.TrimSpace(value))
	if needle == "" {
		return ""
	}
	inList := false
	for scan := 0; scan < len(hay); {
		offset := strings.Index(hay[scan:], needle)
		if offset < 0 {
			break
		}
		idx := scan + offset
		scan = idx + len(needle)
		// No word-boundary test here on purpose: the corpus glues list entries
		// together (`...PerèsPierre Albert BrasseurHenri Hennery...`) when markup
		// is stripped, so requiring a boundary would miss exactly the case this
		// lever exists for.
		beforeStart := idx - 60
		if beforeStart < 0 {
			beforeStart = 0
		}
		after := hay[idx+len(needle):]
		if len(after) > 60 {
			after = after[:60]
		}
		if propertyBeforeRe.MatchString(hay[beforeStart:idx]) || propertyAfterRe.MatchString(after) {
			return "" // stated as the property: nothing to object to
		}
		if !inList && looksLikeEnumeration(haystack, idx, len(needle)) {
			inList = true
		}
		// Keep scanning: a later occurrence may state the property properly.
	}
	if !inList {
		return ""
	}
	return "The value on your answer line (" + value + ") appears only as one entry in an ENUMERATION " +
		"(a cast/crew/name list). The question asks for the " + strings.ToLower(property) + ", and a list entry cannot " +
		"state a " + strings.ToLower(property) + " - every name in that list would match equally, so this is a pick, not " +
		"an answer. Do NOT re-render the same matrix. Find the sentence that STATES the property (shapes like " +
		"`born <value>`, `<value>'s birth name was ...`, `née <value>`, `real name <value>`) and quote it, or ship " +
		"`Guessed Answer: **" + value + "** (assumption: no chunk read states the " + strings.ToLower(property) + ")`."
}

// namePropertyHint returns extra repair text for a question that asks for a
// NAME PROPERTY, appended to the ungrounded-value directive. The generic advice
// ("run a new retrieval with a different anchor") sends the model hunting for
// the same slot; the property question needs a different shape of evidence — a
// sentence that STATES whose name it is — so say so explicitly.
func namePropertyHint(question, value string) string {
	property := namePropertyQuestionRe.FindString(question)
	if property == "" || strings.TrimSpace(value) == "" {
		return ""
	}
	return " This question asks for the " + strings.ToLower(property) + ": what you need is a sentence that STATES that " +
		"property about a value (shapes like `born <value>`, `<value>'s birth name was ...`, `née <value>`, `本名 <value>`), " +
		"not another mention of a name. Search for the property word itself together with a distinctive clue " +
		"(grep_chunks `birth name.*<clue>` / `<clue>.*born`), and quote the sentence in the Reasoning Chain. A name that only " +
		"appears inside a cast/crew/name list never satisfies this - every entry there matches equally."
}

// looksLikeEnumeration reports whether the window around a match reads as a list
// rather than prose: either many separators, or a dense run of capitalized
// (name-like) tokens. The corpus loses list separators when markup is stripped,
// so the capitalized-run test carries the weight.
func looksLikeEnumeration(haystack string, idx, length int) bool {
	start := idx - 200
	if start < 0 {
		start = 0
	}
	end := idx + length + 200
	if end > len(haystack) {
		end = len(haystack)
	}
	window := haystack[start:end]
	separators := strings.Count(window, ",") + strings.Count(window, ";") + strings.Count(window, "、") + strings.Count(window, "·")
	if separators >= 4 {
		return true
	}
	return len(nameTokenRe.FindAllString(window, -1)) >= 8
}

// normalizeForMatch folds text for a substring test: lowercase, drop markdown
// emphasis and collapse whitespace.
var nonWordRe = regexp.MustCompile(`[^\p{L}\p{N}]+`)

func normalizeForMatch(s string) string {
	return strings.TrimSpace(nonWordRe.ReplaceAllString(strings.ToLower(s), " "))
}

// answerValueIsGrounded reports whether the value shipped on the answer line
// appears in the text the tools actually returned. A named value that never
// occurs in any read chunk cannot be corpus-supported: it was synthesized.
//
// Purely numeric values are exempt - a count or a year is routinely derived
// (summed, subtracted) rather than quoted, so an absent literal proves nothing.
// An EMPTY haystack also passes: when the session records no tool results the
// check has nothing to work with, and failing every answer on that would be
// worse than skipping it.
func answerValueIsGrounded(value, haystack string) bool {
	value = strings.TrimSpace(value)
	if len([]rune(value)) < 2 || haystack == "" {
		return true // nothing to check against
	}
	if !strings.ContainsAny(value, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") {
		return true // numeric/derived: exempt
	}
	spacedHay := " " + normalizeForMatch(haystack) + " "
	if strings.Contains(spacedHay, " "+normalizeForMatch(value)+" ") {
		return true
	}
	// Token coverage: a value that is MORE complete than the corpus wording — a
	// middle name the source omitted, a formal title, an accent-free spelling —
	// must not read as unsupported, or a CORRECT answer gets sent back. On q784
	// the gold "Jacqueline Georgette Cantrelle" was refused nine times because
	// the corpus only ever wrote "Jacqueline Cantrelle".
	//
	// The allowance absorbs a MIDDLE omission and nothing else: the value's LAST
	// token is its head, the noun that says what the entity IS, so a value whose
	// head the corpus never writes with the rest is a DIFFERENT entity rather
	// than a fuller spelling of one. Observed on #71: a deliverable shipped "The
	// Ballast Bank Bar" over a chunk about "The Wexford Ballast Bank", and with
	// the tail optional three of four tokens present passed the check — the
	// audit concluded PASS and the answering step was never sent back to fix it.
	// A differently headed form now goes back for repair, and the repair turn
	// restates it the way the corpus writes it, which is the fix the run owes.
	tokens := strings.Fields(normalizeForMatch(value))
	if len(tokens) < 2 {
		return false
	}
	// Locality, not just token coverage: the rule above asks whether each token
	// appears SOMEWHERE in the haystack, and the haystack is every chunk the run
	// read - so a value whose tokens never share a document can read as grounded.
	// The window is deliberately generous for the opposite reason, and q283 is the
	// case that must STAY grounded: `Zimri Eder` is the WIKIPEDIA spelling of the
	// protagonist (doc 82489, served) while a fan wiki and a review write `Zimri
	// Elder` (the gold), so the run shipped a spelling the corpus attests verbatim
	// and declared the tie between the two. An earlier note here blamed that value
	// on cross-document assembly, which the corpus does not support - a matcher
	// keyed to the gold's spelling would fail a value that IS attested.
	// So: the tokens must occur IN ORDER inside one window, the head (the last
	// token - what the entity IS) may never be the missing one, and at most ONE
	// non-final token may be missing, which keeps the middle-omission allowance
	// intact (q784's "Jacqueline Georgette Cantrelle" over "Jacqueline Cantrelle").
	return valueTokensAppearNearby(strings.Fields(normalizeForMatch(haystack)), tokens)
}

// groundingWindowSlack is how many words a sliding match may span beyond the
// value's own length: the slack absorbs the words a source inserts between the
// value's tokens (a title, an epithet, the middle name the VALUE omits).
const groundingWindowSlack = 4

// valueTokensAppearNearby reports whether tokens occur in words in order, inside
// one window, with at most one skipped NON-FINAL token and the final one matched.
//
// It is a forward pass over the window with the state (matched, skipped) rather
// than a greedy walk, because the two admissible moves conflict: at "Jacqueline
// Cantrelle" the token `cantrelle` must be held for the head while `georgette`
// (the value's own middle token) is the one skipped. A greedy matcher that skips
// the value's token on the first mismatch eats the head and rejects the q784 gold.
func valueTokensAppearNearby(words, tokens []string) bool {
	last := len(tokens) - 1
	span := last + groundingWindowSlack
	reach := make([][2]bool, len(tokens)+1)
	for start := 0; start < len(words); start++ {
		if words[start] != tokens[0] {
			continue
		}
		end := start + span + 1
		if end > len(words) {
			end = len(words)
		}
		for i := range reach {
			reach[i] = [2]bool{}
		}
		reach[1][0] = true
		for j := start + 1; j < end; j++ {
			// Skipping a value's own token consumes NO haystack word, so it has to
			// be closed over BEFORE the word at this position is matched: otherwise
			// the match of `cantrelle` is only ever tried while the state still
			// expects `georgette`, and the two moves that q784 needs (omit the
			// middle token, then match the head) can never combine.
			for i := 0; i < last; i++ {
				if reach[i][0] {
					reach[i+1][1] = true
				}
			}
			next := make([][2]bool, len(tokens)+1)
			for i := 0; i <= len(tokens); i++ {
				for skipped := 0; skipped < 2; skipped++ {
					if !reach[i][skipped] {
						continue
					}
					// The corpus may insert words the value does not carry.
					next[i][skipped] = true
					if i < len(tokens) && words[j] == tokens[i] {
						next[i+1][skipped] = true
					}
				}
			}
			reach = next
			if reach[len(tokens)][0] || reach[len(tokens)][1] {
				return true
			}
		}
	}
	return false
}

// locateToolNames are the tools whose calls the Candidate Matrix's `Searched:`
// lines are supposed to record.
var locateToolNames = []string{"grep_chunks", "search_bm25_chunks", "search_semantic_chunks", "search_chunks"}

// searchedLineRe matches a matrix line that records a locate call, capturing the
// tool name it credits.
var searchedLineRe = regexp.MustCompile(`(?im)Searched:\s*(grep_chunks|search_bm25_chunks|search_semantic_chunks|search_chunks)`)

// unrecordedLocateTools returns the locate tools the run actually called but the
// deliverable never credits on a `Searched:` line, sorted.
//
// It exists for DIAGNOSIS, and deliberately not as a gate rejection: measured on
// 703 scored rows, 69% used some locate tool whose call the matrix never names
// (grep_chunks in 362 of them), so rejecting runs on it would reject two out of
// three deliverables over bookkeeping. The cost of that silence is real, though:
// on #71 the ONE call that surfaced the answer's own document was a
// search_semantic_chunks query the matrix omitted, so nothing in the archived row
// showed the pure-vector leg had done the work — the log had to be read to see
// it. The warning makes that visible without changing what the model is asked to
// do; whether to enforce the schema is a separate, evidence-gated decision.
func unrecordedLocateTools(matrix string, toolCallCounts map[string]int) []string {
	recorded := map[string]struct{}{}
	for _, m := range searchedLineRe.FindAllStringSubmatch(matrix, -1) {
		recorded[strings.ToLower(m[1])] = struct{}{}
	}
	out := make([]string, 0, len(locateToolNames))
	for _, name := range locateToolNames {
		if toolCallCounts[name] <= 0 {
			continue
		}
		if _, ok := recorded[name]; ok {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// auditVerdictExcerptMax bounds how much of one audit verdict is archived. The
// excerpt only has to be enough to see WHICH claims were contested and why; the
// full text of ten rounds would bloat every benchmark row.
const auditVerdictExcerptMax = 400

var (
	// auditOpinionRe matches the auditor's per-item opinion sub-lines. The
	// auditor echoes the whole deliverable back and hangs one `- audit: <opinion>`
	// under every audited line, so the deliverable's own text is 95% of the
	// verdict string and none of it is a finding.
	auditOpinionRe = regexp.MustCompile(`(?m)^\s*-\s*audit:\s*(.+?)\s*$`)
	// auditResultLineRe matches the overall verdict line that closes the audit.
	auditResultLineRe = regexp.MustCompile(`(?m)^\s*Audit Result:.*$`)
)

// auditFindings distils one audit verdict into its findings: the non-pass
// opinions and the overall verdict line, ` | `-joined, with the echoed
// deliverable dropped.
//
// The auditor answers by echoing the producer's message back with one
// `- audit: <opinion>` sub-line under every audited line, so the deliverable is
// ~95% of the verdict string and none of it is a finding — a plain truncation of
// the verdict records the DELIVERABLE again, with the finding past the cut. A
// passing item is skipped too: it says nothing a reader would not assume from the
// round's suspect count.
func auditFindings(verdict string) string {
	findings := make([]string, 0, 4)
	for _, m := range auditOpinionRe.FindAllStringSubmatch(verdict, -1) {
		opinion := strings.TrimSpace(m[1])
		if opinion == "" || strings.EqualFold(opinion, "pass") {
			continue
		}
		findings = append(findings, opinion)
	}
	if overall := strings.TrimSpace(auditResultLineRe.FindString(verdict)); overall != "" {
		findings = append(findings, overall)
	}
	if len(findings) == 0 {
		// Every opinion passed, or the shape was not recognised — say so rather
		// than return an empty string a reader would take for "no issues".
		return "audit produced no parseable opinion"
	}
	return strings.Join(findings, " | ")
}

// recordAuditVerdict archives one audit's findings, in the same order as
// Suspects, so the counts and the auditor's own words stay in step (entry i of
// AuditVerdicts is the verdict that produced entry i of Suspects).
//
// Without this a failure shows a curve (5→3→1→0) but no reason, and the only way
// to explain the run is to reverse-engineer the shipped deliverable — which is how
// the #221 analysis twice landed on the wrong cause.
func recordAuditVerdict(audit *GateAuditRecord, verdict string) {
	if audit == nil || strings.TrimSpace(verdict) == "" {
		return
	}
	excerpt := auditFindings(verdict)
	if runes := []rune(excerpt); len(runes) > auditVerdictExcerptMax {
		excerpt = string(runes[:auditVerdictExcerptMax]) + "…"
	}
	audit.AuditVerdicts = append(audit.AuditVerdicts, excerpt)
}

// auditPayload is the ONE JSON object answer_auditor audits: the producer's
// complete FINAL message, verbatim, plus the gate's own mechanical suspicions
// about it. The question under audit is pinned into the auditor's system prompt
// at construction time and never re-sent.
type auditPayload struct {
	FinalMessage string `json:"final_message"`
	// GatePrechecks carries the gate's text-only reading of the deliverable (see
	// collectGatePrechecks) as EVIDENCE for the auditor, not as a verdict: the
	// auditor must rule on every entry, and what ships follows from its verdict.
	GatePrechecks []string `json:"gate_prechecks,omitempty"`
}

// buildAuditPayload serializes the deliverable into answer_auditor's JSON - the
// shape the audit contract advertises. It is a verbatim passthrough, not an
// extraction: the auditor reads the FINAL message's own md structure (##
// Candidate Matrix, ## Reasoning Chain, the Final/Guessed Answer line) and echoes
// it back with audit opinions, so the gate must not reshape or truncate it.
func buildAuditPayload(final string, prechecks []string) string {
	b, err := json.Marshal(auditPayload{FinalMessage: final, GatePrechecks: prechecks})
	if err != nil {
		// json.Marshal of plain strings cannot fail.
		return ""
	}
	return string(b)
}

// The gate's mechanical precheck kinds. They are record vocabulary, so they are
// stable identifiers rather than prose.
const (
	precheckUngroundedValue   = "ungrounded_answer_value"
	precheckListOnlyValue     = "list_only_answer_value"
	precheckCitationGrounding = "citation_only_grounding"
	precheckNoAnswerValue     = "answer_value_missing"
)

// collectGatePrechecks runs the gate's cheap, text-only checks over a deliverable
// that is about to be audited, returning one line per suspicion (empty when the
// deliverable looks clean) and nil when there is nothing to audit at all.
//
// These checks used to be pre-audit REJECTIONS: each one skipped the audit and
// sent the producer a directive of its own. Measured over 24 hard benchmark
// questions that cost far more than it caught - 62 rejections, 11 short-circuits,
// and the auditor never ran on more than half the set, so none of its own checks
// (grounding, slot properties, siblings) could fire. q253 is the clearest case:
// one value refused ten times in a row, the whole audit budget gone, no verdict
// produced. The question is a CONTENT judgement, and the auditor is the component
// that holds the tools to make it; the gate's reading becomes evidence for that
// judgement, and the rounds it would have refused are counted per kind.
func collectGatePrechecks(question, shipped, final, haystack string) []string {
	if strings.TrimSpace(final) == "" {
		return nil // nothing to audit; the caller asks for a deliverable
	}
	var prechecks []string
	if shipped == "" {
		prechecks = append(prechecks, precheckNoAnswerValue+
			": the deliverable carries an answer heading with NO value; the contract requires"+
			" `Final Answer: **<value>**` (or the Guessed variant) as its LAST line")
	} else {
		if !answerValueIsGrounded(shipped, haystack) {
			prechecks = append(prechecks, precheckUngroundedValue+
				": the value `"+shipped+"` appears in NO chunk this run read")
		}
		if reason := listOnlyNameReason(question, shipped, haystack); reason != "" {
			prechecks = append(prechecks, precheckListOnlyValue+": "+reason)
		}
	}
	if gaps := citationOnlyGroundings(final); len(gaps) > 0 {
		prechecks = append(prechecks, precheckCitationGrounding+
			": these lines are grounded only by a bibliographic entry: "+strings.Join(gaps, "; "))
	}
	return prechecks
}

// precheckKinds extracts each precheck's kind (its text before the first colon)
// so the record can count them per kind rather than as one lump.
func precheckKinds(prechecks []string) []string {
	kinds := make([]string, 0, len(prechecks))
	for _, p := range prechecks {
		kind, _, _ := strings.Cut(p, ":")
		kinds = append(kinds, strings.TrimSpace(kind))
	}
	return kinds
}
