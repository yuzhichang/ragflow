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
	"strings"
)

// finalAnswerValueRe extracts the value from the FOS-mandated answer line —
// `Final Answer: **<value>**` or `Guessed Answer: **<value>** (assumption: ...)`
// — tolerating heading markers, the bold tokens, and the Guessed variant's
// trailing assumption note. Trailing sentence punctuation after the closing
// bold is ALSO tolerated: models routinely write `**...15 人**。`, and the
// strict whitespace-only tail flipped finalAnswerValue to empty on the 关羽
// run — which pushed a perfectly good deliverable into finalizeAnswer, whose
// output re-rendered the whole document and shipped TWO Final Answer lines.
var finalAnswerValueRe = regexp.MustCompile(
	`(?im)^[^\S\n]*(?:#{1,4}\s*)?\**\s*(?:final|guessed)\s+answer\s*:?\s*\*{2}([^*]+?)\*{2}` +
		`(?:\s*\(assumption:[^)]*\))?[。．.，,；;！!？?\s]*$`)

// finalAnswerTwoLineRe is the value-on-the-next-line variant models emit
// routinely (`## Final Answer` then `**Bhowani Junction**`). q1228 shipped it
// and the single-line matcher above read it as an EMPTY answer, which pushed a
// perfectly retrievable deliverable into the recovery ladder instead of the
// gate's repair loop.
var finalAnswerTwoLineRe = regexp.MustCompile(
	`(?im)^[^\S\n]*(?:#{1,4}\s*)?\**\s*(?:final|guessed)\s+answer\s*:?\s*\**\s*$` +
		`\n[^\S\n]*\*{2}([^*\n][^\n]*?)\*{2}\s*$`)

// finalAnswerValue returns the value shipped on the FOS answer line, or "" when
// the reply carries none. It is FORMAT PRESENCE, not a correctness judgement —
// the gate uses it only to build the auditor payload, to decide whether a gate
// continuation may replace the standing final, and to trigger the
// finalizeAnswer terminus. Whether the value is CORRECT is exclusively
// answer_auditor's business.
func finalAnswerValue(final string) string {
	m := finalAnswerValueRe.FindStringSubmatch(final)
	if m == nil {
		// Fall back to the two-line shape before declaring the answer line
		// empty: `Final Answer:` alone on its line with the value under it is
		// a formatting habit, not a missing answer.
		if m2 := finalAnswerTwoLineRe.FindStringSubmatch(final); m2 != nil {
			return strings.TrimSpace(strings.Trim(m2[1], "*"))
		}
		return ""
	}
	return strings.TrimSpace(m[1])
}

// finalAnswerLineCount counts the FOS answer lines in s. The finalize
// synthesis must produce exactly ONE; zero (the answer line lost its shape)
// or two-plus (the model re-rendered the deliverable inside an answer line)
// marks the synthesis degenerate, and the caller ships the standing
// deliverable instead.
func finalAnswerLineCount(s string) int {
	return len(finalAnswerValueRe.FindAllString(s, -1))
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
var answerLabelRe = regexp.MustCompile(`(?i)^([^\S\n]*(?:#{1,4}\s*)?\**\s*)(final|guessed)(\s+answer\s*:?)`)

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
	// kind of fact an enumeration can never establish.
	namePropertyQuestionRe = regexp.MustCompile(`(?i)\b(birth name|full birth name|full name|given name|real name|maiden name|birth surname|née|né)\b`)
	// propertyBeforeRe / propertyAfterRe match the phrasing that STATES such a
	// property, applied to a window just before / after the value.
	propertyBeforeRe = regexp.MustCompile(`(?i)(born|née|né|real name|birth ?name|given name|full name|maiden name|surname)\s*[:,\-]?\s*$`)
	propertyAfterRe  = regexp.MustCompile(`(?i)^\s*[:,\-]?\s*(was born|is born|born)\b`)
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
	return strings.Contains(" "+normalizeForMatch(haystack)+" ", " "+normalizeForMatch(value)+" ")
}

// auditPayload is the ONE JSON object answer_auditor audits: the producer's
// complete FINAL message, verbatim. The question under audit is pinned into
// the auditor's system prompt at construction time and never re-sent.
type auditPayload struct {
	FinalMessage string `json:"final_message"`
}

// buildAuditPayload serializes the deliverable into answer_auditor's one-key
// JSON - the same shape the audit contract advertises. It is a verbatim
// passthrough, not an extraction: the auditor reads the FINAL message's own
// md structure (## Candidate Matrix, ## Reasoning Chain, the Final/Guessed
// Answer line) and echoes it back with audit opinions, so the gate must not
// reshape or truncate it.
func buildAuditPayload(final string) string {
	b, err := json.Marshal(auditPayload{FinalMessage: final})
	if err != nil {
		// json.Marshal of plain strings cannot fail.
		return ""
	}
	return string(b)
}
