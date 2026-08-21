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
