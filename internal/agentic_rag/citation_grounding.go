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
	"regexp"
	"strings"
)

// This is the MECHANICAL half of the sibling/family discipline. The auditor is
// asked (in its prompt) to widen its own read and to reject a `Tested` line that
// only NAMES its candidate, but that is a model rule: whether it fires depends on
// the round. Measured over 17 q221 runs, the wrong member of a two-book family
// shipped 10 times, and the passes that let it through had a shape the gate can
// see for itself — the candidate was "grounded" by a BIBLIOGRAPHIC ENTRY, i.e. a
// citation that names the work and states nothing about it:
//
//	Tested: "Opium: A Portrait of the Heavenly Demon" (1999) - clue [1, 2, 3] supported
//	  (doc: 59224.md, chunk_id: a8d8adae46b0e123, snippet: "Hodgson, Barbara. Opium:
//	   A Portrait of the Heavenly Demon. San Francisco: Chronicle Books. 1999.")
//
// Both siblings can be "supported" that way by the SAME Sources chunk, which is why
// the matrix then reads as a two-candidate tie broken by title preference. A
// citation proves a work exists; it cannot establish what the work is about, so a
// line leaning on one is not evidence and the gate rejects it before spending an
// audit on it.

// citationSnippetRe matches the opening of a bibliographic entry — `Surname,
// Forename` — which is what a Sources/bibliography section looks like and what
// prose almost never looks like.
var citationSnippetRe = regexp.MustCompile(`^[\p{Lu}][\p{L}'’\-]+,\s+\S`)

// citationYearTailRe matches an entry that ENDS on its date, optionally closed by
// a period or a page range.
var citationYearTailRe = regexp.MustCompile(`(?:19|20)\d{2}(?:[.,]?\s*\d*[-–]\d*)?\.?\s*$`)

// lastYearRe finds the entry's closing year, so the sentence-break count is taken
// before it (a title containing a year must not inflate the count).
var lastYearRe = regexp.MustCompile(`(?:19|20)\d{2}`)

// isCitationShapedSnippet reports whether a cited snippet is a bibliographic
// entry rather than a passage about the work.
//
// Three signals, all required, keep it from firing on prose that merely mentions a
// work: it opens on `Surname, Forename`, it ends on a year, and it carries at
// least two `". "` breaks before that year — the `Author. Title. Place:
// Publisher, Year.` skeleton. Prose like `Hodgson, Barbara wrote Opium in 1999.`
// passes the first two and fails the third.
func isCitationShapedSnippet(snippet string) bool {
	s := strings.TrimSpace(snippet)
	if s == "" || !citationSnippetRe.MatchString(s) || !citationYearTailRe.MatchString(s) {
		return false
	}
	years := lastYearRe.FindAllStringIndex(s, -1)
	if len(years) == 0 {
		return false
	}
	return strings.Count(s[:years[len(years)-1][0]], ". ") >= 2
}

// candidateOfGroundingLine extracts the candidate a Tested/Retained line claims,
// stripped of the markdown a deliverable wraps it in.
func candidateOfGroundingLine(line string) string {
	at, labelLen := -1, 0
	for _, label := range []string{"Tested:", "Retained:"} {
		if j := strings.Index(line, label); j >= 0 && (at < 0 || j < at) {
			at, labelLen = j, len(label)
		}
	}
	if at < 0 {
		return ""
	}
	rest := line[at+labelLen:]
	// The candidate runs to the field separator (` - `) or to the parenthesised
	// field list, whichever comes first.
	for _, cut := range []string{" - ", " — ", " (doc:", " (`doc:", " ("} {
		if j := strings.Index(rest, cut); j >= 0 {
			rest = rest[:j]
		}
	}
	return strings.Trim(strings.TrimSpace(rest), "*`\"'“” ")
}

// snippetOfGroundingLine extracts the line's quoted snippet.
func snippetOfGroundingLine(line string) string {
	i := strings.Index(line, "snippet:")
	if i < 0 {
		return ""
	}
	rest := line[i+len("snippet:"):]
	rest = strings.TrimLeft(rest, " \t*`")
	if !strings.HasPrefix(rest, "\"") {
		return ""
	}
	rest = rest[1:]
	if j := strings.Index(rest, "\""); j >= 0 {
		return rest[:j]
	}
	return rest
}

// titleLikeCandidate reports whether a candidate is a title/name rather than a
// bare number or year: those are the values a citation can legitimately carry
// (its own date), so the check must not fire on them.
func titleLikeCandidate(candidate string) bool {
	c := strings.TrimSpace(candidate)
	if c == "" {
		return false
	}
	if hasNoLetters(c) {
		return false
	}
	return strings.Contains(c, ":") || len(strings.Fields(c)) >= 2
}

// citationOnlyGroundings returns the blocked-by-citation lines in the deliverable:
// a Tested/Retained line whose cited snippet is a bibliographic entry that names
// the line's own candidate. Empty when there are none.
//
// The `snippet names the candidate` part is what keeps it precise: a citation
// carried on a line about something ELSE (a year clue, an author's dates) is not
// this defect, because the entry's date can genuinely be that line's evidence.
func citationOnlyGroundings(final string) []string {
	var out []string
	for _, line := range strings.Split(final, "\n") {
		candidate := candidateOfGroundingLine(line)
		if !titleLikeCandidate(candidate) {
			continue
		}
		snippet := snippetOfGroundingLine(line)
		if !isCitationShapedSnippet(snippet) {
			continue
		}
		normCandidate := normalizeForMatch(candidate)
		if normCandidate == "" || !strings.Contains(normalizeForMatch(snippet), normCandidate) {
			continue
		}
		where := ""
		if i := strings.Index(line, "chunk_id:"); i >= 0 {
			rest := strings.TrimSpace(line[i+len("chunk_id:"):])
			if j := strings.IndexAny(rest, ",)`"); j >= 0 {
				rest = rest[:j]
			}
			where = " (chunk " + strings.TrimSpace(rest) + ")"
		}
		out = append(out, candidate+where)
	}
	return out
}

// citationGroundingDirective is the repair instruction for the check above. It
// names the offending lines and the two lawful repairs.
func citationGroundingDirective(lines []string) string {
	return ("These candidate lines are grounded by a BIBLIOGRAPHIC ENTRY, which names the work and states " +
		"nothing about it: " + strings.Join(lines, "; ") + ". A citation proves a work EXISTS - it cannot " +
		"establish what the work is about, who wrote it in what capacity, or which constraint it satisfies, " +
		"and it cannot tell two sibling works apart (the same Sources chunk carries both). Widen the read of " +
		"that document (list_chunks with number_neighbors, or chained anchored calls) or locate a chunk whose " +
		"TEXT states the claim, and cite that chunk on the line instead; keep the citation only as corroboration " +
		"for the work's existence, never as a line's support. If no chunk states the retained work's fitness " +
		"while one states a sibling's, the evidence points at the sibling - ship that, or ship a DECLARED TIE " +
		"(a `(tie: ...)` clause per rival on the answer line), never the title you happen to prefer.")
}

// hasNoLetters reports whether a string carries no letters at all (a pure number,
// year, percentage or date) — the same notion the FOS schema uses to tell a numeric
// answer from a named one.
func hasNoLetters(s string) bool {
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r > 0x7f {
			return false
		}
	}
	return true
}
