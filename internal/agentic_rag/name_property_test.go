/*
 *  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package agentic_rag

import (
	"strings"
	"testing"
)

// q784's shape: the wrong name was one entry among many in a cast list, and the
// question asks for a birth name — which a cast list cannot state.
const castListHaystack = `<chunk chunk_id="c1" doc_id="24653">` +
	`<match_snippet>...du directeurRobert VattierGisèle GrayAndré BervilJacques DynamJacques Meyran` +
	`Félix PaquetAndrée ServilangeJean DaurandPhilippe JanvierJoe BreitbardMarcel Perès` +
	`Pierre Albert BrasseurHenri HenneryHarry MaxJean-Pierre LorrainRené PascalÉdouard Rousseau` +
	`Jacques AngelvinJean SylvainMercédès BrarePaul Deman...</match_snippet></chunk>`

func TestListOnlyNameReason(t *testing.T) {
	question := "Give me this person's full birth name. - born in the 1920s in Paris - appeared in a comedy movie in the 1950s."

	// A name picked out of an enumeration, with no property statement: rejected.
	reason := listOnlyNameReason(question, "Pierre Albert Brasseur", castListHaystack)
	if reason == "" {
		t.Fatal("a value seen only in a cast list must be rejected for a birth-name question")
	}
	if !strings.Contains(reason, "ENUMERATION") || !strings.Contains(reason, "birth name") {
		t.Fatalf("reason must name the problem and the property: %q", reason)
	}

	// The same value stated as the property: nothing to object to.
	stated := "Pierre Albert Brasseur was born in Paris in 1925, the son of a tailor."
	if got := listOnlyNameReason(question, "Pierre Albert Brasseur", stated); got != "" {
		t.Fatalf("a property-stating sentence must pass, got %q", got)
	}
	if got := listOnlyNameReason(question, "Jacqueline Georgette Cantrelle",
		"Born Jacqueline Georgette Cantrelle in Paris, she took the stage name Cantrelle."); got != "" {
		t.Fatalf("`Born <value>` must pass, got %q", got)
	}
	if got := listOnlyNameReason(question, "Jacqueline Cantrelle", "Jacqueline Cantrelle, née Georgette, died in 1994."); got != "" {
		t.Fatalf("`née` after the value must pass, got %q", got)
	}

	// The lever is scoped: no name-property question, or a non-name value, and
	// it stays silent.
	if got := listOnlyNameReason("Who directed the 1950s comedy?", "Pierre Albert Brasseur", castListHaystack); got != "" {
		t.Fatalf("a question that asks for no name property must not be gated, got %q", got)
	}
	if got := listOnlyNameReason(question, "1955", castListHaystack); got != "" {
		t.Fatalf("a numeric value must not be gated, got %q", got)
	}
	// A prose haystack without the value at all: the grounding lever owns that.
	if got := listOnlyNameReason(question, "Someone Else", "A short prose paragraph about a tailor."); got != "" {
		t.Fatalf("a value absent from the haystack must not be gated here, got %q", got)
	}
}
