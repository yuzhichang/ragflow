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

package codexagent

import (
	"encoding/json"
	"errors"
	"testing"

	codexgo "github.com/zealbase/codex-app-server-go"
)

// TestTurnErrorDetailPrefersTurnError pins that a failed turn surfaces the turn's own
// error payload (which carries the real cause, e.g. the provider's HTTP status) rather
// than the bare, often empty, `error` notification that arrives first.
func TestTurnErrorDetailPrefersTurnError(t *testing.T) {
	const detailed = `{"message":"unexpected status 404 Not Found, url: https://api.minimaxi.com/responses"}`
	turn := &codexgo.Turn{Error: json.RawMessage(detailed)}

	if got := turnErrorDetail(turn, errors.New("(code=\"\"): ")); got != detailed {
		t.Fatalf("turn error detail: got %q, want %q", got, detailed)
	}
}

func TestTurnErrorDetailFallsBackToStreamError(t *testing.T) {
	streamErr := errors.New("codexagent: the Codex server reported an error with no detail")

	if got := turnErrorDetail(nil, streamErr); got != streamErr.Error() {
		t.Fatalf("fallback to stream error: got %q, want %q", got, streamErr.Error())
	}
	if got := turnErrorDetail(&codexgo.Turn{}, streamErr); got != streamErr.Error() {
		t.Fatalf("empty turn error falls back: got %q, want %q", got, streamErr.Error())
	}
}

func TestTurnErrorDetailPlaceholder(t *testing.T) {
	if got := turnErrorDetail(nil, nil); got != "no error detail" {
		t.Fatalf("placeholder: got %q, want %q", got, "no error detail")
	}
}
