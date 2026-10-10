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

package runtime

import (
	"context"
	"testing"
)

// TestUserIDFromContext pins the round-trip the Codex MCP bridge relies on: the
// ticket owner attached with WithUserID is readable downstream, and an empty id is
// left unattached (so retrieval falls back to canvas state).
func TestUserIDFromContext(t *testing.T) {
	if got := UserIDFromContext(context.Background()); got != "" {
		t.Fatalf("unset: got %q, want empty", got)
	}

	ctx := WithUserID(context.Background(), "user-1")
	if got := UserIDFromContext(ctx); got != "user-1" {
		t.Fatalf("set: got %q, want %q", got, "user-1")
	}

	if got := UserIDFromContext(WithUserID(ctx, "   ")); got != "user-1" {
		t.Fatalf("blank id must not overwrite: got %q", got)
	}
}

func TestUserIDFromContextNil(t *testing.T) {
	//nolint:staticcheck // an explicit nil ctx must not panic
	if got := UserIDFromContext(nil); got != "" {
		t.Fatalf("nil ctx: got %q, want empty", got)
	}
}
