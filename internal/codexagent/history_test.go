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
	"testing"
)

func TestMessagesToItems(t *testing.T) {
	items := MessagesToItems([]map[string]interface{}{
		{"role": "system", "content": "ignored"},
		{"role": "user", "content": "hello"},
		{"role": "assistant", "content": "hi there"},
		{"role": "user", "content": "   "},
		{"role": "tool", "content": "dropped"},
	})
	if len(items) != 2 {
		t.Fatalf("want 2 items (user+assistant), got %d: %s", len(items), items)
	}
	var first map[string]interface{}
	if err := json.Unmarshal(items[0], &first); err != nil {
		t.Fatal(err)
	}
	if first["type"] != "message" || first["role"] != "user" {
		t.Fatalf("user item shape: %v", first)
	}
	content, _ := first["content"].([]interface{})
	part, _ := content[0].(map[string]interface{})
	if part["type"] != "input_text" || part["text"] != "hello" {
		t.Fatalf("user content part: %v", part)
	}

	var second map[string]interface{}
	_ = json.Unmarshal(items[1], &second)
	content2, _ := second["content"].([]interface{})
	part2, _ := content2[0].(map[string]interface{})
	if part2["type"] != "output_text" || part2["text"] != "hi there" {
		t.Fatalf("assistant content part: %v", part2)
	}
}

func TestScopeFingerprint(t *testing.T) {
	a := ScopeFingerprint("t1", []string{"kb1", "kb2"})
	b := ScopeFingerprint("t1", []string{"kb2", "kb1"})
	if a != b {
		t.Fatal("scope fingerprint must be order-independent")
	}
	if a == ScopeFingerprint("t2", []string{"kb1", "kb2"}) {
		t.Fatal("different tenant must change the fingerprint")
	}
	if a == ScopeFingerprint("t1", []string{"kb1"}) {
		t.Fatal("removing a KB must change the fingerprint")
	}
	if ScopeFingerprint("t1", nil) == ScopeFingerprint("t1", []string{"kb1"}) {
		t.Fatal("no-KB must differ from one-KB")
	}
}

func TestHistoryFingerprintDetectsDrift(t *testing.T) {
	base := []map[string]interface{}{
		{"role": "user", "content": "q1"},
		{"role": "assistant", "content": "a1"},
	}
	same := []map[string]interface{}{
		{"role": "user", "content": "q1"},
		{"role": "assistant", "content": "a1"},
	}
	if HistoryFingerprint(base) != HistoryFingerprint(same) {
		t.Fatal("identical histories must hash equally")
	}
	// A deleted message (as DeleteSessionMessage would produce) must drift.
	deleted := []map[string]interface{}{{"role": "assistant", "content": "a1"}}
	if HistoryFingerprint(base) == HistoryFingerprint(deleted) {
		t.Fatal("a removed message must change the fingerprint")
	}
	edited := []map[string]interface{}{
		{"role": "user", "content": "q1"},
		{"role": "assistant", "content": "edited"},
	}
	if HistoryFingerprint(base) == HistoryFingerprint(edited) {
		t.Fatal("an edited message must change the fingerprint")
	}
}

func TestHistoryWithoutCurrent(t *testing.T) {
	if HistoryWithoutCurrent(nil) != nil {
		t.Fatal("nil history stays nil")
	}
	msgs := []map[string]interface{}{
		{"role": "user", "content": "old"},
		{"role": "assistant", "content": "old-a"},
		{"role": "user", "content": "current"},
	}
	got := HistoryWithoutCurrent(msgs)
	if len(got) != 2 || got[1]["content"] != "old-a" {
		t.Fatalf("HistoryWithoutCurrent = %v", got)
	}
}
