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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"regexp"
	"sort"
	"strings"
)

// citationMarkerRe matches the backend-assigned [ID:N] citation markers the chat service
// inserts into the final answer before it is persisted. The service inserts them as
// " [ID:N]" (leading space), so the optional whitespace is part of the match — otherwise
// "chunk_id: c1" and "chunk_id: c1 [ID:0]" would still hash differently.
var citationMarkerRe = regexp.MustCompile(`\s*\[ID:\d+\]`)

// MessagesToItems converts RAGFlow conversation messages ({role, content}) into the raw
// Responses API items Codex accepts via thread/inject_items.
//
// system messages are skipped: Codex carries its instructions on the thread, not as
// injected history. Blank-content messages are dropped (an empty input_text item is
// rejected upstream). The caller passes the history WITHOUT the current user message —
// that one is the turn input, not a replay.
func MessagesToItems(messages []map[string]interface{}) []json.RawMessage {
	items := make([]json.RawMessage, 0, len(messages))
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		content, _ := msg["content"].(string)
		content = strings.TrimSpace(content)
		if content == "" {
			continue
		}
		var item map[string]interface{}
		switch role {
		case "user":
			item = responsesMessageItem("user", "input_text", content)
		case "assistant":
			item = responsesMessageItem("assistant", "output_text", content)
		default:
			continue
		}
		if raw, err := json.Marshal(item); err == nil {
			items = append(items, raw)
		}
	}
	return items
}

func responsesMessageItem(role, textType, text string) map[string]interface{} {
	return map[string]interface{}{
		"type": "message",
		"role": role,
		"content": []map[string]interface{}{
			{"type": textType, "text": text},
		},
	}
}

// ScopeFingerprint is a stable hash of the tenant + dataset scope. A change (KB added,
// removed or rebound, or tenant changed) means the thread's baked-in MCP URL no longer
// matches the conversation and the thread must be rebuilt.
func ScopeFingerprint(tenantID string, datasetIDs []string) string {
	sorted := append([]string(nil), datasetIDs...)
	sort.Strings(sorted)
	h := sha256.New()
	h.Write([]byte(tenantID))
	h.Write([]byte{0})
	h.Write([]byte(strings.Join(sorted, "\x1f")))
	return hex.EncodeToString(h.Sum(nil))
}

// HistoryFingerprint is a stable hash of the visible conversation history. It detects
// when the thread's history has drifted from RAGFlow's (a deleted message, a turn that
// ran on a different engine, an out-of-band edit), which requires a rebuild + replay.
//
// Backend-assigned [ID:N] citation markers are stripped first: the chat service persists
// the decorated answer (raw answer + markers), while a fresh turn records the raw answer,
// so without this a citation-bearing conversation would look drifted and rebuild forever.
func HistoryFingerprint(messages []map[string]interface{}) string {
	h := sha256.New()
	for _, msg := range messages {
		role, _ := msg["role"].(string)
		content, _ := msg["content"].(string)
		h.Write([]byte(role))
		h.Write([]byte{0})
		h.Write([]byte(citationMarkerRe.ReplaceAllString(content, "")))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// CombineFingerprints folds several fingerprints into one 64-char hash, so a fingerprint
// that composes other fingerprints still fits a size:64 column.
func CombineFingerprints(parts ...string) string {
	h := sha256.New()
	for _, p := range parts {
		h.Write([]byte(p))
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// HistoryWithoutCurrent returns the messages excluding the final one, which the caller
// treats as the current turn input. It is the history a fresh thread must be seeded with.
func HistoryWithoutCurrent(messages []map[string]interface{}) []map[string]interface{} {
	if len(messages) == 0 {
		return nil
	}
	return messages[:len(messages)-1]
}

// InstructionsFingerprint is a stable hash of a thread's base instructions. It is folded
// into the scope fingerprint because the instructions are baked into the thread at create
// time and thread/resume does not reload them: a change (e.g. toggling the citation
// contract) must force a rebuild.
func InstructionsFingerprint(instructions string) string {
	sum := sha256.Sum256([]byte(instructions))
	return hex.EncodeToString(sum[:])
}
