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

package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/viper"
)

// CodexConfig configures the mode 8 Codex engine: the shared Codex app-server this
// process dials over WebSocket, plus the capability-isolation posture. It does NOT
// carry a model: the model is chosen per thread from the dialog's configuration at
// thread creation (see the mode 8 plan, D4).
type CodexConfig struct {
	// Endpoint is the shared Codex app-server WebSocket URL (ws:// or wss://).
	// Empty disables mode 8 (the engine degrades to the regular pipeline).
	Endpoint string
	// BearerToken authenticates the WebSocket connection.
	BearerToken string
	// MCPPublicBase is the base URL the Codex server uses to reach this process's
	// MCP bridge, e.g. https://ragflow.internal. When empty it is derived from the
	// API server config at request time.
	MCPPublicBase string
	// WorkDir is the empty directory Codex runs in. It must exist on the machine the
	// shared Codex runs on; empty means RAGFlow creates a local temp dir per turn.
	WorkDir string
	// ApprovalPolicy is the Codex approval policy. Default "untrusted" — the most
	// restrictive usable value. Note "never" means "never ask", not "never run".
	ApprovalPolicy string
	// SandboxMode is the Codex sandbox mode. Default "read-only": mode 8 must only
	// reason and call the MCP retrieval tools, never execute or write.
	SandboxMode string
	// ResponsesProviders is the allowlist of dialog-model provider names (the
	// driver Name(), lower-case) that speak the Responses wire API Codex requires.
	// A mode 8 turn whose dialog model resolves to a provider outside this list is
	// refused and falls back to the regular pipeline (see the plan, decision U2).
	ResponsesProviders []string
	// TurnTimeout bounds one turn's wall-clock time.
	TurnTimeout time.Duration
}

const defaultCodexTurnTimeout = 30 * time.Minute

// defaultResponsesProviders are the providers known to speak the Responses API.
var defaultResponsesProviders = []string{"openai", "azure-openai", "minimax"}

func (c *Config) parseCodexConfig(v *viper.Viper) error {
	// LookupEnv preserves explicitly empty overrides (see parseMCPConfig).
	value := func(key, fallback string) string {
		if value, ok := os.LookupEnv("RAGFLOW_CODEX_" + strings.ToUpper(key)); ok {
			return value
		}
		if v.IsSet("codex." + key) {
			return v.GetString("codex." + key)
		}
		return fallback
	}

	endpoint := strings.TrimSpace(value("endpoint", ""))
	if endpoint != "" && !hasWSScheme(endpoint) {
		return fmt.Errorf("invalid codex endpoint (want ws:// or wss://): %s", endpoint)
	}

	approval := value("approval_policy", "untrusted")
	switch approval {
	case "untrusted", "on-request", "never":
	default:
		return fmt.Errorf("invalid codex approval_policy: %s", approval)
	}

	sandbox := value("sandbox_mode", "read-only")
	switch sandbox {
	case "read-only", "workspace-write", "danger-full-access":
	default:
		return fmt.Errorf("invalid codex sandbox_mode: %s", sandbox)
	}

	timeoutSeconds, err := strconv.Atoi(value("turn_timeout", strconv.Itoa(int(defaultCodexTurnTimeout.Seconds()))))
	if err != nil || timeoutSeconds <= 0 {
		return fmt.Errorf("invalid codex turn_timeout: %s", value("turn_timeout", ""))
	}

	c.apiServer.Codex = CodexConfig{
		Endpoint:           endpoint,
		BearerToken:        value("bearer_token", ""),
		MCPPublicBase:      strings.TrimRight(value("mcp_public_base", ""), "/"),
		WorkDir:            strings.TrimSpace(value("work_dir", "")),
		ApprovalPolicy:     approval,
		SandboxMode:        sandbox,
		ResponsesProviders: parseResponsesProviders(value("responses_providers", strings.Join(defaultResponsesProviders, ","))),
		TurnTimeout:        time.Duration(timeoutSeconds) * time.Second,
	}
	return nil
}

// parseResponsesProviders splits a comma-separated provider allowlist, lower-casing and
// dropping blanks. An empty result means "no restriction" (check disabled).
func parseResponsesProviders(raw string) []string {
	out := make([]string, 0, 4)
	for _, p := range strings.Split(raw, ",") {
		if p = strings.ToLower(strings.TrimSpace(p)); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func hasWSScheme(endpoint string) bool {
	return strings.HasPrefix(endpoint, "ws://") || strings.HasPrefix(endpoint, "wss://")
}
