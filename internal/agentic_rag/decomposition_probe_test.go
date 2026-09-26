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
//  See the specific language governing permissions and limitations under
//  the License.

package agentic_rag

// Independent stage-one probe: run ONLY the question-decomposition stage -
// planner, check_decomposition, and the review round - against a live model.
// No explorer, no auditor, no corpus, no retrieval: the plan and the review's
// verdict lines are the entire output, so decomposition quality (clause
// coverage, root richness, pair clauses) can be iterated on without paying for
// an end-to-end smoke run.
//
// Guarded by DECOMP_PROBE=1: ordinary `go test ./...` never reaches the wire.
//
//	DECOMP_PROBE=1 \
//	DECOMP_QUESTION="Two individuals from different industries ..." \
//	DECOMP_API_KEY="..." \
//	DECOMP_BASE_URL="https://api.minimax.chat" \
//	DECOMP_TEMPERATURE=0.5 \
//	go test ./internal/agentic_rag/ -run TestDecompositionProbe -v -timeout 10m
//
// DECOMP_QUESTIONS_FILE (one question per line) sweeps several questions in
// one go. The MiniMax driver is hardcoded because that is the benchmark
// provider; other providers only need the driver swapped here.

import (
	"bufio"
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"ragflow/internal/entity/models"
)

func TestDecompositionProbe(t *testing.T) {
	if os.Getenv("DECOMP_PROBE") != "1" {
		t.Skip("stage-one live probe: set DECOMP_PROBE=1 to run")
	}
	apiKey := os.Getenv("DECOMP_API_KEY")
	if apiKey == "" {
		t.Fatal("DECOMP_API_KEY is required")
	}
	questions := probeQuestions(t)

	baseURL := envOr("DECOMP_BASE_URL", "https://api.minimax.chat")
	modelName := envOr("DECOMP_MODEL", "MiniMax-M3")
	temp := 0.5
	if s := os.Getenv("DECOMP_TEMPERATURE"); s != "" {
		temp = parseProbeFloat(t, s)
	}

	driver := models.NewMinimaxModel(
		map[string]string{"default": baseURL},
		models.URLSuffix{
			Chat:   "v1/text/chatcompletion_v2",
			Models: "v1/models",
			Files:  "v1/files/list",
		},
	)
	name := modelName
	cm := models.NewChatModel(driver, &name, &models.APIConfig{ApiKey: &apiKey})
	eino := models.NewEinoChatModel(cm, &models.ChatConfig{Temperature: &temp})

	for i, question := range questions {
		question = strings.TrimSpace(question)
		if question == "" {
			continue
		}
		t.Run(qLabel(i, question), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
			defer cancel()
			in := Input{
				Model:             eino,
				TenantID:          "probe",
				ToolCallCounts:    map[string]int{},
				ToolCallDurations: NewDurationAccumulator(),
			}
			plan, conversational := runDecompositionStage(ctx, in, question)
			if conversational {
				t.Logf("stage one ruled the message conversational; no plan")
				return
			}
			if plan == "" {
				t.Fatal("stage one returned no plan (check the server-side warnings above)")
			}
			t.Logf("PLAN (%d bytes):\n%s", len(plan), plan)
			if findings := checkDecomposition(plan); len(findings) > 0 {
				t.Errorf("plan carries %d mechanical findings:", len(findings))
				for _, f := range findings {
					t.Errorf("  - %s", f)
				}
			} else {
				t.Logf("mechanical check: clean")
			}
		})
	}
}

// probeQuestions assembles the probe's questions: DECOMP_QUESTION first, then
// every non-empty line of DECOMP_QUESTIONS_FILE.
func probeQuestions(t *testing.T) []string {
	t.Helper()
	var questions []string
	if q := os.Getenv("DECOMP_QUESTION"); q != "" {
		questions = append(questions, q)
	}
	if path := os.Getenv("DECOMP_QUESTIONS_FILE"); path != "" {
		f, err := os.Open(path)
		if err != nil {
			t.Fatalf("DECOMP_QUESTIONS_FILE: %v", err)
		}
		defer f.Close()
		scanner := bufio.NewScanner(f)
		for scanner.Scan() {
			if q := strings.TrimSpace(scanner.Text()); q != "" {
				questions = append(questions, q)
			}
		}
	}
	if len(questions) == 0 {
		t.Fatal("set DECOMP_QUESTION or DECOMP_QUESTIONS_FILE")
	}
	return questions
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func parseProbeFloat(t *testing.T, s string) float64 {
	t.Helper()
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("DECOMP_TEMPERATURE: %v", err)
	}
	return f
}

func qLabel(i int, question string) string {
	r := []rune(question)
	if len(r) > 40 {
		r = r[:40]
	}
	return strings.Join([]string{"q#", strconv.Itoa(i + 1), ":", string(r)}, " ")
}
