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
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/adk/session"
	einocommon "github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/components/tool"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	"go.uber.org/zap"

	"ragflow/internal/common"
)

// questionDecompositionTemplateID selects the decomposition agent from
// conf/agentic_rag.yaml (tools: [check_decomposition]; content holds the
// planner prompt with the worked examples).
const questionDecompositionTemplateID = "question-decomposition"

// decompositionNoPlanMarker is what the decomposition agent returns for a
// purely conversational message - nothing that needs facts, so there is no
// plan to pin and the explorer answers it per its own Intent rule.
const decompositionNoPlanMarker = "NO-DECOMPOSITION"

// decompositionMaxIterations bounds the planner's own ReAct loop: it writes
// one plan and runs the checker a few times, so twelve rounds are generous.
const decompositionMaxIterations = 12

// decompositionMaxRepairRounds caps the Go-side repair loop: after the agent
// returns a plan, checkDecomposition is run HERE (deterministic, free - the
// adoption of the in-agent checker cannot be taken on trust), and a plan that
// still carries findings is sent back for repair at most this many times.
// An exhausted budget pins the last plan unverified: availability over
// perfection - the explorer works the plan as written and the delivery gate
// sees the outcome.
const decompositionMaxRepairRounds = 2

// decompositionPlanHeadRe finds the first block heading in the agent's final
// message: the plan is everything from there down (the agent may fence the
// plan or prefix one short line; the trailing fence is trimmed).
var decompositionPlanHeadRe = regexp.MustCompile(`(?m)^### Sub-question`)

// NewQuestionDecompositionAgent builds the stage-one agent: same construction
// as the answer auditor (a standalone ChatModelAgent over its own template),
// with one difference - its toolset holds ONLY the checker, because the one
// thing this stage must never do is retrieve.
func NewQuestionDecompositionAgent(
	ctx context.Context,
	model einocommon.BaseChatModel,
	tenantID string,
	datasetIDs []string,
	question string,
	toolDurations *durationAccumulator,
) (*adk.ChatModelAgent, error) {
	tmpl, err := resolveTemplateFor(questionDecompositionTemplateID)
	if err != nil {
		return nil, fmt.Errorf("question decomposition: %w", err)
	}
	// Same tool wrapping as the auditor: the checker's calls land in the run's
	// shared duration ledger, so per-question cost accounting covers this
	// stage too (the COUNT comes from consumeAgentEvents' tally instead).
	tools := toolsFor(tmpl, tenantID, datasetIDs)
	if toolDurations != nil {
		wrapped := make([]tool.BaseTool, len(tools))
		for i, t := range tools {
			if it, ok := t.(tool.InvokableTool); ok {
				wrapped[i] = &instrumentedTool{InvokableTool: it, acc: toolDurations}
			} else {
				wrapped[i] = t
			}
		}
		tools = wrapped
	}
	cfg := &adk.ChatModelAgentConfig{
		Name:        tmpl.ID,
		Description: tmpl.Description,
		// Same pinning as the auditor: the question lives in the instruction,
		// so every turn's input is just the directive.
		Instruction:      tmpl.Content + "\n\n## The question to decompose\n\n" + question,
		Model:            model,
		MaxIterations:    decompositionMaxIterations,
		ModelRetryConfig: agentModelRetryConfig(),
		ToolsConfig: adk.ToolsConfig{
			ToolsNodeConfig: compose.ToolsNodeConfig{
				Tools:               tools,
				ExecuteSequentially: false,
			},
		},
	}
	agent, err := adk.NewChatModelAgent(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("question decomposition: build agent: %w", err)
	}
	return agent, nil
}

// runDecompositionStage runs stage one end to end and returns the plan text to
// pin into the explorer's context, or "" when there is nothing to pin: the
// message was purely conversational, the agent returned no readable plan, or
// the stage failed (availability over perfection - the explorer then works
// without a pinned plan exactly as it did before this stage existed).
//
// The agent's own checker calls are only half the verification: after every
// agent round the plan is re-checked HERE with the same deterministic
// checkDecomposition the tool wraps, because two measured runs (#11/#12) showed
// an in-prompt call contract the model simply never exercised. A plan that
// still carries findings after the repair budget is pinned anyway, with a
// warning in the log - the explorer is told the plan may be imperfect.
func runDecompositionStage(ctx context.Context, in Input, question string) string {
	if strings.TrimSpace(question) == "" {
		return ""
	}
	agent, err := NewQuestionDecompositionAgent(ctx, in.Model, in.TenantID, in.DatasetIDs, question, in.ToolCallDurations)
	if err != nil {
		common.WarnCtx(ctx, "agentic_rag: question-decomposition agent unavailable", zap.Error(err))
		return ""
	}
	// The stage owns a private conversation: one planner session per Run, no
	// history reuse (the explorer's session semantics do not apply here).
	store := session.NewInMemoryStore[adk.Message](nil)
	conv := newConversation("decomposition", store)

	directive := "Decompose the question pinned in your instructions."
	plan := ""
	for round := 0; round <= decompositionMaxRepairRounds; round++ {
		iter := conv.runner(ctx, agent, false).Run(ctx, []adk.Message{schema.UserMessage(directive)})
		final, _, err := consumeAgentEvents(ctx, iter, func(string, string) {}, in.ToolCallCounts, nil, nil)
		if err != nil {
			conv.discardFailedTurn(ctx, conv.head(ctx))
			common.WarnCtx(ctx, "agentic_rag: question-decomposition run failed", zap.Error(err))
			return ""
		}
		plan = extractPlan(final)
		if plan == "" {
			if strings.Contains(final, decompositionNoPlanMarker) {
				common.InfoCtx(ctx, "agentic_rag: question-decomposition skipped (conversational message)")
			} else {
				common.WarnCtx(ctx, "agentic_rag: question-decomposition returned no readable plan")
			}
			return ""
		}
		findings := checkDecomposition(plan)
		if len(findings) == 0 {
			common.InfoCtx(ctx, "agentic_rag: question-decomposition verified",
				zap.Int("rounds", round+1), zap.Int("plan_bytes", len(plan)))
			return plan
		}
		common.WarnCtx(ctx, "agentic_rag: question-decomposition plan failed the mechanical check",
			zap.Int("round", round+1), zap.Strings("findings", findings))
		directive = "check_decomposition reports:\n- " + strings.Join(findings, "\n- ") +
			"\n\nFix every finding and return ONLY the corrected plan."
	}
	return plan
}

// extractPlan returns the plan portion of the agent's final message: everything
// from the first `### Sub-question` heading down, with a trailing code fence
// trimmed. Empty when the message carries no plan at all.
func extractPlan(final string) string {
	i := strings.Index(final, "### Sub-question")
	if i < 0 {
		return ""
	}
	plan := strings.TrimSpace(final[i:])
	plan = strings.TrimSuffix(plan, "```")
	return strings.TrimSpace(plan)
}
