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
	"ragflow/internal/entity/models"

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

// plannerTemplateID selects the planner agent from conf/agentic_rag.yaml
// (tools: [check_decomposition]; content holds the planner prompt with the
// worked examples).
const plannerTemplateID = "planner"

// PlanAuditorTemplateID selects the independent plan auditor: a separate
// agent that holds the question and the plan and audits the plan against the
// question clause by clause. The planner auditing its own plan is structurally
// blind to the defects it introduces - #36's self-review pruned four
// discriminating clauses and its own clause mapping went blind with them -
// while the one round the delivery-style grading was applied handed q875 its
// first win. Exported: the chat pipeline builds the auditor's model instance.
const PlanAuditorTemplateID = "plan_auditor"

// decompositionNoPlanMarker is what the decomposition agent returns for a
// purely conversational message - nothing that needs facts, so there is no
// plan to pin and the explorer answers it per its own Intent rule.
const decompositionNoPlanMarker = "NO-DECOMPOSITION"

// decompositionMaxIterations bounds the planner's own ReAct loop: it writes
// one plan, calls the checker once per revision and emits the plan once the
// checker is clean. Twenty leaves room for a duplicate call and two repair
// rounds - twelve was measured to be too tight (q1005, smoke #16: the planner
// called the checker twice per iteration and ran out of iterations with a
// clean plan in hand, so the stage returned nothing).
const decompositionMaxIterations = 20

// decompositionMaxRepairRounds caps the Go-side repair loop: after the agent
// returns a plan, checkDecomposition is run HERE (deterministic, free - the
// adoption of the in-agent checker cannot be taken on trust), and a plan that
// still carries findings is sent back for repair at most this many times.
// An exhausted budget pins the last plan unverified: availability over
// perfection - the explorer works the plan as written and the delivery gate
// sees the outcome.
const decompositionMaxRepairRounds = 2

// decompositionAuditRounds caps the audit-driven repair loop that follows a
// mechanically clean plan: each round is one plan-auditor turn plus one
// planner repair turn, and both are the cheapest turns in the pipeline - no
// retrieval, no chunk reads, a few KB of prompt - while the plan is the
// single input the explorer, the auditor and the delivery gate all work
// from. Eight audits beat one shipped plan that drops clauses: the budget is
// deliberately generous because the stage's tokens cost a fraction of the
// explorer's.
const decompositionAuditRounds = 8

// decompositionPlanHeadRe finds the first block heading in the agent's final
// message: the plan is everything from there down (the agent may fence the
// plan or prefix one short line; the trailing fence is trimmed).
var decompositionPlanHeadRe = regexp.MustCompile(`(?m)^### Sub-question`)

// NewPlannerAgent builds the stage-one agent: same construction
// as the answer auditor (a standalone ChatModelAgent over its own template),
// with one difference - its toolset holds ONLY the checker, because the one
// thing this stage must never do is retrieve.
func NewPlannerAgent(
	ctx context.Context,
	model einocommon.BaseChatModel,
	tenantID string,
	datasetIDs []string,
	question string,
	toolDurations *durationAccumulator,
) (*adk.ChatModelAgent, error) {
	tmpl, err := resolveTemplateFor(plannerTemplateID)
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

// runDecompositionStage runs stage one end to end. It returns the plan text to
// pin into the explorer's context, or "" with conversational=true when the
// stage ruled the message purely conversational - the caller then skips the
// explorer-auditor pipeline entirely and answers directly. It also returns ""
// (conversational=false) when there is a plan to pin but the stage failed or
// returned nothing readable: the explorer then works unpinned, exactly as it
// did before this stage existed (availability over perfection).
//
// The agent's own checker calls are only half the verification: after every
// agent round the plan is re-checked HERE with the same deterministic
// checkDecomposition the tool wraps, because two measured runs (#11/#12) showed
// an in-prompt call contract the model simply never exercised. A plan that
// still carries findings after the repair budget is pinned anyway, with a
// warning in the log - the explorer is told the plan may be imperfect.
// resolvedQuestionHeader is the line a plan opens with when the pinned
// question needed resolving against the conversation's earlier turns (see the
// "The resolved question" section of the question-decomposition template).
// Everything downstream — the explorer's input, the auditor's pinned
// question — reads THIS line as the question, never the raw fragment.
const resolvedQuestionHeader = "## Resolved question"

// extractResolvedQuestion returns the self-contained question the plan opens
// with, or "" when the plan carries no such line (single-turn runs restate
// the pinned question verbatim, and a planner that skipped the line leaves
// the caller falling back to the raw last user message).
func extractResolvedQuestion(plan string) string {
	i := strings.Index(plan, resolvedQuestionHeader)
	if i < 0 {
		return ""
	}
	rest := strings.TrimPrefix(plan[i+len(resolvedQuestionHeader):], ":")
	if nl := strings.IndexAny(rest, "\r\n"); nl >= 0 {
		rest = rest[:nl]
	}
	return strings.TrimSpace(rest)
}

// runDecompositionStage derives its inputs from the conversation itself: the
// question is the last user message, and the turns before it are the context
// a follow-up's references resolve against (priorTurns). No caller-side
// splitting of in.Messages — the stage owns the reading of its input.
func runDecompositionStage(ctx context.Context, in Input) (string, bool) {
	question := lastUserQuestion(in.Messages)
	prior := priorTurns(in.Messages)
	if strings.TrimSpace(question) == "" {
		return "", false
	}
	agent, err := NewPlannerAgent(ctx, in.Model, in.TenantID, in.DatasetIDs, question, in.ToolCallDurations)
	if err != nil {
		common.WarnCtx(ctx, "agentic_rag: question-decomposition agent unavailable", zap.Error(err))
		return "", false
	}
	// The stage owns a private conversation: one planner session per Run, no
	// history reuse (the explorer's session semantics do not apply here).
	store := session.NewInMemoryStore[adk.Message](nil)
	conv := newConversation("decomposition", store)

	directive := "Decompose the question pinned in your instructions."
	plan := ""
	for round := 0; round <= decompositionMaxRepairRounds+decompositionAuditRounds; round++ {
		input := []adk.Message{schema.UserMessage(directive)}
		if round == 0 && len(prior) > 0 {
			// A multi-turn conversation: the turns before the current question
			// ride along with the first directive, so the planner can resolve
			// the references the pinned fragment makes (it sees no other
			// copy of the history — its session starts empty). Repair and
			// review rounds run on the directive alone: the context is
			// already in the session.
			//
			// An empty message trips some provider APIs (the same reason
			// turnMessages guards its synthetic assistant line), so blank
			// history entries are dropped rather than replayed.
			input = make([]adk.Message, 0, len(prior)+1)
			for _, m := range prior {
				if m != nil && strings.TrimSpace(m.Content) != "" {
					input = append(input, m)
				}
			}
			input = append(input, schema.UserMessage(directive))
		}
		iter := conv.runner(ctx, agent, false).Run(ctx, input)
		final, _, err := consumeAgentEvents(ctx, iter, func(string, string) {}, in.ToolCallCounts, nil, nil)
		if err != nil {
			conv.discardFailedTurn(ctx, conv.head(ctx))
			common.WarnCtx(ctx, "agentic_rag: question-decomposition run failed", zap.Error(err))
			return "", false
		}
		plan = extractPlan(final)
		if plan == "" {
			if strings.Contains(final, decompositionNoPlanMarker) {
				common.InfoCtx(ctx, "agentic_rag: question-decomposition skipped (conversational message)")
				return "", true
			}
			common.WarnCtx(ctx, "agentic_rag: question-decomposition returned no readable plan")
			return "", false
		}
		findings := checkDecomposition(plan)
		if len(findings) == 0 {
			// Mechanically clean is not yet audited: the independent plan
			// auditor holds the resolved question and walks it clause by
			// clause, so a clause the planner dropped or pruned surfaces as
			// UNMAPPED - the defect class the self-review was structurally
			// blind to (#36: the self-review itself pruned c7-c10 and its own
			// mapping went blind with them). Findings ride back to the
			// planner as the next directive; the loop budget is generous
			// because plan-stage turns are the cheapest in the pipeline.
			auditedQuestion := extractResolvedQuestion(plan)
			if auditedQuestion == "" {
				auditedQuestion = question
			}
			audited := runPlanAuditor(ctx, in, plan, auditedQuestion)
			if len(audited) == 0 {
				common.InfoCtx(ctx, "agentic_rag: question-decomposition verified",
					zap.Int("rounds", round+1), zap.Int("plan_bytes", len(plan)))
				return plan, false
			}
			common.WarnCtx(ctx, "agentic_rag: plan audit reported findings",
				zap.Int("round", round+1), zap.Strings("findings", audited))
			directive = "The independent plan auditor reports:\n- " + strings.Join(audited, "\n- ") +
				"\n\nFix every finding and return ONLY the corrected plan."
			continue
		}
		common.WarnCtx(ctx, "agentic_rag: question-decomposition plan failed the mechanical check",
			zap.Int("round", round+1), zap.Strings("findings", findings))
		directive = "check_decomposition reports:\n- " + strings.Join(findings, "\n- ") +
			"\n\nFix every finding and return ONLY the corrected plan."
	}
	common.WarnCtx(ctx, "agentic_rag: question-decomposition audit budget exhausted - keeping the last mechanically valid plan")
	return plan, false
}

// NewPlanAuditorAgent builds the independent plan auditor: a separate agent
// that holds the resolved question and the plan and audits the plan against
// the question clause by clause (see the plan_auditor template). It is a
// different agent from the planner on purpose - the planner auditing its own
// plan is structurally blind to the defects it introduces: #36's self-review
// pruned four discriminating clauses (c7-c10) and its own clause mapping went
// blind with them, so the plan shipped without the question's second half.
// It owns the checker so its mechanical claims are grounded in the same tool
// the planner used, and it never rewrites - findings only.
func NewPlanAuditorAgent(
	ctx context.Context,
	model einocommon.BaseChatModel,
	tenantID string,
	datasetIDs []string,
	question string,
	plan string,
	toolDurations *durationAccumulator,
) (*adk.ChatModelAgent, error) {
	tmpl, err := resolveTemplateFor(PlanAuditorTemplateID)
	if err != nil {
		return nil, fmt.Errorf("plan auditor: %w", err)
	}
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
		// Both artifacts pinned: the resolved question the mapping walks and
		// the plan the mapping walks it against.
		Instruction:      tmpl.Content + "\n\n## Resolved question under audit\n\n" + question + "\n\n## The plan under audit\n\n" + plan,
		Model:            model,
		MaxIterations:    decompositionAuditMaxIterations,
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
		return nil, fmt.Errorf("plan auditor: build agent: %w", err)
	}
	return agent, nil
}

// decompositionAuditMaxIterations bounds the plan auditor's own ReAct loop:
// it reads the two pinned artifacts, optionally re-runs the checker to ground
// a mechanical claim, and emits the mapping and findings. Four leaves room
// for a duplicate call and a re-check.
const decompositionAuditMaxIterations = 4

// planAuditFindingsRe extracts one finding line from the plan auditor's
// output: `- UNMAPPED clause: ...` / `- suspect: ...`. `pass` items and the
// Audit Result line are shaped out of the pattern.
var planAuditFindingsRe = regexp.MustCompile(`(?m)^\s*-\s+(UNMAPPED clause:.*|INVENTED constraint:.*|suspect: .*)$`)

// planAuditModelFor is the model the independent plan auditor runs on: its
// own instance when the caller built one (pinned sampling, separate failover
// state), else the producer's. Named rule, same shape as auditModelFor.
func planAuditModelFor(in Input) *models.EinoChatModel {
	if in.PlanAuditModel != nil {
		return in.PlanAuditModel
	}
	return in.Model
}

// runPlanAuditor audits the plan with the independent plan auditor and
// returns its findings - empty when the audit passes. An audit outage must
// not cost the run its plan: on error or unreadable output the fallback is
// the plan as written (nil findings), the same contract as every other
// audit-failure path in the pipeline.
func runPlanAuditor(ctx context.Context, in Input, plan, question string) []string {
	agent, err := NewPlanAuditorAgent(ctx, planAuditModelFor(in), in.TenantID, in.DatasetIDs, question, plan, in.ToolCallDurations)
	if err != nil {
		common.WarnCtx(ctx, "agentic_rag: plan auditor unavailable", zap.Error(err))
		return nil
	}
	// A private conversation per audit: the auditor sees the two pinned
	// artifacts and nothing of the planner's session.
	store := session.NewInMemoryStore[adk.Message](nil)
	conv := newConversation("plan-audit", store)
	iter := conv.runner(ctx, agent, false).Run(ctx, []adk.Message{schema.UserMessage("Audit the plan pinned in your instructions.")})
	final, _, err := consumeAgentEvents(ctx, iter, func(string, string) {}, in.ToolCallCounts, nil, nil)
	if err != nil {
		conv.discardFailedTurn(ctx, conv.head(ctx))
		common.WarnCtx(ctx, "agentic_rag: plan auditor run failed", zap.Error(err))
		return nil
	}
	var findings []string
	for _, m := range planAuditFindingsRe.FindAllStringSubmatch(final, -1) {
		findings = append(findings, m[1])
	}
	return findings
}

// runConversationalReply answers a purely conversational message WITHOUT the
// explorer-auditor pipeline: no ReAct loop, no retrieval tools, no delivery
// gate - the decomposition stage has already ruled that nothing here needs
// facts. One direct generation over the caller's own history produces the
// plain-prose reply; the same emit convention as Run ships it.
func runConversationalReply(ctx context.Context, in Input) (string, error) {
	instruction := "You are RAGFlow. The last message is purely conversational - a greeting, thanks or farewell, nothing that needs facts or retrieval. Reply in plain prose, briefly and naturally. Do not use tools, and do not produce a Candidate Matrix, evidence lines or a Final Answer label."
	msgs := make([]*schema.Message, 0, len(in.Messages)+1)
	msgs = append(msgs, schema.SystemMessage(instruction))
	msgs = append(msgs, in.Messages...)
	res, err := in.Model.Generate(ctx, msgs)
	if err != nil {
		return "", fmt.Errorf("conversational reply: %w", err)
	}
	reply := ""
	if res != nil {
		reply = res.Content
	}
	emit(ctx, in.OnDelta, reply, "")
	return reply, nil
}

// extractPlan returns the plan portion of the agent's final message: everything
// from the first `### Sub-question` heading down, with a trailing code fence
// trimmed. Empty when the message carries no plan at all.
func extractPlan(final string) string {
	i := strings.Index(final, "### Sub-question")
	if i < 0 {
		return ""
	}
	// A `## Resolved question` line written before the first block is part of
	// the plan: it is what every downstream consumer reads as the question
	// (see extractResolvedQuestion). Prose without the header stays excluded.
	if j := strings.LastIndex(final[:i], resolvedQuestionHeader); j >= 0 {
		i = j
	}
	plan := strings.TrimSpace(final[i:])
	plan = strings.TrimSuffix(plan, "```")
	return strings.TrimSpace(plan)
}
