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
	"strconv"
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

// decompositionReviewRounds caps the self-review loop that follows a
// mechanically clean plan: each round must output one verdict line per
// checklist item (parsed here - a review without them is sent back), a FAIL
// buys a rewrite re-checked by checkDecomposition, and the last mechanically
// valid plan is the fallback when the budget is spent.
const decompositionReviewRounds = 3

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
	agent, err := NewQuestionDecompositionAgent(ctx, in.Model, in.TenantID, in.DatasetIDs, question, in.ToolCallDurations)
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
	for round := 0; round <= decompositionMaxRepairRounds; round++ {
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
			common.InfoCtx(ctx, "agentic_rag: question-decomposition verified",
				zap.Int("rounds", round+1), zap.Int("plan_bytes", len(plan)))
			return reviewDecompositionPlan(ctx, in, conv, agent, plan), false
		}
		common.WarnCtx(ctx, "agentic_rag: question-decomposition plan failed the mechanical check",
			zap.Int("round", round+1), zap.Strings("findings", findings))
		directive = "check_decomposition reports:\n- " + strings.Join(findings, "\n- ") +
			"\n\nFix every finding and return ONLY the corrected plan."
	}
	return plan, false
}

// reviewDecompositionPlan runs the self-review loop over a plan that already
// passes the mechanical check, and returns the plan to pin.
//
// conv and agent MUST be the same conversation and agent the writing and
// repair rounds ran: the session history lives in (store, id), not in the
// Runner, and threading both through here is what lets the review re-read the
// plan it just wrote, the checker findings it already saw, and the turns that
// produced them - the same continuity the explorer's repair turns and the
// auditor's cross-pass session rely on. A fresh agent or store would blind
// the review to its own history.
//
// The mechanical check holds no copy of the question, so the defects it cannot
// see are exactly the ones no later stage can see either: #28 shipped a plan
// that had dropped one of the question's discriminating clauses (#27's plan
// carried it), and the explorer, the auditor and the checker all worked from
// the plan as written. The planner itself is the only component that still has
// the question in hand, so the review is a pass over its own plan against the
// principles in its instructions (see "The review round" in the
// question-decomposition template).
//
// A review is only as good as it is auditable: each round must output ONE
// verdict line per checklist item (`1. PASS - ...` / `1. FAIL - ...`), and the
// verdict block is parsed HERE - a round with missing verdicts is sent back
// (a rubber-stamp review is indistinguishable from no review), a round with
// FAILs buys a rewrite, and every rewritten plan is re-run through the
// mechanical check before it can replace the current one. The loop runs until
// every item passes or the budget is spent; the last mechanically valid plan
// is the fallback on every failure path.
func reviewDecompositionPlan(ctx context.Context, in Input, conv *conversation, agent *adk.ChatModelAgent, plan string) string {
	current := plan
	directive := "## Review round\n\nRe-read the plan you just wrote against the checklist titled " +
		"\"The review round\" in your instructions. Item 1 is answered with the full clause mapping - " +
		"one line per clause of the resolved question, `clause <n>: <clause> -> c<k> (block <m>)` or " +
		"`-> UNMAPPED` - placed BEFORE the verdict lines. Output the mapping, then one verdict line " +
		"per checklist item in order - `1. PASS - <the clauses or blocks you checked>` or `1. FAIL - " +
		"<what fails, and the exact clause of the question it concerns>` - all six; THEN the plan, " +
		"rewritten where an item failed and unchanged where all held."
	for round := 1; round <= decompositionReviewRounds; round++ {
		iter := conv.runner(ctx, agent, false).Run(ctx, []adk.Message{schema.UserMessage(directive)})
		final, _, err := consumeAgentEvents(ctx, iter, func(string, string) {}, in.ToolCallCounts, nil, nil)
		if err != nil {
			conv.discardFailedTurn(ctx, conv.head(ctx))
			common.WarnCtx(ctx, "agentic_rag: question-decomposition review round failed", zap.Error(err))
			return current
		}
		verdicts, reviewed := splitReviewOutput(final)
		if reviewed == "" {
			common.WarnCtx(ctx, "agentic_rag: question-decomposition review returned no plan - keeping the current plan")
			return current
		}
		// Item 1's mapping is the clause-coverage audit trail: one line per
		// clause of the resolved question, `-> c<k>` when a block carries it,
		// `-> UNMAPPED` when none does. #32 q875 shipped a plan that dropped
		// three of the question's discriminating clauses (the 1990s setback,
		// the USA-born-or-not pair, the USA residence) while the review's
		// verdict lines all said PASS - a walk-through verdict cannot catch
		// what it never enumerates. So the mapping is enforced on both ends:
		// missing entirely, the review is not auditable; UNMAPPED lines are
		// dropped clauses confessed, and a verdict claiming PASS over them is
		// a self-contradiction. Either way the round is sent back to fix the
		// plan, until the budget runs out (then the last valid plan stays).
		verdictBlock := final[:strings.Index(final, "### Sub-question")]
		unmapped := strings.Count(verdictBlock, "-> UNMAPPED")
		mapped := len(clauseMapRe.FindAllString(verdictBlock, -1))
		if unmapped > 0 {
			common.WarnCtx(ctx, "agentic_rag: question-decomposition review mapping names UNMAPPED clauses",
				zap.Int("round", round), zap.Int("unmapped_clauses", unmapped))
			directive = "Your clause mapping carries " + strconv.Itoa(unmapped) +
				" `-> UNMAPPED` line(s) - clauses of the resolved question that no block carries. " +
				"Put each one into the block whose variable it discriminates, or into a " +
				"`filter`/`verify` block of its own, re-verify with `check_decomposition`, then " +
				"output the updated mapping, the six verdict lines and the corrected plan."
			current = reviewed
			continue
		}
		if mapped == 0 {
			common.WarnCtx(ctx, "agentic_rag: question-decomposition review carried no clause mapping",
				zap.Int("round", round))
			directive = "The review is not auditable without the item-1 clause mapping: one line per " +
				"clause of the resolved question, `clause <n>: <clause> -> c<k> (block <m>)` or " +
				"`-> UNMAPPED`, placed before the verdict lines. Output the mapping, the six " +
				"verdict lines, then the plan."
			current = reviewed
			continue
		}
		if findings := checkDecomposition(reviewed); len(findings) > 0 {
			common.WarnCtx(ctx, "agentic_rag: question-decomposition review broke the mechanical check",
				zap.Int("round", round), zap.Strings("findings", findings))
			directive = "check_decomposition reports:\n- " + strings.Join(findings, "\n- ") +
				"\n\nFix every finding, then output the six verdict lines and the corrected plan."
			continue
		}
		missing, fails := auditReviewVerdicts(verdicts)
		if len(missing) > 0 {
			common.WarnCtx(ctx, "agentic_rag: question-decomposition review verdicts incomplete",
				zap.Int("round", round), zap.Int("found", len(verdicts)), zap.Strings("missing", missing))
			directive = "The review is not auditable without one verdict line per checklist item. Output " +
				"all six verdict lines - `N. PASS - <what you checked>` or `N. FAIL - <what fails>` - " +
				"then the plan."
			current = reviewed
			continue
		}
		if len(fails) > 0 {
			common.WarnCtx(ctx, "agentic_rag: question-decomposition review reported failures",
				zap.Int("round", round), zap.Strings("fails", fails))
			directive = "Your review reported failures:\n- " + strings.Join(fails, "\n- ") +
				"\n\nRewrite the plan so every item passes, re-verify it with `check_decomposition`, " +
				"then output the six verdict lines and the corrected plan."
			current = reviewed
			continue
		}
		common.InfoCtx(ctx, "agentic_rag: question-decomposition review accepted",
			zap.Int("round", round), zap.Int("verdicts", len(verdicts)),
			zap.Bool("revised", reviewed != current), zap.Int("plan_bytes", len(reviewed)))
		return reviewed
	}
	common.WarnCtx(ctx, "agentic_rag: question-decomposition review budget exhausted - keeping the last mechanically valid plan")
	return current
}

// reviewVerdictRe matches one review verdict line: `N. PASS - ...` or
// `N. FAIL - ...` for checklist items 1-6.
var reviewVerdictRe = regexp.MustCompile(`(?m)^\s*([1-6])\.\s*(PASS|FAIL)\b`)

// clauseMapRe matches one item-1 clause-mapping line: `clause <n>: <the
// clause verbatim> -> c<k> (block <m>)` (or `-> UNMAPPED`). The mapping is
// the clause-coverage audit trail the review round must open with.
var clauseMapRe = regexp.MustCompile(`(?im)^\s*\**clause\s*\d+\**\s*:`)

// splitReviewOutput separates the review's verdict block (everything before
// the first `### Sub-question` heading) from the plan, and returns the
// verdict lines that name a checklist item.
func splitReviewOutput(final string) ([]string, string) {
	i := strings.Index(final, "### Sub-question")
	if i < 0 {
		return nil, ""
	}
	var verdicts []string
	for _, m := range reviewVerdictRe.FindAllStringSubmatch(final[:i], -1) {
		verdicts = append(verdicts, m[1]+" "+m[2])
	}
	return verdicts, extractPlan(final)
}

// auditReviewVerdicts checks the verdict block for completeness (all six
// checklist items decided) and for open failures. It returns the missing item
// numbers and the failing item numbers.
func auditReviewVerdicts(verdicts []string) (missing, fails []string) {
	decided := map[string]string{}
	for _, v := range verdicts {
		parts := strings.SplitN(v, " ", 2)
		decided[parts[0]] = parts[1]
	}
	for n := 1; n <= 6; n++ {
		item := strconv.Itoa(n)
		v, ok := decided[item]
		if !ok {
			missing = append(missing, "item "+item)
		} else if v == "FAIL" {
			fails = append(fails, "item "+item)
		}
	}
	return missing, fails
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
