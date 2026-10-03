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

package service

import (
	"context"
	"fmt"

	"go.uber.org/zap"

	"ragflow/internal/common"
	"ragflow/internal/entity"
	modelModule "ragflow/internal/entity/models"
)

// defaultAgenticTemplateID is the agent a reasoning-level-selected turn runs.
// It must name a template in conf/agentic_rag.yaml; resolveTemplateFor fails
// the turn loudly if it does not, which is the intended behaviour for an
// operator who removed or renamed the template.
const defaultAgenticTemplateID = "smart-reasoning"

// agenticModelChain resolves only the model explicitly selected by the dialog
// (or the tenant default when no model is selected), plus any failover members
// the dialog configures. Agentic RAG must not silently broaden the dialog's
// model choice to every chat model owned by the tenant: a failover chain is
// exactly the list the dialog's author chose, and nothing else.
func (s *ChatPipelineService) agenticModelChain(ctx context.Context, chat *entity.Chat) ([]*modelModule.ChatModel, error) {
	primary, err := s.agenticPrimaryModel(ctx, chat)
	if err != nil {
		return nil, err
	}

	chain := []*modelModule.ChatModel{primary}
	// Failover members are best-effort per entry: a member that no longer
	// resolves (deleted, deactivated, or re-typed) must not take the whole turn
	// down when a working primary exists. Skipping it keeps the surviving
	// members usable instead of collapsing to a hard error.
	for _, llmID := range agenticFailoverModelIDs(chat) {
		if llmID == "" || llmID == chat.LLMID {
			continue
		}
		target, resolveErr := s.ModelProviderSvc.modelSolver().
			ResolveModelConfig(ctx, chat.TenantID, entity.ModelTypeChat, llmID)
		if resolveErr != nil || target == nil {
			common.WarnCtx(ctx, "agentic_rag: skipping unresolvable failover model",
				zap.String("llm_id", llmID), zap.Error(resolveErr))
			continue
		}
		chain = append(chain, modelModule.NewChatModel(target.Driver, &target.ModelName, target.APIConfig))
	}
	common.InfoCtx(ctx, "agentic model chain resolved",
		zap.Int("chain", len(chain)), zap.String("primary", chat.LLMID))
	return chain, nil
}

// agenticPrimaryModel resolves the dialog's own model, or the tenant default
// when it selected none.
func (s *ChatPipelineService) agenticPrimaryModel(ctx context.Context, chat *entity.Chat) (*modelModule.ChatModel, error) {
	var (
		target *ModelTarget
		err    error
	)
	if chat.LLMID == "" {
		target, err = s.ModelProviderSvc.modelSolver().ResolveDefaultModelConfig(ctx, chat.TenantID, entity.ModelTypeChat)
	} else {
		target, err = s.ModelProviderSvc.modelSolver().ResolveModelConfig(ctx, chat.TenantID, entity.ModelTypeChat, chat.LLMID)
	}
	if err != nil || target == nil {
		if err == nil {
			err = fmt.Errorf("no chat model resolved for tenant %s", chat.TenantID)
		}
		return nil, fmt.Errorf("resolve chat model: %w", err)
	}
	return modelModule.NewChatModel(target.Driver, &target.ModelName, target.APIConfig), nil
}

// agenticFailoverModelIDs reads the dialog's ordered failover list from
// llm_setting. Order is the author's priority: EinoChatModel walks the chain in
// order and the sticky cursor keeps a healthy member in front, so a member that
// failed once is not re-probed on every call of a long turn.
//
// The list is per conversation and lives in llm_setting rather than a tenant
// group table, so it needs no cross-entity join and cannot outlive its dialog.
func agenticFailoverModelIDs(chat *entity.Chat) []string {
	if chat == nil || chat.LLMSetting == nil {
		return nil
	}
	raw, ok := chat.LLMSetting["failover_llm_ids"]
	if !ok {
		return nil
	}
	items, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		if id, ok := item.(string); ok && id != "" {
			out = append(out, id)
		}
	}
	return out
}
