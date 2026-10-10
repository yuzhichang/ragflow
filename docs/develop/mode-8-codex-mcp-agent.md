# mode 8 开发计划：Codex Agent（RAGFlow 作为前端与共享 Codex 的中介，经 MCP 访问 agent_rag 工具）

状态：草案（待评审）
作者：zhichyu
日期：2026-10-09
关联代码：`internal/service/chat_pipeline.go`、`internal/agentic_rag/`、`internal/mcp/`
外部依赖：[`codex-app-server-go`](https://github.com/yuzhichang/codex-app-server-go/tree/feat/schema-alignment-and-coverage)（本地 checkout 于 `~/github.com/Zealbase/codex-app-server-go`）

---

## 1. 目标

在现有 session mode 0–6 之外新增 **mode 8**。该模式在对话阶段：

1. **RAGFlow Go 后端作为中介**：连接一个**共享的、监听 WebSocket 的 Codex app-server**（下称“共享 Codex”），代表前端发起并驱动一次 Codex turn。
2. **Codex 通过 MCP 访问 agent_rag 检索工具**（`grep_chunks` / `list_chunks` / `search_bm25_chunks` 等），而不是由 RAGFlow 自己跑 ReAct 循环。
3. 对前端而言，mode 8 与 mode 5/6 完全同构：一次对话请求进来，`AsyncChatResult` 增量流出去，最终带引用与引用计数。

一句话：**mode 5/6 是“RAGFlow 用自己的模型 + 自己的工具循环”；mode 8 是“Codex 当大脑，RAGFlow 把 agent_rag 工具经 MCP 借给它”。**

### 非目标（本期不做）

- 不改造 Codex app-server 本体，不改其协议；只作为 WS 客户端接入。
- 不把 mode 8 暴露给 OpenAI 兼容入口（`internal/service/openai_chat.go`）——与 mode 5/6 一致，`reasoning` 不从这里转发。
- 不实现 Codex 端的文件系统/命令执行审批后端（审批姿态见 §5.5：**accept MCP 检索工具、decline 命令/文件**）。
- 不删除 mode 5/6；三者并存，由 level 选择。

---

## 2. 现状：session mode 0–6

“session mode”是聊天请求里的 **数值 `reasoning` 级别**，由 `resolveReasoningLevel`（`internal/service/chat_pipeline.go:6136`）从请求 kwargs 优先、dialog `prompt_config` 次之解析出来。分发只有一个入口：`selectAgenticEngine(level, hasKBs)`（`chat_pipeline.go:6115`），在 `AsyncChat` 早期（`chat_pipeline.go:351`）调用。

| Level | 含义 | 引擎 | 主要代码 |
|---|---|---|---|
| 0 | 关闭 | 普通 RAG / solo | pipeline |
| 1–4 | low/medium/high/ultra | harness graph | `internal/rag/agentic-rag/`，`harnessModeForLevel` |
| 5 | Agentic（自探索语料库） | eino ReAct explorer + delivery gate | `internal/agentic_rag/agent.go` `Run` |
| 6 | Iterative（每轮重写摘要） | 角色解耦、摘要驱动循环 | `internal/agentic_rag/iterative.go` `RunIterative` |

关键约束（新增 level 必须遵守）：

- 常量与分发：`reasoningLevelAgentic=5`、`reasoningLevelIterative=6`（`chat_pipeline.go:6073/6084`）、`agenticEngine` 枚举（`chat_pipeline.go:6088`）、`selectAgenticEngine`（`chat_pipeline.go:6115`）。
- `harnessMaxLevel=4` 同时约束 `harnessModeForLevel` 与 `reasoningNeedsAgenticGraph`（`chat_pipeline.go:2341`）。**任何新 level 不得落到 harness 的 `ultra` 分支。**
- 无 KB 对 5/6 的处理：`selectAgenticEngine` 在 `!hasKBs` 时返回 `""`，因为 5/6 对 `kb_id` 作用域做引用解析，空作用域会让引用从任意 KB 解析出来。
- **mode 8 不受“无 KB 则无引擎”约束**：Codex 是通用助手，无 KB 也能对话/推理，仍可访问。此时 MCP 检索类工具（`grep_chunks` 等）在**没有 `kb_id` 作用域的调用上返回错误**，而不是回退到空/默认作用域去检索——由 Codex 读取工具错误并据此应对。因此 `selectAgenticEngine` 必须在 `!hasKBs` 早退**之前**处理 mode 8（见 §4.1）。
  - 相应地在 `AsyncChat` 里，dispatch（`chat_pipeline.go:351`）位于 solo 快路径（`chat_pipeline.go:358` 的 `!hasKBs && !useWebSearch`）之前，所以 level==7 且无 KB 时仍会进 mode 8 引擎，不会被短路成普通 LLM 回答。
- 前端 `web/src/components/message-input/next.tsx::normalizeThinkingLevel`（第 59 行，正则 `/^[0-6]$/`）与 `thinkingOptions`（第 169 行）是 level 变得用户可见的唯一位置；漏改会让存储的级别静默降级为 1。

### 2.1 agent_rag 工具现状（MCP bridge 要复用的东西）

模板工具注册表在 `internal/agentic_rag/config.go:86` `toolRegistry()`：

| 工具 | 构造 | 依赖 |
|---|---|---|
| `search_chunks` | `NewSearchChunksTool(tenantID, datasetIDs)` | `runtime.GetRetrievalService()` |
| `search_bm25_chunks` | `NewSearchBm25ChunksTool(...)` | `runtime.GetBm25Service()` |
| `search_semantic_chunks` | `NewSearchSemanticChunksTool(...)` | `runtime.GetRetrievalService()`（VectorOnly） |
| `grep_chunks` | `NewGrepChunksTool(...)` | `runtime.GetGrepService()` |
| `list_chunks` | `NewListChunksTool(...)` | 读 chunk（`dao.DB`） |
| `think` / `todo_write` / `run_javascript` / `check_decomposition` | 无作用域 | 纯计算 |

> 上表是 agent_rag 现有工具全集。**mode 8 的 MCP bridge 只暴露其子集**：`search_chunks`、`think`、`todo_write`、`run_javascript`、`check_decomposition` 均**不暴露**（见 D7/D9）。

- 检索实现的核心是 `runLocateSearch`（`internal/agentic_rag/locate_search.go:68`），tenant/dataset 作用域在构造时注入，**运行时从 `ctx` 取 ledger**（`internal/agentic_rag/agent.go` 里 `withServedLedger`）。
- `tenantID` 必须是 **chat 的 owning tenant**（`chat.TenantID`），不是请求用户 ID；索引名由 tenant 构造，用错会稳定返回空结果。

---

## 3. mode 8 目标架构

```
              ┌────────────────────────────────────────────────────────────┐
 前端/BFF ──▶ │  RAGFlow Go 后端 (mode 8 engine)                            │
 (AsyncChat)  │                                                            │
              │  1. 解析 level==7, 取 chat.TenantID / KBIDs / session id        │
              │  2. 铸造 thread 级 MCP 票据(token -> tenant/KB scope)          │
              │  3. codex SDK 客户端(WS) ──▶ 共享 Codex app-server           │
              │       StartThread(建)/ResumeThread(复用) + RunStreamed       │
              │       线程 config override: mcp_servers.ragflow = {         │
              │         url: http://<ragflow>/mcp/codex/<token> }           │
              │  4. 消费 Codex 事件(+恢复事件) -> AsyncChatResult 增量        │
              │  5. 收尾:引用/工具计数 -> 最终 AsyncChatResult               │
              └───────────────┬────────────────────────────┬───────────────┘
                              │ WS (JSON-RPC)              │ 事件
                              ▼                            │
              ┌───────────────────────────┐                │
              │ 共享 Codex app-server      │                │
              │  (WebSocket, 多租户共享)   │                │
              │  ── 调 MCP tools ──────────┼────────────────┘
              └───────────┬───────────────┘
                          │ MCP (streamable HTTP)
                          ▼
              ┌───────────────────────────────────────────┐
              │ RAGFlow MCP bridge (internal/mcp 扩展)     │
              │  grep / bm25 / semantic / list_chunks      │
              │  -> 检索实现 / grep / read chunk           │
              │  scope = 票据解析出的 tenant + datasets    │
              └───────────────────────────────────────────┘
```

### 三方角色

- **共享 Codex app-server**：长期运行、监听 WebSocket、多租户共享。它自带推理循环，是真正的“大脑”。
- **RAGFlow Go 后端**：既是 Codex 的 WS 客户端（驱动 thread/turn、消费事件），又是检索工具的 MCP 服务端（把 agent_rag 工具按每次会话的作用域暴露给 Codex）。
- **前端**：不变，仍是 `AsyncChat` 的消费方。

### 关键设计决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | MCP 传输选 **streamable HTTP**（不是 stdio） | 共享 Codex 可能远程；stdio MCP server 必须与 Codex 同机同进程环境，无法承载 per-thread 的 tenant/KB 作用域。HTTP 端点可由 RAGFlow 直接暴露，作用域随票据走。 |
| D2 | 作用域用 **thread 级票据** 传递 | Codex 共享，多个租户/会话并发，MCP server 不能持全局作用域。票据 -> `{tenantID, datasetIDs, chatID}`，**随 thread 生命周期存活**；session 删除时失效。**票据可在每次 resume 时轮换**（D3 修好后 config override 会随 `thread/resume` 重发）。 |
| D3 | 通过 **thread config override** 注入 MCP server（**已实测确认**），**建/恢复都注入**；但 **MCP 绑定只在建 thread 时生效** | SDK 的 `WithThreadConfigOverride`（`thread.go:86`）经 `thread/{start,resume}.config` 深合并。实测（本地 codex 0.162.0）：`thread/start` 的 `mcp_servers.<name>` overlay 生效（codex 会对该 URL 发起 MCP handshake，`mcpServerStatus/list` 可见）；**`thread/resume` 的 overlay 不会重载 `mcp_servers`**（resume 时改 URL，origin 不变）。因此 **MCP URL/票据/作用域变化必须重建 thread**（见 §5.2 第 5 点）。`Client.ResumeThread` 丢弃 opts 的问题已在本 fork 修复（commit `412c6d4`），resume 会转发的字段对 model/sandbox/instructions 等有效。 |
| D4 | **创建 Codex thread 时，模型由 dialog 配置的模型决定** | 与 dialog 的 `llm_id` 一致：用 `WithThreadModelProvider(providerID)` + `WithThreadConfigOverride("model_providers.<id>", {...})` 声明并选用 dialog 解析出的 provider/model（见 SDK `examples/custom-provider`）。模型是创建 thread 时的固定选择，不随 turn 变更。 |
| D5 | 引用回填由 **thread 的 system prompt** 约定，交给 Codex 实现 | 在创建 thread 时（`WithThreadBaseInstructions` / `WithThreadDeveloperInstructions`）说明引用要求，让 Codex 在其答案中按约定标注来源；后端按约定解析回填 reference。 |
| D6 | **Codex thread 与 RAGFlow chat session 一一对应** | 一个 chat session 一个 thread，持久映射，跨 turn 复用与 resume；见 §5.2。 |
| D7 | **不向 Codex 暴露 `think` / `todo_write` / `run_javascript` 等玩具工具** | Codex 内置的 plan / reasoning 能力远强于这几个工具，重复暴露只会造成工具选择冲突。 |
| D8 | **mode 8 不因无 KB 而禁用；无 scope 时 MCP 检索工具返回错误** | Codex 是通用助手，无 KB 也能对话/推理。检索类工具在无 `kb_id` 作用域时返回 `IsError`，交由 Codex 读取错误后应对，绝不用空作用域检索。 |
| D9 | **不暴露 MCP 工具 `search_chunks`** | 用户决定（2026-10-09）。其余检索工具保留：`search_semantic_chunks` / `search_bm25_chunks` / `grep_chunks` / `list_chunks`。 |
| D10 | **对话历史重放给该 Codex thread** | RAGFlow 保存的会话历史在 thread（首建或重建）时经 `thread/inject_items`（`ThreadInjectItems`，`thread.go:500`）注入 thread 的模型可见历史，使 Codex 具备多轮上下文，不依赖 Codex 侧持久记忆。 |
| D11 | **无 session id 的入口退化为新建 thread** | `openai_chat.go` / `bot_completion.go` 未提供 session id，无法做 1:1 映射：该次请求新建一次性 thread（StartThread + 重放请求携带的历史 + 跑 turn + 弃用），不查/不写映射、不 resume。 |

---

## 4. 组件与改动清单

### 4.1 调度层（`internal/service/chat_pipeline.go`）

1. 新增常量 `reasoningLevelCodex = 7`，注释说明它像 5/6 一样命名“引擎”而非 harness 深度。
2. `agenticEngine` 增加 `engineCodex agenticEngine = "codex"`。
3. `selectAgenticEngine(level, hasKBs)`：case `reasoningLevelCodex` 必须放在 `if !hasKBs { return "" }` **之前**返回 `engineCodex`，使 mode 8 不受 KB 约束：
   ```go
   func selectAgenticEngine(level int, hasKBs bool) agenticEngine {
       if level == reasoningLevelCodex {
           return engineCodex // mode 8 无需 KB：检索工具在无 scope 时返回错误
       }
       if !hasKBs {
           return ""
       }
       switch level { ... }
   }
   ```
4. `AsyncChat` 的 dispatch（`chat_pipeline.go:351`）增加 `case engineCodex:`，调用新引擎入口。
5. `reasoningNeedsAgenticGraph` 的 `reasoningLevel > harnessMaxLevel` 已能拒绝 7，无需改动；但**必须加注释**说明 7 与 5/6 同级。

### 4.2 新引擎入口（新文件 `internal/service/codex_pipeline.go`）

仿 `iterativeSynthesis`（`internal/service/iterative_pipeline.go:55`）的通道契约：

```
func (s *ChatPipelineService) codexAgent(ctx, userID, chat, messages, stream, kwargs, useWebSearch, quote) (<-chan AsyncChatResult, error)
```

- 打开 buffered channel（容量 16，与 5/6 一致），goroutine 生产、`defer close(out)`。
- 解析 `datasetIDs`（照抄 `chat_pipeline.go:2703` 的写法）与 `tenantID = chat.TenantID`。`datasetIDs` **允许为空**（会话未绑定 KB）：此时仍进入 mode 8，只是 MCP 检索工具会返回错误（见 §4.4 第 6 点），`tenantID` 仍从 chat 取。
- 从 `common.SessionIDFromContext(ctx)` 取 session id 作为 thread 键。
- 调 `internal/codexagent` 包跑一次 turn：传入 `messages`（历史 + 当前 user 消息）用于在 thread 新建/重建时重放历史（§5.2 第 4 点），并把事件回调映射到 `AsyncChatResult`（见 §5.3）。
- 收尾时写最终结果（answer + reference + tool counts）。

### 4.3 新包 `internal/codexagent/`（Codex 引擎本体）

职责：封装“铸造票据 → 连接共享 Codex → 起/复用 thread → 跑 turn → 流事件 → 释放票据”。

文件建议：

- `client.go`：Codex 客户端生命周期的获取（进程级单例连接池，多个 turn 复用同一 WS 连接；`WithReconnectingWSTransport` + `WithAutoReconnect`）。
- `thread.go`：thread 与 RAGFlow session 的映射（key = `(tenant_id, session_id)`，session id 取自 `common.SessionIDFromContext(ctx)`；见 §5.2）。
- `history.go`：RAGFlow 消息历史 → Responses API item 的转换与 `ThreadInjectItems` 重放（D10）。
- `run.go`：一次 turn 的流式执行与事件翻译。
- `mcp_ticket.go`：票据的铸造/校验/回收（与 `internal/mcp` 协作）。

### 4.4 MCP bridge（扩展 `internal/mcp/`）

现有 `internal/mcp/server.go` 已用 `modelcontextprotocol/go-sdk v1.8.0` 暴露 `ragflow_retrieval` / `ragflow_list_datasets` / `ragflow_list_chats`，并有 `NewHandler`（`internal/mcp/http.go:52`）支持 streamable HTTP。mode 8 需要：

1. 新增一个 **按票据作用域的 server 工厂**：`NewCodexToolServer(ctx, scope Scope) *sdk.Server`，其中 `Scope{ TenantID string; DatasetIDs []string }`。
2. 注册与 `toolRegistry()`（`config.go:86`）**同名同义**的工具，参数 schema 复用 `internal/agentic_rag` 内对应工具的结构（如 `search_bm25_chunks` / `search_semantic_chunks` 的参数结构），避免两套契约漂移。
3. 工具实现**直接调用现有的检索实现**（`runLocateSearch` / grep / chunk 读取），作用域从 `scope` 注入，而不是从全局 `runtime.GetRetrievalService()` 里兜底 tenant。暴露的最小切面：
   - `search_semantic_chunks` → `locate_search.go` 的 `locateSearchSpec`（vectorOnly，tenantID=scope.TenantID, boundDatasetIDs=scope.DatasetIDs）。
   - `grep_chunks` → `internal/agentic_rag/grep_service.go` 的 `GrepService`。
   - `search_bm25_chunks` → `runtime.GetBm25Service()`。
   - `list_chunks` → `internal/agentic_rag/tool_list_chunks.go` 的读取逻辑。
   - **不暴露**：`search_chunks`（D9）、`think` / `todo_write` / `run_javascript` / `check_decomposition`（D7）。
4. 新增路由：`POST /mcp/codex/:token`（同 handler 内区分），走 `StreamableHTTPHandler`，`Stateless: true`。票据校验放在接收中间件里，参照 `internal/mcp/http.go:70` 的 `AddReceivingMiddleware` 模式。
5. 票据无效/过期 → 返回 MCP 错误，不落到任何默认 KB 作用域。
6. **无 scope 的检索调用返回错误**：当 `Scope.DatasetIDs` 为空（会话未绑定 KB）时，`search_semantic_chunks` / `search_bm25_chunks` / `grep_chunks` / `list_chunks` 一律返回 `IsError: true` 的 MCP 结果（消息说明“本次会话未绑定知识库”），**绝不**以空 `dataset_ids` 去检索。这样无 KB 的 mode 8 仍可正常对话/推理，检索请求则显式失败，Codex 读到错误后自行应对（例如直接回答或告知用户）。

### 4.5 配置（`internal/server/config/`）

新增 `CodexConfig`（参照 `mcp_config.go` 的写法与测试）：

| 键 | 说明 | 默认 |
|---|---|---|
| `codex.endpoint` | 共享 Codex 的 WS 地址 | 空 → mode 8 不可用 |
| `codex.bearer_token` | WS 鉴权 | 空 |
| `codex.mcp_public_base` | RAGFlow 对 Codex 可达的 MCP base URL | 由 `api_server` 推导 |
| `codex.approval_policy` | 见 §5.5（能力隔离），默认最严格的可用档 | `untrusted` |
| `codex.sandbox_mode` | 只读 sandbox（见 §5.5） | `read-only` |
| `codex.turn_timeout` | 单 turn 墙钟预算 | 30m（对齐 `iterativeSynthesisTimeout`） |

> 注意：**不设 `codex.model` 默认值**。模型由 dialog 配置（`chat.LLMID`）在创建 thread 时决定（D4）；配置层只提供 Codex 侧的连接、隔离与超时参数。

未配置 `codex.endpoint` 时：`selectAgenticEngine` 仍返回 `engineCodex`，但引擎入口应**降级到普通 RAG 并明确记日志**，而不是 panic（对齐“无 KB 走普通分支”的诚实降级原则）。

**模型解析（D4）**：创建 thread 时，复用现有的 dialog 模型解析（`getLLMModelConfig` / `agenticModelChain` 所依赖的模型配置读取路径）拿到 `chat.LLMID` 对应的 provider 信息，在 `StartThread` 时**同时**声明模型名与 provider：

```go
client.StartThread(ctx,
    codexgo.WithThreadModel(resolvedModelName), // ← 必填：dialog 模型解析出的实际模型名
    codexgo.WithThreadModelProvider(providerID),
    codexgo.WithThreadConfigOverride("model_providers."+providerID, map[string]any{
        "name": providerID, "base_url": baseURL,
        "wire_api": "responses", // Codex 仅支持 Responses API
        "experimental_bearer_token": apiKey,
    }),
    // D5：引用约定写进 thread 的 system/developer instructions
    codexgo.WithThreadBaseInstructions(citationContract),
    // P1#2：只读 sandbox（见 §5.5）
    codexgo.WithThreadSandbox(codexgo.SandboxReadOnly),
)
```

- ⚠️ **必须传 `WithThreadModel(resolvedModelName)`**：只传 `WithThreadModelProvider` 会继承 Codex 的默认模型名，可能调用错误模型或被自定义 provider 拒绝。`getLLMModelConfig`（`chat_pipeline.go:2292`）已返回 `(cfg, modelName, factoryName, baseURL, err)`——用其中的 **`modelName`** 作为 `WithThreadModel` 的实参，`factoryName`/`baseURL` 用于 provider override。**`chat.LLMID` 的复合 id（如 `provider/model@instance`）不等于模型名**，不能直接拿 llm_id 当模型名。
- Codex 的 wire API 只有 `responses`（见 SDK `examples/custom-provider` 注释）；dialog 模型若不支持 Responses API，需要在计划阶段确认（U2 剩余部分）。
- thread 一旦建立，模型即固定；后续 turn 走 `ResumeThread` 复用同一 provider（注意 D3：resume 不重新声明配置，依赖服务端持久化）。


### 4.6 前端（`web/`）

1. `web/src/components/message-input/next.tsx:60`：正则 `/^[0-6]$/` → `/^[0-7]$/`。
2. `thinkingOptions`（`next.tsx:169`）：新增 `{ label: t('chat.thinkingLevelCodex'), value: '7', description: t('chat.thinkingLevelCodexDescription') }`，排序置于最高档。
3. locale 增加 `thinkingLevelCodex` / `thinkingLevelCodexDescription`：**`en.ts`、`zh.ts`**（与现有 mode 5/6 的键位置一致；其余语言（ja/tr/az/uz）本就没有 5/6 的引擎档键，按同一先例回退英文，不单独注入未翻译键）。
4. 遵循 `web/AGENTS.md`（前端约定）与 `web/CLAUDE.md`（后端变体仅迁移期指引，新工作面向 Go API）。

### 4.7 引用与 quote

- quote gate 语义与 mode 5/6 保持一致：请求 kwargs 优先，`prompt_config` 只能进一步收紧（`quoteEnabled`）。
- **引用由 Codex 按 thread system prompt 的约定实现（D5）**：创建 thread 时通过 `WithThreadBaseInstructions`（或 `WithThreadDeveloperInstructions`）注入一段“引用契约”，要求 Codex 标注来源，后端据台账生成 `reference.chunks`。
  - ⚠️ **契约必须固定为「引用 chunk ID」，而不是让 Codex 自己编 `[ID:N]`**：现有 `ExtractCitedChunkIDs` 只识别 `chunk_id:\s*<id>`（`citations.go:42`），`InsertCitationMarkers` 对已含 `[ID:` 的整行会跳过（`citations.go:102`）。因此**模型的原始 `[ID:N]` 既无法被解析、也无法被可靠校正**——一旦后端过滤掉无效 chunk，模型给的编号会与最终 reference 顺序错位。
  - **正确做法**：契约要求 Codex 在其答案中按 `chunk_id: <id>` 形式（或等价的稳定、可解析 id 形式）标注来源；后端收集这些 id（复用 `ExtractCitedChunkIDs`）→ 用本 turn MCP 台账做白名单校验 → **由后端按校验后的顺序统一生成 `[ID:N]` 标记与 `reference.chunks`**（复用现有 numbering/`InsertCitationMarkers` 逻辑），模型不参与编号。
  - 契约内容需写明：id 的书写形式（逐字复制检索结果里的 chunk_id）、以及“只标注来自检索工具结果的来源”；不得要求模型输出 `[ID:N]` 数字。
  - `reference.chunks[N]` 的下标语义与 mode 5/6 一致（前端 `[ID:N]` 解析不变）。
  - Codex 标注的 id 若不在本 turn 台账内，保留文本但不生成引用（与 mode 5“不可解析则不标注”一致）。


---

## 5. 详细设计

### 5.1 MCP 工具映射表

| MCP 工具名 | 底层实现 | 作用域来源 | 备注 |
|---|---|---|---|
| `search_semantic_chunks` | 复用 `NewSearchSemanticChunksTool` 的 schema 与执行路径（`runLocateSearch` VectorOnly） | 票据 `Scope` | 无 scope 返回错误 |
| `search_bm25_chunks` | 复用 `NewSearchBm25ChunksTool` / `runtime.GetBm25Service()` | 票据 | 无 scope 返回错误 |
| `grep_chunks` | 复用 `NewGrepChunksTool` / `GrepService` | 票据 | 引擎不支持 regexp 时回 `ErrRegexpNotSupported`；无 scope 返回错误 |
| `list_chunks` | 复用 `NewListChunksTool` 的 schema 与执行路径 | 票据 | 参数为 `doc_id`(必填) + `anchor_chunk_ids`(必填, ≤20) + `number_neighbors`(可选)；**无 `offset/limit`**（见 `tool_list_chunks.go:53-57`）；无 scope 返回错误 |

> **MCP bridge 必须复用现有工具的 schema 与执行路径**（同名同义）：不要为 MCP 另写参数（例如给 `list_chunks` 加 `offset/limit`，或直接读 `dao.DB`）。构造器、`Info()` 的 JSON schema、`InvokableRun` 的语义三者都从 `internal/agentic_rag` 复用，仅把作用域从「构造时注入」改为「票据解析的 `Scope`」。

> **不暴露 `search_chunks`**（D9）：用户决定，其余检索工具保留。注意 `search_semantic_chunks` 是 `search_chunks` 的纯向量版，功能与之高度重叠。
>
> **无 scope（会话未绑定 KB）时**：以上检索类工具全部返回 `IsError: true` 的错误结果，不会用空 `dataset_ids` 发起检索（见 §4.4 第 6 点）。这是 mode 8 能在无 KB 下仍访问 Codex 的前提——对话可用，检索显式失败。
>
> **不暴露** `think` / `todo_write` / `run_javascript` / `check_decomposition`（D7）：Codex 内置的 plan/reasoning/命令能力远强于这些，重复暴露会造成工具选择冲突。MCP 只暴露“检索语料”这一类 Codex 自身没有、必须借用 RAGFlow 的能力。

> 工具返回格式：沿用现有 XML（`<search_results>` / `<tool_error>`），Codex 侧作为纯文本工具输出消费即可，无需 Codex 理解 XML 语义。

### 5.2 Thread 与 RAGFlow 会话的映射（一一对应）

**目标：多轮复用同一 Codex thread。** 但 thread 只承载「模型对话状态」；**权威数据（历史、KB 作用域、权限）每轮从 RAGFlow 现算**，thread 与之漂移时重建。

**映射表** `codex_thread_map`：`(tenant_id, session_id)` 唯一索引（复用 root `AGENTS.md` 的 GORM 唯一索引约定），存 `codex_thread_id`、`scope_fingerprint`（KB 集合 + 授权摘要）、`history_fingerprint`（已注入历史的版本）、`ticket_id`、`created_at/updated_at`。

1. **会话标识 = RAGFlow session id（会话，不是 dialog/chat id）**：一个 dialog（`chat.ID`）下可有多个 session，每段多轮各自独立。`ChatCompletions` 已把 session id 放进 ctx（`chat_session.go:1262/1315` 的 `common.WithSessionID`），mode 8 引擎经 `common.SessionIDFromContext(ctx)`（`internal/common/logger.go:381`）读取。
2. **是否用持久 thread 取决于存储语义，而非仅 session id 是否存在（P1#7）**：`store_history_messages=false` 的请求必须走**一次性 thread**（`StartThread` + 跑完即弃，不读/不写映射），即使 ctx 里带着 session id——否则非存储/测试输入会留在持久 thread 污染后续正常对话。判定依据是入口的「不存储历史」语义，不是 `SessionIDFromContext != ""`。
3. **无 session id**（`openai_chat.go` / `bot_completion.go`）→ **一次性 thread（D11）**：`StartThread` + 重放本次请求携带的历史 + 跑 turn + 丢弃，不查/不写映射、不 `ResumeThread`。
4. **每轮流程**：
   - 取当前 session 的授权、KB 集合、历史；计算 `scope_fingerprint` / `history_fingerprint`。
   - 查映射：
     - 命中且两个 fingerprint 均一致 → `client.ResumeThread(ctx, codexThreadID, opts...)`。
     - 未命中，或 **fingerprint 不一致** → **重建**：失效旧票据、`StartThread` + 重放历史 + 覆盖映射（见 5/6 点）。
5. **KB 作用域逐轮绑定（P1#8）**：作用域**不固定于首轮**。每轮从当前 dialog/session 配置解析 `DatasetIDs` 与授权，并据此生成新票据。**实测确认 `thread/resume` 的 config overlay 不会重载 `mcp_servers`（D3）**，所以 `scope_fingerprint` 变化（增删/改绑 KB、无 KB→有 KB）时**必须重建 thread**：失效旧票据、`StartThread` + 重放历史 + 覆盖映射。不能用「resume 时刷新 MCP URL」来改作用域。
6. **历史一致性（P1#6）**：Codex thread 的历史会与 RAGFlow 漂移——`DeleteSessionMessage` 删除消息（`chat_session.go:523`）、同一 session 内切换 reasoning level、其他模式写入的新回合，都不会进入已有 thread。用 `history_fingerprint`（如可见消息 id 序列的哈希）逐轮校验；不一致则**重建 thread 并重放**（若 SDK/服务端支持，也可用 `Revert` + `ThreadInjectItems` 精准同步）。**不能只在 `ResumeThread` 失败时才重放。**
7. **并发安全（P1#3）**：`SessionThread.turnMu` 是**对象级**（`thread.go:144`），而 `ResumeThread` 每次都返回新对象，所以它**不能**跨请求串行同一 session 的两个 turn。锁必须**覆盖整个 turn**：作用域/票据绑定 → resume-or-start → 历史注入 → run → 收尾 → 释放，使用 per-session 锁（进程内 singleflight + **跨进程分布式锁**，键 = `(tenant_id, session_id)`）。锁范围不能只包住「查表-或-创建」。
8. **对话历史重放（D10）**：对新 thread（首次创建、fingerprint 变化导致的重建、无 session id / 非存储的一次性 thread）注入历史，再跑当前轮：
   - 机制：`client.ThreadInjectItems(ctx, ThreadInjectItemsRequest{ThreadID, Items})`（`thread.go:500`），`Items` 是 **Responses API item 的原始 JSON**。
   - 注入内容：可用历史（有 session 时取 `chat_session` 的 history；无 session / 非存储时取本次请求的 `messages`）**去掉最后一条（当前 user 消息）**，逐条转为 Responses API item。当前 user 消息作为 `Run`/`RunInputs` 的输入，不重复注入。
   - 转换：RAGFlow message（`role`/`content`）→ Responses API item 的映射函数（如 `{"type":"message","role":"user","content":[{"type":"input_text","text":...}]}`），在 `internal/codexagent` 内实现并单测。
9. **建/恢复都注入配置**：模型（D4）、MCP override（D3）、引用契约 instructions（D5）、sandbox（P1#2）在 `StartThread` **和每次 `ResumeThread`** 都以同一 `ThreadOption` 集合声明（SDK 修复后 resume 会转发，见 D3）。这消除了「配置只在首轮生效」的隐患，也使票据/作用域可以逐轮刷新。剩余实测点（§8 U1b）：resume 的 config overlay 对 `mcp_servers` 是否**重载**，决定 KB 变化走「刷新」还是「重建」。
10. **失效回退**：`ResumeThread` 失败（共享 Codex 重启 / thread 被回收）→ `StartThread` + 覆盖映射 + 按第 8 点重放 RAGFlow 历史，记录告警。
11. **生命周期**：turn 结束后可 `thread.Close()` 释放 SDK 信号量槽（`WithMaxThreads`），下一 turn 再 `ResumeThread`；session 删除时清理映射、失效票据、删除/归档 Codex thread。

### 5.3 事件 → AsyncChatResult 映射

参考 SDK 事件（`events.go` / `events_extra.go`）：

| Codex 事件 | 映射 |
|---|---|
| `ItemAgentMessageDeltaEvent{Text}` | `AsyncChatResult{Answer: text, Final:false}` |
| `ItemReasoningTextDeltaEvent{Text}` / `ItemReasoningSummaryTextDeltaEvent{Text}` | `AsyncChatResult{Reasoning: text}`，配合 `StartToThink/EndToThink` 标记（同 `chat_pipeline.go:2813` 的 think 边界逻辑） |
| `ItemCommandExecutionOutputDeltaEvent` | 忽略或仅 debug 日志（mode 8 不面向工具执行回显） |
| `TurnCompletedEvent{TurnID, Status}` | 收尾：写最终结果 |
| `ErrorEvent` | 转 `**ERROR**: ...` 最终结果 |
| `McpToolCallProgressEvent` / `ItemStartedEvent(kind=mcpToolCall)` | 累积工具调用计数与命中文档台账（§5.4） |
| `sdk/reconnect*` / `sdk/sessionBackfilled` | **必须处理，不能忽略**（见下） |

**断线与 backfill（P1#5）**：SDK 明确「断线期间的通知不会重放」，`RunStreamed` 只转发匹配当前 turn 的事件、且**不转发 `sdk/sessionBackfilled`**（`thread.go:243-292` 的 `eventMatchesTurn` 过滤）。因此若 turn 在断线期间完成，仅靠 `RunStreamed` 会**收不到终局、丢文本、甚至挂到超时**。处理方式：

- 除 turn 的流式通道外，**同时订阅 `client.Events()`**，处理 `sdk/reconnectStarted/Succeeded/Failed`、`sdk/sessionRecovered`、`sdk/sessionBackfilled`。
- 收到 `sdk/sessionBackfilled` 时按 **`threadID`/`turnID` 合并权威状态**：用 `TurnRead` / 回填的 turns 取该 turn 的终局 `Items` 与 status，补齐缺失的最终答案，并对已流出的增量**去重**（按 turn id + item id 归并，不以文本拼接为准）。
- 终局判定以「权威 turn 状态」为准，不能以「流式通道关闭」为准（通道也可能因消费者落后而被终止）。
- 事件里的 `sdk/*` 只用于内部归并与补齐，**不原样转发给前端**（避免重复/协议泄漏）。

> **实测修正（codex 0.162.0）**：SDK 的 backfill 依赖 `thread/turns/list`，而本机 codex **不支持**该 RPC（`list_turns is not supported yet`），故 `sdk/sessionBackfilled` 实际不携带可合并的 turns —— **上述归并逻辑在本版本不可达**。据此实现**已移除** backfill 订阅/缓存（避免死代码与每 thread 的 map 泄漏），仅保留必需的兜底：**流式通道在未到终局前关闭 → 判该 turn 失败**（绝不把半截答案当终局）。若未来 codex 支持 `thread/turns/list`，再按本节恢复归并。

think 边界：mode 8 的 reasoning 与 content 可能交错，必须像 `agenticRag` 一样用 `thinking` 布尔量保证 `<think>` / `</think>` 各发一次，避免前端 `mergeAnswerChunk` 拼接错位（见 `chat_pipeline.go:2813` 注释）。

### 5.4 工具台账（用于计数与引用校验）

在 MCP bridge 侧累积每 turn：

- 每次 `search_*` / `grep_*` / `list_chunks` 的调用次数、命中 chunk/doc id、耗时。
- 最终按 mode 5/6 的字段填 `AsyncChatResult`：`ToolCallCounts`、`RetrievedDocIDs`、`ServedDocIDs`、`DeepReadChunks`、`ShallowReadChunks`。
- 台账同时作为 **引用解析的白名单**：Codex 答案里出现的 chunk id 只有落在本 turn 台账内才生成 reference（§4.7）。

台账落地在 MCP server 的 per-ticket 状态里最自然；票据是 **thread 级**（D2），故台账按 thread 累积，引擎在每 turn 收尾时读取增量并复位（或按 turn 分组记录）。

### 5.5 审批 / 安全（能力隔离）

⚠️ **`ApprovalNever()` 不是「拒绝执行」**：SDK 定义它为「**从不请求审批**」（`types.go:488`）——即 agent 会**直接执行**而不询问，是更宽松、不是更严格。mode 8 只应让 Codex 做「推理 + 调 MCP 检索」，绝不能执行命令/改文件。因此隔离必须靠**沙箱 + 禁用内置执行能力**，而不是靠审批策略：

- **只读 sandbox**：建 thread 时 `WithThreadSandbox(codexgo.SandboxReadOnly)`（或称 `read-only`）。禁止写文件、改变工作目录。
- **工作目录隔离**：`WithThreadCWD` 指向一个空的、无敏感的目录（不暴露 RAGFlow 代码/配置/密钥）。
- **审批策略**：用 `untrusted` 档，并安装 `codexagent.Dispatcher()`（`internal/codexagent/approval.go`）：**accept `mcp_tool_call` elicitation**（检索工具必需——否则连同工具一起被拒，见 §12 实测），**decline** 命令执行/文件变更/权限/其它 elicitation。注意：这只能拦「需要审批的动作」，**不能替代 sandbox**。
- **禁用/收敛内置工具**：通过 thread config override 关闭 Codex 的内置命令/文件工具（若 Codex 支持）；只保留 MCP 检索 + 推理。
- **前置验证（进入 M2 的门槛，P1#2）**：必须在真实共享 Codex 上实测确认——(a) 只读 sandbox 生效（写文件/执行命令被拒）；(b) 审批请求被 `decline` 后 turn 不挂死；(c) 无内置执行工具可用。三条全绿才允许接入，并把验证脚本/清单纳入 CI（integration 层）。
- **多租户隔离（重点）**：
  - 票据必须不可猜测（随机、足够熵），**thread 级有效期**（D2），校验失败即拒绝；session 删除/重建时立即失效。
  - MCP 工具作用域完全由票据决定，**绝不接受工具参数里的 `dataset_ids` 越权**（工具参数里的 `dataset_ids` 只能落在票据 `DatasetIDs` 子集内，参照 `resolveDatasetScope` 的“bound 优先、请求须在内”语义）。
  - **票据 `DatasetIDs` 为空时，检索类工具返回错误**（§4.4 第 6 点），而不是用空作用域检索——这既保证无 KB 对话可用，也杜绝“空 scope 泄漏到任意 KB”。
  - tenantID 一律取 `chat.TenantID`。
- 共享 Codex 若还开放 fs/命令能力，需与 MCP 检索能力解耦（靠上面的 sandbox + 审批 + 内置工具收敛约束，并以前置验证为准）。

### 5.6 错误、超时、降级

- 单 turn 墙钟预算 `codex.turn_timeout`（默认 30m），与 mode 5/6 对齐；超时 → 中断并写终局错误结果。
- 未配置 `codex.endpoint` → 降级普通 RAG + 明确告警。
- WS 断开 → 依赖 `WithAutoReconnect`；恢复后按 §5.3 用客户端恢复事件/backfill 合并权威状态、补齐终局，`sdk/*` 不原样转发前端。
- MCP 工具全失败 → 以 `<tool_error>` 形态返回给 Codex（Codex 可见），同时引擎记 `ToolCallErrors`。

---

## 6. 依赖与版本固定

- `github.com/zealbase/codex-app-server-go` 要求 **Go ≥ 1.25**；RAGFlow 当前 `go 1.27`，兼容。
- 本地 checkout 的 `go.mod` 声明 module 为 `github.com/zealbase/codex-app-server-go`，而 GitHub fork 在 `github.com/yuzhichang/codex-app-server-go`。**注意 module path 与仓库 URL 不一致**：
  - 方案 A：直接 `require github.com/zealbase/codex-app-server-go`（若该路径可拉取）。
  - 方案 B：`require github.com/zealbase/codex-app-server-go` + `replace ... => github.com/yuzhichang/codex-app-server-go <commit/branch>`，锁定 `feat/schema-alignment-and-coverage` 的具体 commit。
  - 建议 **方案 B**，把 feature 分支固定到 commit，避免跟踪浮动分支。
- `modelcontextprotocol/go-sdk v1.8.0` 已在依赖中（`go.mod:55`），MCP bridge 无需新增依赖。
- ⚠️ **已知 SDK 缺陷：`RunStreamed` 事件订阅竞态（P1#9）——已修复（M0-a）**。原 `RunStreamed` 先调 `TurnStart`、返回后才订阅（`thread.go:243-256`），快速服务端会丢事件/挂超时。**已在本 fork 修复并提交 `412c6d4`**：改为「先 `sub := client.Events()`，再 `TurnStart`」（与 `RunInputs` 对齐），并加确定性回归测试 `thread_runstream_race_test.go::TestRunStreamedSubscribesBeforeTurnStart`（在 `turn/start` 返回前注入事件；已验证未修复时该测试失败/超时）。
- ✅ **同一提交还修复了 `Client.ResumeThread` 丢弃 opts 的问题**：现在把 `ThreadOption` 集映射到 `ThreadResumeParams`（该结构本就支持完整 override），每轮 resume 都会重新声明 model/provider/config(MCP)/sandbox/instructions。回归测试：`thread_config_test.go::TestResumeThreadForwardsOverrides`。
- 依赖锁定：`replace github.com/zealbase/codex-app-server-go => github.com/yuzhichang/codex-app-server-go 412c6d4`（M0-e 落 `go.mod`）。
- ⚠️ **同步 `SessionThread.Run` 与本机 codex 不兼容（实测 0.162.0）**：其收尾走 `thread/read`（`IncludeTurns`），而该 codex 返回 `[-32601] list_turns is not supported yet`。**M2 必须用 `RunStreamed`**（它基于事件订阅，实测可用：能收到 `turn/completed`），不要用 `Run`/`WaitForTurn`。turn 的最终文本从流式事件（`ItemAgentMessageDeltaEvent` + `ItemCompletedEvent`/`TurnCompletedEvent`）组装。

---

## 7. 测试计划

按 `AGENTS.md` 的 Go 测试分层（build tag）：

- **Unit（无 tag，`bash build.sh --test`）**
  - `selectAgenticEngine`：新增 `{7, true} -> engineCodex`、**`{7, false} -> engineCodex`**（mode 8 不受 KB 约束；注意 5/6 的 `{5,false}/{6,false} -> ""` 行为不变）。扩展 `chat_pipeline_test.go:2730` 的表。
  - `resolveReasoningLevel` / `harnessMaxLevel` 边界：mode 8 不进 harness（扩展 `chat_pipeline_test.go:2664` 系列）。
  - MCP bridge 工具：用 in-memory/httptest 桩验证“票据作用域 → 检索请求”的映射，以及越权 `dataset_ids` 被裁剪；**空 `DatasetIDs` 时检索类工具返回 `IsError` 而非空作用域检索**。
  - 事件 → `AsyncChatResult` 翻译：用假的 Codex 事件序列驱动，断言 think 边界、增量和终局字段。
  - `codexagent` 票据铸造/校验/回收的单测。
  - **thread 映射**：`(tenant_id, session_id)` 命中→`ResumeThread`、未命中→`StartThread`+落库；**整个 turn 期间同一 session 的并发请求被串行**（P1#3）；**无 session id 或 `store_history_messages=false` 时走一次性 thread（不读/不写映射）**（D11/P1#7）。
  - **历史重放（D10）**：RAGFlow messages → Responses API items 的转换；首建时注入“历史去掉最后一条 user 消息”，当前 user 消息作为 turn input 不重复注入。
  - **fingerprint 漂移**：`scope_fingerprint` 变化（加/删/改绑 KB、无→有 KB）→ 重建 thread；`history_fingerprint` 变化（删除消息 `DeleteSessionMessage`、同 session 切换 reasoning level、他模式写入）→ 重建或同步（P1#6/P1#8）。
  - **模型选择**：断言 `StartThread` 带上 `WithThreadModel(resolvedModelName)`（`getLLMModelConfig` 的 `modelName`），而非仅 provider（P1#4）。
  - **backfill 归并**：模拟断线期间 turn/completed，断言收到 `sdk/sessionBackfilled` 后能补齐终局且与已流出增量去重（P1#5）。
  - **RunStreamed 快速完成**：协议桩在 `TurnStart` 前就发 delta/completed，断言无丢失、终局可达（P1#9）。
  - **引用编号**：模型输出 `chunk_id:` 后，断言由后端统一生成 `[ID:N]` 且过滤无效 id 后编号仍与 `reference.chunks` 对齐（P2#11）。
  - **list_chunks 契约**：断言复用现有 schema（`doc_id`/`anchor_chunk_ids`/`number_neighbors`），无 `offset/limit`（P2#10）。
- **Integration（`-tags integration`）**
  - 对真实（或容器化）共享 Codex WS 端点跑一次 `StartThread` → `RunStreamed` → MCP 工具被调用并返回命中。
  - **能力隔离前置验证（P1#2）**：只读 sandbox 拒绝写/执行、审批 decline 不挂死、无内置执行工具可用——三条全绿。
  - 对真实 ES/Infinity + 真 KB 验证 MCP 工具返回非空、引用可解析。
  - 多轮：同一 session 连续两轮复用同一 thread，第二轮能看到第一轮上下文（历史重放生效）。
- **E2E（`-tags e2e`）**
  - 前端 level=7 → `AsyncChat` → Codex → MCP → 检索 → 最终答案含引用；同一 session 多轮。
- **Manual（`-tags manual`，仅本地）**
  - 长会话/断线重连/票据过期的演练。

回归：确认 mode 0–6 行为不变（现有测试全绿）。

---

## 8. 已决与未决

### 已决（2026-10-09 评审）

- **D4 模型**：创建 thread 时由 dialog 配置的模型决定——`WithThreadModel(resolvedModelName)` + `WithThreadModelProvider` + `model_providers.<id>` override；`modelName` 取自 `getLLMModelConfig`，不得用复合 llm_id。
- **D3 MCP 注入**：当前 Codex 已支持经 thread config override 注入 streamable-HTTP MCP server；**仅在建 thread 时注入**（`ResumeThread` 不转发配置，见 D3 注）。
- **D5 引用**：thread 的 system prompt 约定「引用 chunk ID」，**由后端统一生成 `[ID:N]`**（§4.7 / P2#11）。
- **D7 工具集**：不暴露 `think` / `todo_write` / `run_javascript` 等玩具工具。
- **D6 映射**：Codex thread 与 RAGFlow session 一一对应；thread 只承载模型状态，权威数据每轮现算、漂移即重建。
- **D8 无 KB 也可用**：mode 8 不受 `hasKBs` 约束，无 KB 仍访问 Codex；MCP 检索类工具在无 `kb_id` 作用域时返回错误，而非空作用域检索。
- **D9 不暴露 `search_chunks`**：MCP 工具集为 `search_semantic_chunks` / `search_bm25_chunks` / `grep_chunks` / `list_chunks`。
- **D10 对话历史重放**：thread（首建/重建/一次性）时经 `ThreadInjectItems` 把历史注入 thread。
- **D11 无 session id 退化**：`openai_chat.go` / `bot_completion.go` 无 session id → 每次调用新建一次性 thread（重放请求携带的历史）。

### 未决

- **U1（前置验证，逐项实测）**：(a) `mcp_servers.<name>` 的 streamable-HTTP 配置字段/鉴权形状；(b) **resume 的 config overlay 是否会在服务端重载 `mcp_servers` 条目**（决定 KB/票据变化走「resume 刷新」还是「重建 thread」）——**已非架构阻塞**（SDK 已修复为 resume 转发 override，见 D3/commit `412c6d4`，即使不重载也可回退重建）；(c) 只读 sandbox + decline 的能力隔离效果（P1#2）。三项状态在 §11.2 独立记录。
- **U2（剩余）**：dialog 模型需支持 Codex 的 `responses` wire API；若支持则用解析出的 `modelName`，否则需在 RAGFlow 侧做 Responses 兼容层或回退 provider。
- **U3（已定）**：引用契约固定为「引用 chunk ID，后端编号」——见 §4.7，不再是未决。
- **U4**：`codex.turn_timeout` 与 Codex 侧最大 turn 时长是否冲突；中断（`TurnInterrupt`）如何触发。

### 评审修正记录（2026-10-09，均已落文）

| # | 问题 | 处理位置 |
|---|---|---|
| P1#1 | 票据生命周期与 thread 复用冲突；`ResumeThread` 不转发配置 | D2/D3、§5.2 第 9 点、§5.4 |
| P1#2 | `ApprovalNever` 不能禁止执行 | §5.5（只读 sandbox + decline + 禁内置工具 + 前置验证）、D3 |
| P1#3 | SDK 不保证同 thread 跨请求串行 | §5.2 第 7 点（锁覆盖整个 turn + 分布式锁） |
| P1#4 | 模型示例未选 dialog 模型 | §4.5（加 `WithThreadModel`）、D4 |
| P1#5 | 忽略 backfill 会丢答案/挂超时 | §5.3（订阅恢复事件、按 turn 合并去重） |
| P1#6 | 历史删除/切换模式后仍用旧上下文 | §5.2 第 6 点（history_fingerprint 校验/重建） |
| P1#7 | 非存储请求污染持久 thread | §5.2 第 2 点（按存储语义决定一次性 thread） |
| P1#8 | 固定 KB 作用域不反映配置变化 | §5.2 第 5 点（逐轮绑定 scope + 变化重建） |
| P1#9 | SDK `RunStreamed` 订阅竞态 | §6（修复或绕开 + 快速完成桩测试） |
| P2#10 | `list_chunks` 契约写错 | §5.1（复用现有 schema，无 offset/limit） |
| P2#11 | 模型 `[ID:N]` 无法可靠校正 | §4.7（固定 chunk ID 契约，后端统一编号） |

---

## 9. 里程碑与实施顺序

**M0 是硬门禁；M0-b…M0-d 未全部实测通过前，M1 及之后仍视为阻塞，不得开工。** M0 内部按以下顺序执行：

1. **M0-a — 修复 SDK 订阅竞态（P1#9）✅ 已完成（代码+单测级）**：fork 提交 `412c6d4`——`RunStreamed` 改为先订阅再 `TurnStart`；并附带修复 `ResumeThread` 丢弃 opts。回归测试 `TestRunStreamedSubscribesBeforeTurnStart`、`TestResumeThreadForwardsOverrides`；SDK 全量单测绿。
2. **M0-b — 验证 MCP 注入（U1(a)）**：确认 `mcp_servers.<name>` streamable-HTTP 的配置字段与鉴权形状，Codex 能实际加载并调用该 server。**未实测。**
3. **M0-c — 验证 resume 配置 overlay 是否重载 MCP（U1(b)，已非架构阻塞）**：确认 `thread/resume` 携带的 config overlay 是否会重载 `mcp_servers` 条目——决定 KB/票据变化走「resume 刷新」还是「重建 thread」。两种结果都可行（SDK 已能转发 override）。**未实测。**
4. **M0-d — 验证内置工具限制与隔离（U1(c) / P1#2）**：只读 sandbox 生效、审批 decline 不挂死、内置执行工具不可用。**未实测。**
5. **M0-e — 骨架 ✅ 已完成**：新增 `CodexConfig` 与解析测试；新增 `reasoningLevelCodex` + `engineCodex` + `selectAgenticEngine`（level 7 在 `!hasKBs` 早退前处理）+ AsyncChat dispatch（`codexConfigured()` 门控 + `codexAgent` seam）；前端 level 7 可见（正则 `[0-7]`、Codex 选项、en/zh locale）。SDK 依赖的 `replace 412c6d4` 推迟到 M2（`codexagent` 真正 import SDK 时再加，否则 `go mod tidy` 会删掉未使用的 require）。

后续阶段：

6. **M1 — MCP bridge ✅ 已完成**：`internal/mcp/codex.go` 新增线程级票据注册表（mint/resolve/revoke/TTL）、`NewCodexToolServer(scope)`（暴露 `search_semantic_chunks` / `search_bm25_chunks` / `grep_chunks` / `list_chunks`，**复用 eino 工具的 Info() schema 与 `InvokableRun`**）、无 scope 返回 `IsError`、`NewCodexHandler` 与路由 `Any /mcp/codex/:token`（用票据鉴权，不经过用户鉴权）。单测覆盖票据、工具集、无 scope 错误、未知 token 401。
7. **M2 — Codex 客户端接入**：`internal/codexagent` 连接共享 Codex、按 `(tenant_id, session_id)` + fingerprint 起/复用/重建 thread（D4/D6/P1#6/P1#8）、建 thread 时注入 MCP override（D3）、重放历史（D10）、按 dialog 模型选 `WithThreadModel`（P1#4）、只读 sandbox（P1#2）、**整个 turn 持锁**（P1#3）、**订阅恢复事件合并 backfill**（P1#5）、把事件翻译成 `AsyncChatResult`；集成测试。
8. **M3 — 贯通 ✅ 已实现（编译+单测级）**：`codexAgent` 接入 `internal/codexagent.Runner`；每轮 mint 线程级票据并拼 `MCPPublicBase + /mcp/codex/<token>`；解析 dialog 模型（`resolveCodexModelTarget`）并用 **Responses 能力白名单**（`codex.responses_providers`，决策 U2=A）门控——不兼容则降级普通 RAG；reasoning 用 `<think>` 边界、answer 增量、终局带 `Reference`（`buildAgenticReference`）。`codex_pipeline.go` 接入 dispatch。
9. **M4 — 加固**：多租户隔离、超时/降级、重连、非存储/无 session 入口，文档收尾。

每一步遵循根 `AGENTS.md`：小步、单一实现路径、无兼容分支、改完跑最窄相关测试（`bash build.sh --test ./internal/...`）。

---

## 10. 需要复核的既有事实（写代码前逐条核对）

- `selectAgenticEngine` 当前签名是 `(level int, hasKBs bool)`，注意 `agent_mode` 参数已被移除（`chat_pipeline.go:6100` 注释）。
- 前端 `normalizeThinkingLevel` 正则当前为 `/^[0-6]$/`（`next.tsx:60`），注释也需同步更新（提及 codex）。
- `AsyncChatResult` 字段（`chat_pipeline.go:93`）中 mode 5/6 用到的台账字段需与 mode 8 保持一致，便于 benchmark 按字段对比。
- `internal/mcp/http.go` 的 `Stateless: true` 是否满足 Codex MCP 客户端对会话的要求（若需有状态会话，需改用 SSE/有状态 streamable）。
- **SDK 行为已核对（2026-10-09，本地 checkout）**：`ResumeThread` 丢弃配置（`client.go:489-508`）；`ApprovalNever`=从不请求审批（`types.go:488`）；`RunStreamed` 先 TurnStart 后订阅（`thread.go:243-256`）；`turnMu` 为对象级（`thread.go:144`）；`list_chunks` 参数为 `doc_id`+`anchor_chunk_ids`+`number_neighbors`（`tool_list_chunks.go:53-57`）；`citations.go` 仅解析 `chunk_id:`、跳过已含 `[ID:` 的行。

---

## 11. 状态记录（文档修订 vs 运行验证）

两类状态**分开记录**，不得混为一谈：**「文档已修订」**只表示计划书已改，**「运行验证通过」**必须附实测环境、命令与结果。写入方各自独立，一方通过不代表另一方通过。

### 11.1 文档修订状态（本文件）

| 项 | 状态 | 日期 | 说明 |
|---|---|---|---|
| P1#1–#9、P2#10–#11 修订 | ✅ 已修订 | 2026-10-09 | 见 §8「评审修正记录」 |
| M0 门禁顺序（a→e）固定 | ✅ 已修订 | 2026-10-09 | §9 |
| U1(b) 由「架构关键阻塞」降级为「刷新 vs 重建的实测点」 | ✅ 已修订 | 2026-10-09 | SDK 修复 `ResumeThread` 转发 override（commit `412c6d4`），§3 D2/D3、§5.2、§8 U1 |

### 11.2 运行验证状态（实测，逐项独立）

> 记录格式：`项 | 状态 | 环境/命令 | 结果 | 日期`。未实测前一律 `⏳ 未验证`。

| 门禁项 | 状态 | 环境 / 命令 | 结果 | 日期 |
|---|---|---|---|---|
| M0-a SDK 订阅竞态修复 + 回归测试 | ✅ 已通过（单测级） | `cd ~/github.com/Zealbase/codex-app-server-go && go test -count=1 . ./internal/...` | 全绿；未修复时 `TestRunStreamedSubscribesBeforeTurnStart` 5s 超时失败 | 2026-10-09 |
| M0-b MCP 注入字段/鉴权（U1(a)） | ✅ 已通过（实测） | 本地 codex 0.162.0 + SDK：`WithThreadConfigOverride("mcp_servers.ragflow", {url})` | config 形状 `[mcp_servers.<name>] url=...` 生效；`mcpServerStatus/list` 可见该 server 且 codex 对其发起 MCP handshake | 2026-10-09 |
| M0-c resume overlay 是否重载 MCP（U1(b)） | ✅ 已通过（实测，结论=不重载） | 同上：resume 时把 URL 从 :9 改为 :10 | resume 后 `mcpServerStatus/list` 的 origin 仍为 :9 → **resume 不重载 mcp_servers**；作用域/票据变化必须**重建 thread** | 2026-10-09 |
| M0-d 隔离（U1(c)/P1#2） | ✅ 通过（(c) 例外见备注） | `go test -tags integration -run 'TestSandboxReadOnlyBlocksWrites\|TestApprovalDeclineDoesNotHang' ./internal/codexagent/` | (a) **read-only sandbox 生效**；(b) **审批 decline 不挂死**：生产 `Dispatcher`（命令/文件 decline）+ read-only 下，写文件 turn 自行完成、不写文件；(c) 未关闭内置命令/文件工具（codex 0.162 无该配置项；靠 read-only + decline 限制） | 2026-10-09 |
| M0-e 骨架编译/单测 | ✅ 已通过 | `bash build.sh --test ./internal/server/config/... ./internal/service/...` + `npx tsc --noEmit`（仅新改动文件）+ `npx oxlint` | config/service 全绿；前端改动文件无新增类型/lint 错误 | 2026-10-09 |
| M1 MCP bridge 单测 | ✅ 已通过 | `bash build.sh --test ./internal/mcp/... ./internal/router/...` | 全绿（票据 mint/resolve/revoke/过期、工具集不含 search_chunks、无 scope 返回 IsError、未知 token 401） | 2026-10-09 |
| M1 端到端（Codex 实调 MCP） | ✅ 已通过（实测） | `go test -tags integration -run TestCodexCallsBridgeTool ./internal/mcp/` | codex 经 bridge 实调 `search_semantic_chunks`：`answer="chunk-e2e-0001"`、`ledger.calls=map[search_semantic_chunks:1]`、`ledger.chunks=map[chunk-e2e-0001]`；根因与修复见 §12 | 2026-10-09 |
| 评审修正（P1#5/#7、§5.2/§5.4/§5.5） | ✅ 已通过（单测/集成级） | `bash build.sh --test ./internal/{codexagent,mcp,service,server/config}/...` + `go test -tags integration`（M2/sandbox/E2E） | 无终局不判成功（backfill 归并在 0.162 不可达、已移除，见 §5.3）；`store_history_messages=false`→一次性 thread；空 CWD；session 删除清映射+失效票据；`AsyncChatResult` 补齐工具计数/命中文档/deep-shallow | 2026-10-09 |
| §7 缺测补齐（作用域/并发/隔离） | ✅ 已通过 | `bash build.sh --test ./internal/agentic_rag/... ./internal/codexagent/...` + 上述 M0-d 集成 | `resolveDatasetScope` 越权即拒（不静默裁剪）；`sessionLock` 键=(tenant,session) 且同会话串行；审批 decline 不挂死（生产 Dispatcher） | 2026-10-09 |
| M2 runner 端到端（真 codex + 真模型） | ✅ 已通过 | `RAGFLOW_CODEX_IT=1 OPENAI_BASE_URL/KEY/MODEL=... go test -tags integration -run TestRunnerAgainstLocalCodex ./internal/codexagent/` | turn1 `rebuilt=true content="PONG"`；turn2 `rebuilt=false`（复用同一 thread）`content="PONG2"` | 2026-10-09 |
| M3 贯通（`codexAgent` 接线） | ✅ 已通过（编译+单测级） | `bash build.sh --test ./internal/service/... ./internal/server/config/...` | 全绿；`codexAgent` 走 Runner + 票据 + Responses 白名单门控 | 2026-10-09 |
| M3 端到端（AsyncChat → Codex → MCP → 引用） | ✅ 已通过（实测，范围有限） | `go test -tags cgo,static,integration -run TestCodexAsyncChatEndToEnd ./internal/service/`（需 CGO env） | `AsyncChat(level 7)` 经 dialog 模型解析→runner→本地 codex WS→answer `"E2EOK"`；**MCP 仅 handshake，未触发工具调用；无真实 KB/ES，引用链路未端到端** | 2026-10-09 |
| P2#11 引用台账白名单 | ✅ 已通过（单测级） | `bash build.sh --test ./internal/mcp/... ./internal/service/...` | bridge 每 turn 记录 `chunk_id` 台账，`codexAgent` 以之为引用白名单（仅本 turn 服务的 chunk 生成 `[ID:N]`）；模式 5/6 不受影响 | 2026-10-09 |

**门禁规则**：只要 11.2 中 M0-b…M0-d 任一项为 `⏳ 未验证` 或 `❌ 未通过`，**M1 及之后一律阻塞**。M0-c（U1(b)）已非架构阻塞：无论 resume 是否重载 MCP，都可分别走「resume 刷新票据」或「重建 thread」，两种路径均在计划内。

> 说明：M0-b…M0-d 需连真实共享 Codex，本环境不可执行；代码可先按计划推进（M0-e/M1 不含对真实 Codex 的运行时依赖），但 M1 之后的端到端验证必须以这三项实测通过为前提。

---

## 12. MCP 工具调用审批（2026-10-09 定位并修复 ✅，本机 codex 0.162.0）

**结论：MCP 工具调用受 `mcpServer/elicitation/request` 门控；不装 handler 会被 codex 拒绝，检索路径全死。已定位并修复。**

复现方法：本地 codex（stdio），线程注入 `mcp_servers.ragflow.url` 指向本进程 MCP bridge（活的 httptest），注册假检索服务返回已知 chunk；prompt 明确要求调用 `search_semantic_chunks`。用 `WithRequestHandler` 打印原始请求后定位到：

**根因**：MCP 工具调用前，codex 下发一个 `mcpServer/elicitation/request`，其 `mode="form"`、`_meta.codex_approval_kind="mcp_tool_call"`，message 形如 `Allow the ragflow MCP server to run tool "search_semantic_chunks"?`。**SDK 默认 `DeclineElicitation()`**（无 handler）→ codex 报 “rejected by the user” → 工具从不执行。
- 之前误以为走 `item/permissions/requestApproval`（那 4 个 `ApprovalHandler` 分支**从不被触发**，已加日志证实）。
- `approval_policy` / `approvals_reviewer` / `mcp_servers.*.approval_mode` 都不是正确杠杆（`approvals_reviewer=auto_review` 反而因缺 `codex-auto-review` 模型报错 2013）。

**修复**：`internal/codexagent/approval.go` 提供 `Dispatcher()`（`*codexgo.Dispatcher`，仅设 `Elicitation`）：命中 `codex_approval_kind=="mcp_tool_call"` 的 elicitation → **accept**（`Content={}`），其余 elicitation/命令/文件/权限一律默认拒绝；`Dial` 用 `WithRequestHandler(Dispatcher())` 安装。

**实测（`TestCodexCallsBridgeTool`，已提交）**：`answer="chunk-e2e-0001"`，`ledger.calls=map[search_semantic_chunks:1]`，`ledger.chunks=map[chunk-e2e-0001]` → **codex 真正经 bridge 调用了检索工具并拿到命中**。

**对 §5.5 的修正（重要）**：隔离姿态不能是“统一 decline”。正确姿态是 **accept MCP 工具调用（检索必需）** + **decline 命令执行/文件变更/权限/其它 elicitation**，再由 read-only sandbox 兜底。旧的“统一 decline”会连同检索工具一起挡掉。

（历史阻塞记录见 git 提交 `1f2f07b43`；本节为修复后的定稿。）

---

## 13. 已接受的设计约束（评审 P1-5 / P2-13 的处置）

### P1-5 跨实例整轮锁 —— 按设计**不做**（rejected by design）

**决策**：不实现跨进程/跨实例的分布式锁。会话级串行只由**进程内** `keyedLocks`（键 `(tenant_id, session_id)`，见 `internal/codexagent/run.go`）保证。

**依据（约束前提）**：RAGFlow 保证**单点登录**语义——同一 `session_id` 不会由多个 API 进程并发服务。因此“同 session 同时创建/运行 thread”在本部署模型下不会发生；`Put` 的覆盖式 upsert 也就不会出现跨实例的“最后写入者获胜”竞态。

> ⚠️ **注意：单点登录 ≠ 进程粘性**。单点登录约束的是“同一用户同一时刻只有一个会话”，**不**保证该 session 的后续请求落在**同一个 API 进程**。若横向扩展后请求被轮询到不同进程，进程内锁即失效。

**该前提的部署要求（必须实际满足，否则本约束失效）**：
- **单副本** API 进程（扩展前先确认路由），**或**
- **为同一 session 强制粘性路由**（sticky session / 一致性哈希，键含 `session_id`）。
- 一旦放开“多进程并发服务同一 session”，**必须**改回**分布式锁**（例如按 `(tenant_id, session_id)` 的 DB 咨询锁，覆盖整个 turn），否则检索票据/thread 映射会被并发破坏。此条为**未实现的已知限制**，不是“已解决”。

### P2-13 provider 默认 endpoint —— 采用方案 (b)：由实例显式提供 `base_url`

**决策**：mode 8 **不**从 provider 工厂默认值解析 endpoint；要求 mode 8 对话所选**模型实例显式配置 `base_url`**（指向支持 **Responses API** 的端点）。

**原因**：`codexAgent` 只用 `target.APIConfig.BaseURL`（即模型实例的 `base_url`）；Codex 需要 `wire_api="responses"`，而 provider 工厂默认 URL 未必指向 Responses 端点。把“实例 `base_url` 为空 → 回退工厂默认”留给 Codex 会静默打到错误的 wire API。显式配置让“该实例是否可用于 mode 8”在配置期即可判断。

**接入前校验（已实现）**：`codexResponsesCapable` 会同时要求 provider 在白名单内**且实例 `base_url` 非空**；不满足即**降级普通 RAG 并打明确告警**（`codex (mode 8) requires the dialog model instance to set base_url ...`）。`codexAgent` 亦对空 `base_url` 直接返回明确错误，不会以空 `base_url` 声明 provider。

### P1-7 超时中断：SDK 暴露 turn ID + 已知协议限制

**修复**：`RunStreamed` 从不返回服务端分配的 turn ID，导致「远端已启动、本地尚未消费首个携带 ID 的事件就被取消」时无法定向中断。已在本 fork 新增 `SessionThread.RunStreamedTurn(ctx, input, ...) (<-chan, turnID, err)`——`turnID` 由 `TurnStart` **同步返回**（事件流与 `RunStreamed` 相同，含「先订阅后启动」）；`consume` 改用之，两个取消出口都有 ID 可中断。SDK pin 提升到 `yuzhichang/codex-app-server-go v0.0.0-20261010044256-14e2b46c590a`（commit `14e2b46`）。

**已知协议限制（如实记录）**：若 `TurnStart` **取消或报错**，**不代表远端没有 turn**——服务端可能已启动、只是启动响应未送达。此时本地**缺少 turn ID**，属于「启动结果未知，无法定向中断」。这是协议层的固有限制（无 ID 无从指定要中断的 turn），不是实现缺陷；如需彻底消除，需要 Codex 协议支持「按 thread 中断当前 turn」或客户端可查询 thread 的当前 turn ID。
