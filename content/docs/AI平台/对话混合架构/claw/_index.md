# 基于 LLM 的智能体与传统对话系统混合架构

> 面向职位：传音集团 · 资深后端开发 — 大模型算法部
> 核心职责 #2：研发基于 LLM 的智能体和传统对话系统的混合架构
> 覆盖：任务规划、Memory、工具、RAG、对话管理、NLU 等任务的后台系统

---

## 一、为什么要混合架构？

### 1.1 纯 LLM Agent 的不足

| 问题 | 场景 | 影响 |
|------|------|------|
| 幻觉 | 用户问"我的套餐还剩多少流量" | LLM 无法给出精确答案 |
| 不可控 | 多轮对话中的状态管理 | 对话跑偏、任务丢失 |
| 成本高 | 每个请求都调大模型 | 5 亿+ AI 请求的成本不可承受 |
| 延迟大 | 简单意图（天气、闹钟）也走 LLM | 用户体验劣化 |
| 合规难 | 敏感场景（支付、身份验证） | LLM 无法保证确定性输出 |

### 1.2 纯传统对话系统的不足

| 问题 | 场景 | 影响 |
|------|------|------|
| 泛化差 | 用户表达与训练集不一致 | 意图识别失败 |
| 维护成本高 | 每新增一个意图需标注 + 训练 | 迭代慢 |
| 多轮能力弱 | 复杂任务需硬编码对话树 | 开发成本高 |
| 无法处理开放域 | 闲聊、创意生成 | 体验僵硬 |

### 1.3 混合架构的设计原则

```
确定性任务 → 传统对话系统（快、准、省）
  ↑
  │  意图路由（轻量分类器）
  │
模糊/复杂任务 → LLM Agent（泛化、推理、创造）
  ↑
  │  LLM 自主判断
  │
需要专业知识 → LLM + RAG（检索增强生成）
  ↑
  │  工具触发判断
  │
需要外部操作 → LLM + 工具调用（Function Calling）
```

**核心思想**：不是"LLM vs 传统"，而是"LLM + 传统"，让系统在最合适的层级处理每个请求。

---

## 二、整体架构

### 2.1 系统全景图

```
┌──────────────────────────────────────────────────────────────────┐
│                        用户输入                                   │
│  语音/文本/图片/视频 → 多模态输入解析                              │
├──────────────────────────────────────────────────────────────────┤
│                    【输入理解层 — Input Understanding】            │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ ASR 语音识别 │  │ 视觉理解    │  │ 语种检测    │              │
│  │ (多语种)     │  │ (OCR/场景)  │  │ (140种)     │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              NLU Engine（自然语言理解）                    │    │
│  │                                                         │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │    │
│  │  │ 意图分类器    │  │ 实体抽取器   │  │ 情感/安全分析 │  │    │
│  │  │ (BERT小模型)  │  │ (NER/正则)   │  │ │            │  │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │    │
│  │         └──────────────────┼─────────────────┘          │    │
│  │                            ▼                            │    │
│  │                   统一理解结果                           │    │
│  │     { intent, entities, confidence, language, ... }     │    │
│  └────────────────────┬────────────────────────────────────┘    │
├─────────────────────────┼──────────────────────────────────────┤
│                    【路由与调度层 — Routing】                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              智能路由器（Router）                         │    │
│  │                                                         │    │
│  │  if confidence > 0.95 && 确定性意图:                    │    │
│  │      → 传统对话系统（Dialogue Manager）                  │    │
│  │  elif 需要外部知识:                                      │    │
│  │      → LLM + RAG                                       │    │
│  │  elif 需要外部操作:                                      │    │
│  │      → LLM + 工具调用                                   │    │
│  │  elif 复杂推理/多步任务:                                 │    │
│  │      → LLM Agent（Planner + Executor）                  │    │
│  │  else:                                                  │    │
│  │      → 自由对话 LLM                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────────┤
│              【执行层 — Execution（多路径并行）】                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ 传统对话系统  │  │ LLM Agent    │  │ LLM + RAG    │           │
│  │ (Dialogue Mgr)│  │ (Planner +   │  │ (检索+生成)  │           │
│  │              │  │  Tools +     │  │              │           │
│  │ · 确定性意图  │  │  Memory)     │  │ · 知识问答   │           │
│  │ · 快捷指令    │  │              │  │ · 文档问答   │           │
│  │ · 设备控制    │  │ · 任务规划   │  │ · 精准检索   │           │
│  │ · 安全拦截    │  │ · 工具调用   │  │ · 事实核查   │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         └─────────────────┼─────────────────┘                    │
│                           ▼                                       │
├──────────────────────────────────────────────────────────────────┤
│                    【输出层 — Output Generation】                  │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ 文本生成    │  │ 语音合成    │  │ 视觉输出    │              │
│  │ (模板/LLM)  │  │ (TTS多语种) │  │ (卡片/图表) │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 三、核心模块详细设计

### 3.1 NLU Engine（自然语言理解）

```
┌────────────────────────────────────────────┐
│              NLU Pipeline                   │
│                                            │
│  原始输入                                   │
│      │                                     │
│      ▼                                     │
│  ┌──────────────┐                          │
│  │ 语种检测      │  ← 140 种语言             │
│  │ (FastText)    │                          │
│  └──────┬───────┘                          │
│         │                                  │
│         ▼                                  │
│  ┌──────────────┐                          │
│  │ 意图分类      │  ← 轻量 BERT（< 50M 参数） │
│  │               │    端侧可部署             │
│  │ 层级意图体系： │                          │
│  │ L1: 领域      │  语音/翻译/视觉/设备控制   │
│  │ L2: 意图      │  查天气/设闹钟/翻译文本    │
│  │ L3: 子意图    │  中译英/英译法/法译中     │
│  └──────┬───────┘                          │
│         │                                  │
│         ▼                                  │
│  ┌──────────────┐                          │
│  │ 实体抽取      │  ← NER + 正则规则         │
│  │               │    时间/地点/语言/设备名   │
│  └──────┬───────┘                          │
│         │                                  │
│         ▼                                  │
│  ┌──────────────┐                          │
│  │ 置信度评估    │  ← 分类置信度 + 规则匹配  │
│  │               │    决定路由方向           │
│  └──────┬───────┘                          │
│         │                                  │
│         ▼                                  │
│  NLU Result:                               │
│  { intent, entities, confidence, language } │
└────────────────────────────────────────────┘
```

**技术选型**：

| 组件 | 方案 | 理由 |
|------|------|------|
| 语种检测 | FastText Language ID | 轻量、176 种语言、< 1MB 模型 |
| 意图分类 | DistilBERT / TinyBERT | 端侧可部署、推理 < 20ms |
| 实体抽取 | spaCy + 自定义规则 | 结构化实体（时间/设备名）用规则更准 |
| 置信度 | Temperature-scaled Softmax | 避免过度自信，支持阈值判断 |

### 3.2 智能路由器（Router）

```go
// 路由决策（Go 伪代码）
type Router struct {
    traditionalHandler *DialogueManager
    llmAgentHandler    *LLMAgent
    ragHandler         *RAGEngine
    freeChatHandler    *LLMChat
}

func (r *Router) Route(nluResult NLUResult, session Session) Handler {
    // 1. 安全/合规检查（最高优先级）
    if r.isSensitiveIntent(nluResult.Intent) {
        return r.traditionalHandler  // 确定性传统系统
    }
    
    // 2. 高置信度 + 确定性意图 → 传统系统
    if nluResult.Confidence > 0.95 && isDeterministic(nluResult.Intent) {
        return r.traditionalHandler
    }
    
    // 3. 知识密集型 → RAG
    if needsExternalKnowledge(nluResult.Intent) {
        return r.ragHandler
    }
    
    // 4. 需要外部操作 → LLM + 工具
    if needsToolExecution(nluResult.Intent) {
        return r.llmAgentHandler
    }
    
    // 5. 复杂多步推理 → LLM Agent Planner
    if isComplexTask(nluResult) {
        return r.llmAgentHandler
    }
    
    // 6. 自由对话 → 直连 LLM
    return r.freeChatHandler
}

// 意图类型判断
func isDeterministic(intent string) bool {
    deterministicIntents := map[string]bool{
        "device.control.set_alarm":    true,  // 设闹钟
        "device.control.set_timer":    true,  // 设计时器
        "device.control.wifi_toggle":  true,  // 开关 WiFi
        "device.control.bluetooth":    true,  // 蓝牙控制
        "query.weather":               true,  // 查天气（走 API，不走 LLM）
        "query.battery":               true,  // 查电量
        "query.signal":                true,  // 查信号
        "translate.text":              true,  // 翻译（走翻译引擎）
    }
    return deterministicIntents[intent]
}
```

### 3.3 传统对话系统（Dialogue Manager）

```
┌─────────────────────────────────────────────────┐
│           传统对话系统（Dialogue Manager）         │
│                                                 │
│  ┌───────────┐    ┌───────────┐    ┌──────────┐ │
│  │ 意图匹配   │───→│ 对话状态  │───→│ 响应生成  │ │
│  │ (规则引擎) │    │ 管理 (DST)│    │ (模板)   │ │
│  └───────────┘    └───────────┘    └──────────┘ │
│                                                 │
│  适用场景：                                      │
│  · 设备控制（闹钟/WiFi/蓝牙/音量）                 │
│  · 快捷查询（天气/电量/信号/时间）                 │
│  · 翻译请求（路由到翻译引擎）                      │
│  · 安全拦截（支付/身份验证/隐私操作）              │
│                                                 │
│  技术：                                          │
│  · 规则引擎：Drools / 自研规则 DSL                │
│  · 对话状态：有限状态机（FSM）                     │
│  · 响应生成：模板引擎（支持多语言 i18n）           │
└─────────────────────────────────────────────────┘
```

**对话状态管理（DST）**：

```go
// 对话状态机（Go 伪代码）
type DialogueState struct {
    SessionID    string
    CurrentIntent string
    Slots        map[string]string    // 槽位填充
    Context      map[string]any       // 上下文信息
    Step         int                  // 对话步骤
}

type DialogueManager struct {
    stateStore  *RedisStore          // 会话状态存储（Redis）
    ruleEngine  *RuleEngine          // 规则引擎
    template    *I18nTemplate        // 多语言模板
}

func (dm *DialogueManager) Process(nluResult NLUResult, sessionID string) Response {
    // 1. 加载或创建对话状态
    state := dm.stateStore.GetOrCreate(sessionID)
    
    // 2. 槽位填充
    state.FillSlots(nluResult.Entities)
    
    // 3. 规则匹配
    rule := dm.ruleEngine.Match(state)
    
    // 4. 状态转移
    state = rule.Apply(state)
    
    // 5. 生成响应
    response := dm.template.Render(rule.ResponseTemplate, state.Slots, nluResult.Language)
    
    // 6. 保存状态
    dm.stateStore.Save(state)
    
    return response
}
```

### 3.4 任务规划器（Planner）

```
┌─────────────────────────────────────────────────┐
│              LLM Agent — 任务规划器               │
│                                                 │
│  用户输入："帮我查下明天从深圳到内罗毕的机票，     │
│           顺便看看那边的天气怎么样"               │
│                                                 │
│  LLM Planner 输出：                              │
│  ┌─────────────────────────────────────────┐    │
│  │ Plan:                                   │    │
│  │  Step 1: 搜索机票（工具：flight_search）  │    │
│  │    输入: {origin: "深圳", dest: "内罗毕", │    │
│  │          date: "2026-08-25"}            │    │
│  │  Step 2: 查询天气（工具：weather_query）  │    │
│  │    输入: {location: "内罗毕",            │    │
│  │          date: "2026-08-25"}            │    │
│  │  Step 3: 汇总结果（工具：无，LLM 生成）    │    │
│  │    输入: {flight_results, weather_results}│    │
│  └─────────────────────────────────────────┘    │
│                                                 │
│  规划策略：                                      │
│  · ReAct（Reason + Act）                        │
│  · Plan-and-Execute                             │
│  · 支持并行执行（Step 1 和 Step 2 可并行）        │
└─────────────────────────────────────────────────┘
```

**Planner 实现**：

```go
type TaskPlanner struct {
    llm         *LLMClient          // LLM 客户端
    toolRegistry *ToolRegistry      // 工具注册表
}

func (p *TaskPlanner) Plan(userInput string, context string) *Plan {
    // 构建 Planner Prompt
    prompt := fmt.Sprintf(`
You are a task planner. Given the user request and available tools,
create a step-by-step plan.

Available tools:
%s

User request: %s

Context: %s

Return the plan as JSON with steps. Mark steps that can run in parallel.
`, p.toolRegistry.DescribeAll(), userInput, context)
    
    // LLM 生成计划
    planJSON := p.llm.Generate(prompt)
    
    // 解析为执行计划
    return ParsePlan(planJSON)
}
```

### 3.5 Memory 系统

```
┌──────────────────────────────────────────────────────────┐
│                    Memory 分层体系                        │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │
│  │ 短期记忆        │  │ 长期记忆        │  │ 工作记忆   │  │
│  │ (Session)      │  │ (User Profile) │  │ (Context)  │  │
│  │                │  │                │  │            │  │
│  │ · 当前对话历史  │  │ · 用户偏好      │  │ · 当前任务  │  │
│  │ · 槽位填充状态  │  │ · 个人信息      │  │ · 工具结果  │  │
│  │ · 近期交互      │  │ · 行为习惯      │  │ · 中间变量  │  │
│  │                │  │ · 历史意图      │  │            │  │
│  │ 存储：Redis     │  │ 存储：ES + 向量  │  │ 存储：内存  │  │
│  │ TTL：30 min    │  │ TTL：永久      │  │ TTL：单次   │  │
│  └────────────────┘  └────────────────┘  └────────────┘  │
│                                                          │
│  Memory 检索流程：                                        │
│  用户输入 → 意图识别 → Memory 查询 → 注入 Prompt          │
│                                                          │
│  例：                                                     │
│  用户："像上次那样翻译这段话"                               │
│  → 检索长期记忆：用户上次翻译是中→英                      │
│  → 注入：Translate Chinese to English                    │
│  → 执行翻译                                              │
└──────────────────────────────────────────────────────────┘
```

**Memory 实现**：

```go
type MemoryManager struct {
    shortTerm  *ShortTermMemory    // Redis 存储
    longTerm   *LongTermMemory     // ES + 向量数据库
    working    *WorkingMemory      // 内存
}

// 记忆检索与注入
func (m *MemoryManager) EnrichContext(sessionID string, userInput string) Context {
    ctx := Context{}
    
    // 1. 短期记忆（对话历史）
    ctx.RecentTurns = m.shortTerm.GetHistory(sessionID)
    
    // 2. 长期记忆（用户偏好，向量检索 Top-K）
    userProfile := m.longTerm.Retrieve(sessionID, userInput)
    ctx.Preferences = userProfile.Preferences
    ctx.Habits = userProfile.Habits
    
    // 3. 工作记忆（当前任务状态）
    ctx.CurrentTask = m.working.Get(sessionID)
    
    return ctx
}

// 记忆更新
func (m *MemoryManager) Update(sessionID string, turn Turn) {
    // 短期记忆更新
    m.shortTerm.Append(sessionID, turn)
    
    // 重要信息沉淀到长期记忆
    if m.shouldPersist(turn) {
        m.longTerm.Persist(sessionID, turn)
    }
}
```

### 3.6 工具系统（Tool System）

```
┌──────────────────────────────────────────────────────┐
│                  工具注册与调用体系                     │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │              Tool Registry                    │    │
│  │                                               │    │
│  │  内置工具：                                    │    │
│  │  ├─ weather_query（天气查询）                   │    │
│  │  ├─ translate_text（翻译）                     │    │
│  │  ├─ device_control（设备控制）                  │    │
│  │  ├─ search_web（搜索）                         │    │
│  │  └─ calculate（计算器）                        │    │
│  │                                               │    │
│  │  MCP 工具（第三方）：                           │    │
│  │  ├─ flight_search（机票搜索）                   │    │
│  │  ├─ image_gen（图像生成）                       │    │
│  │  └─ ...                                       │    │
│  │                                               │    │
│  │  每个工具注册：                                 │    │
│  │  { name, description, parameters, schema }    │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  工具调用流程：                                       │
│  LLM 输出 Function Call → Tool Router → 执行工具     │
│  → 返回结果 → LLM 生成最终回复                        │
└──────────────────────────────────────────────────────┘
```

**工具调用实现**：

```go
type ToolExecutor struct {
    registry *ToolRegistry
    limiter  *RateLimiter
}

func (e *ToolExecutor) Execute(toolName string, params map[string]any) (*ToolResult, error) {
    // 1. 查找工具
    tool, ok := e.registry.Get(toolName)
    if !ok {
        return nil, fmt.Errorf("tool %s not found", toolName)
    }
    
    // 2. 参数校验
    if err := tool.ValidateParams(params); err != nil {
        return nil, err
    }
    
    // 3. 限流检查
    if !e.limiter.Allow(toolName) {
        return nil, ErrRateLimited
    }
    
    // 4. 执行工具
    result, err := tool.Execute(params)
    if err != nil {
        return nil, err
    }
    
    // 5. 计量
    e.registry.RecordUsage(toolName)
    
    return result, nil
}
```

### 3.7 RAG 引擎（检索增强生成）

```
┌──────────────────────────────────────────────────────────┐
│                    RAG Pipeline                           │
│                                                          │
│  用户问题                                                  │
│      │                                                    │
│      ▼                                                    │
│  ┌──────────────┐                                        │
│  │ 查询改写      │  ← LLM 改写/扩展查询                    │
│  └──────┬───────┘                                        │
│         │                                                 │
│         ▼                                                 │
│  ┌──────────────────────────────────────┐                │
│  │           检索层                      │                │
│  │                                      │                │
│  │  ┌──────────┐  ┌──────────┐         │                │
│  │  │ 向量检索  │  │ 关键词    │         │                │
│  │  │ (Milvus/  │  │ 检索     │         │                │
│  │  │  FAISS)   │  │ (ES)     │         │                │
│  │  └────┬─────┘  └────┬─────┘         │                │
│  │       └──────┬──────┘               │                │
│  │              ▼                      │                │
│  │       ┌──────────────┐              │                │
│  │       │ 混合排序      │ ← RRF/BM25+  │                │
│  │       │ (RRF 融合)    │   向量相似度  │                │
│  │       └──────┬───────┘              │                │
│  └──────────────┼─────────────────────┘                │
│                 │ Top-K Chunks                          │
│                 ▼                                       │
│  ┌──────────────────────────────────────┐               │
│  │           生成层                      │               │
│  │                                      │               │
│  │  Prompt:                             │               │
│  │  "根据以下参考信息回答问题:            │               │
│  │   {retrieved_chunks}                 │               │
│  │   问题: {user_question}              │               │
│  │   如果参考信息不足以回答问题，          │               │
│  │   请明确告知用户"                      │               │
│  └──────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
```

**传音 RAG 知识库来源**：

| 知识库 | 内容 | 用途 |
|--------|------|------|
| 设备手册 | TECNO/itel/Infinix 全系列说明书 | 设备相关问题解答 |
| FAQ | 常见问题与解决方案 | 客服场景 |
| 产品文档 | AI 功能说明、更新日志 | 产品功能查询 |
| 本地知识 | 各国交通、医疗、教育信息 | 本地生活问答 |

**技术选型**：

| 组件 | 方案 | 理由 |
|------|------|------|
| 向量数据库 | Milvus / FAISS | 大规模向量检索、支持分布式 |
| 关键词检索 | Elasticsearch | 成熟、支持多语言分词 |
| Embedding 模型 | bge-m3 / 自研多语种模型 | 支持 100+ 语言 |
| 排序融合 | RRF（Reciprocal Rank Fusion） | 简单有效、无需训练 |

---

## 四、端到端请求处理流程

### 4.1 完整请求流程

```
用户语音输入："帮我把刚才拍的那张照片翻译成英文，然后发给我妈"
    │
    ▼
[1] ASR: 语音 → 文本 "帮我把刚才拍的那张照片翻译成英文，然后发给我妈"
    │
    ▼
[2] NLU:
    intent: "composite_task"
    entities: [{type: "photo", ref: "last_taken"},
               {type: "language", value: "en"},
               {type: "contact", ref: "mother"}]
    confidence: 0.72
    │
    ▼
[3] Router: confidence 0.72 < 0.95, 复合任务 → LLM Agent
    │
    ▼
[4] Planner:
    Step 1: 获取最近拍摄的照片（工具：photo_get_last）
    Step 2: 图片 OCR 提取文字（工具：ocr_image）
    Step 3: 翻译文字为中→英（工具：translate_text）
    Step 4: 生成翻译后的图片（工具：image_overlay_text）
    Step 5: 发送给联系人（工具：contact_send_photo）
    │
    ▼
[5] 执行（部分并行）:
    Step 1 → photo_get_last → 返回照片路径
    Step 2 → ocr_image(照片) → 返回文字 "你好世界"
    Step 3 → translate_text("你好世界", zh→en) → "Hello World"
    Step 4 → image_overlay_text(照片, "Hello World") → 新图片
    Step 5 → contact_send_photo(母亲, 新图片) → 成功
    │
    ▼
[6] 最终回复: "已将翻译后的照片发送给您母亲"
    │
    ▼
[7] TTS: 文本 → 语音播报
    │
    ▼
[8] Memory 更新:
    短期: 记录本次对话
    长期: 用户偏好"翻译后发送"模式
```

### 4.2 技术栈总览

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| 语言 | Go (主力) + Python (ML 服务) | Go 高并发后端，Python 模型推理 |
| API 框架 | Go gin / fiber | 高性能 HTTP 框架 |
| RPC | gRPC + Protobuf | 内部服务间通信 |
| 缓存 | Redis Cluster | 会话状态、热数据、限流 |
| 搜索引擎 | Elasticsearch | 关键词检索、日志分析 |
| 向量数据库 | Milvus / FAISS | 语义检索、RAG |
| 消息队列 | Kafka / Pulsar | 异步任务、事件流 |
| 模型服务 | vLLM / TGI | LLM 推理服务 |
| 容器编排 | Kubernetes | 服务部署与弹性伸缩 |
| 监控 | Prometheus + Grafana | 指标监控 |
| 日志 | ELK Stack | 日志收集与分析 |
| 链路追踪 | Jaeger | 分布式链路追踪 |

---

## 五、性能优化策略

### 5.1 分层延迟控制

| 路径 | 目标延迟 | 优化手段 |
|------|----------|----------|
| 传统系统路径 | < 100ms | Redis 状态、规则引擎、模板渲染 |
| LLM 直连路径 | < 500ms (首 Token) | vLLM PagedAttention、KV Cache |
| RAG 路径 | < 2s | 向量检索 < 50ms + LLM 推理 |
| Agent 多步路径 | < 5s | 工具并行执行、结果缓存 |

### 5.2 成本优化

| 策略 | 说明 | 预期节省 |
|------|------|----------|
| 意图路由 | 简单意图走传统系统，不调 LLM | ~40% 请求不调 LLM |
| 端侧推理 | 轻量任务端侧完成 | ~20% 请求不上云 |
| 结果缓存 | 相同查询缓存结果（天气/翻译） | ~15% 请求走缓存 |
| 模型分级 | 简单任务用小模型，复杂任务用大模型 | ~30% Token 成本 |
| 批量处理 | 非实时请求批量推理 | ~10% 推理成本 |

### 5.3 可扩展性设计

```
水平扩展：
· NLU Service：无状态，K8s HPA 自动扩缩
· Dialogue Manager：Redis 分片存储会话状态
· Agent Orchestrator：无状态，按需扩展
· Tool Service：每个工具独立部署，独立扩缩
· RAG Service：Milvus 集群扩展

垂直扩展：
· LLM 推理：GPU 实例扩容 + 模型并行
· 向量检索：Milvus 分布式部署
```

---

## 六、与现有架构的对比

| 维度 | 纯传统对话系统 | 纯 LLM Agent | 混合架构（本文方案） |
|------|--------------|-------------|---------------------|
| 意图识别准确率 | ~85%（受限于训练集） | ~95%（但可能幻觉） | **~97%**（高置信走传统，低置信走 LLM） |
| 响应延迟（简单任务） | < 50ms | ~2s | **< 50ms**（路由到传统系统） |
| 响应延迟（复杂任务） | 不支持 | ~5s | ~5s（LLM Agent 路径） |
| LLM 调用比例 | 0% | 100% | **~40%**（60% 被传统系统拦截） |
| 成本 | 低 | 高 | **中等**（路由节省 60% 成本） |
| 多语言支持 | 需逐语言训练 | 原生支持 | 混合（传统系统多语言模板 + LLM 原生） |
| 可扩展性 | 差（每新增意图需训练） | 好（Prompt 驱动） | **好**（传统系统覆盖确定性意图，LLM 处理长尾） |

---

## 七、落地路线

| 阶段 | 时间 | 关键交付 |
|------|------|----------|
| **Phase 1** | M1 | NLU Engine v1（意图分类 + 实体抽取）+ Router v1 |
| **Phase 2** | M2 | 传统对话系统迁移（确定性意图从 LLM 路由到规则引擎） |
| **Phase 3** | M3 | Memory 系统（短期 + 长期）+ RAG Engine v1 |
| **Phase 4** | M4-M5 | LLM Agent Planner + 工具系统 + MCP 接入 |
| **Phase 5** | M6 | 性能优化（缓存、端侧推理、模型分级） |
| **Phase 6** | M7-M9 | 全球化部署 + 多语言全面覆盖 |
| **Phase 7** | M10-M12 | 自适应路由（基于反馈的自动路由优化） |

---

## 八、总结

混合架构的核心价值在于**让每个请求走最合适的路径**：

| 路径 | 占比（预估） | 特点 |
|------|-------------|------|
| 传统系统 | ~40% | 快（< 50ms）、准（规则驱动）、省（无 LLM 成本） |
| LLM + 工具 | ~20% | 需要外部操作的场景（查天气、发消息、控制设备） |
| LLM + RAG | ~15% | 需要外部知识的场景（设备手册、FAQ、本地知识） |
| LLM Agent | ~10% | 复杂多步推理（旅行规划、课业辅导） |
| 自由对话 | ~15% | 闲聊、创意生成、开放问答 |

这套架构既保留了传统对话系统的**确定性、低延迟、低成本**，又获得了 LLM 的**泛化能力、推理能力、创造力**。对于传音 Ella 这样日处理亿级请求的硬件助手，混合架构不是选择，而是必然。
