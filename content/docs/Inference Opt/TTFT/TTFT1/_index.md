
---
title: LLM 首字延迟（TTFT）优化方案
weight: 10
---


# LLM 首字延迟 (TTFT) 优化全栈指南

> "In interactive systems, the speed of the first response is not a performance metric — it is a trust signal. If the user waits too long, they assume the system is broken, not thinking."
> —— 改编自 Jakob Nielsen, *Response Times: The 3 Important Limits* [1]

## 一、TTFT 的定义与核心公式

### 1.1 什么是 TTFT？
**TTFT (Time To First Token)** 指从客户端发出请求，到模型生成并返回第一个 Token 的端到端时间。

在 LLM 的自回归生成过程中，推理分为两个截然不同的阶段：
- **Prefill (预填充)**：一次性计算整个 Prompt 的 KV Cache。计算密集，复杂度 $O(N^2)$。
- **Decode (解码)**：自回归逐个生成 Token。访存密集（Memory-bound），复杂度 $O(N)$。

**TTFT 几乎完全由 Prefill 阶段决定。**

### 1.2 TTFT 的核心组成

```
用户点击发送 ─────────────────────────────────────────────────────────────► 看到第一个字
                 │          │           │            │
                 ▼          ▼           ▼            ▼
            [ 网络传输 ] [ 排队等待 ] [ 调度分配 ] [ Prefill 计算 ]
            (T_network)  (T_queue)   (T_sched)     (T_prefill)
            ─────────────┴───────────┴────────────┘
                       端到端 TTFT
```

| 组件 | 含义 | 典型占比 | 优化方向 |
|------|------|---------|---------|
| $T_{queue}$ | 请求在队列中等待的时间 | 0% ~ 80% (高并发时) | 连续批处理、弹性扩缩容 |
| $T_{network}$ | 网络传输与协议握手延迟 | 5% ~ 15% | HTTP/2、连接池、网关优化 |
| $T_{scheduling}$ | 调度器分配 GPU 资源的时间 | < 5% | 异步调度、资源预热 |
| $T_{prefill}$ | 模型计算 Prompt 的 KV Cache | 40% ~ 90% | Flash Attention、Chunked Prefill、PD 分离 |

**用户体验的心理阈值**：
- **< 100ms**：即时响应，用户感觉“直接操作”。
- **100ms ~ 500ms**：流畅，用户能感知轻微延迟但可接受。
- **> 2000ms**：用户产生焦虑，可能重复提交或放弃。

---

## 二、TTFT 的核心瓶颈拆解

### 2.1 Prefill 的计算与访存瓶颈

```
    GPU SRAM (高速缓存)
    ┌─────────────────┐
    │                 │◄───── 权重加载 (HBM -> SRAM)
    │    Compute      │       (140GB for 70B)
    │   (ALU Units)   │
    │                 │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │                 │
    │      HBM        │◄───── 显存带宽瓶颈
    │   (3.35 TB/s)   │       数据搬运 > 实际计算
    └─────────────────┘
```

在 Prefill 阶段，尽管 FLOPS 很高，但现代 GPU (H100/A100) 的算力远超显存带宽。Prefill 阶段大部分时间在**等待数据从 HBM 搬运到 SRAM**，而非实际计算。

### 2.2 排队延迟（The Hidden Killer）

传统推理服务使用 **Static Batching**：一个 Batch 填满后一起送入 GPU，Batch 中所有请求必须等待最长的那个 Prefill 完成才能返回。

```
传统 Static Batching (木桶效应):
───────────────────────────────────────────────────────────── 时间
Req A (短):  [ P ============== ] [ D D D D ]
Req B (长):  [ P ================================ ] [ D D D ]
Req C (中):  [................... 等待 ...................] [ P =========== ] [ D D ]
                                  ▲
                           Req C 的 TTFT 被 Req B 严重阻塞
```
在高并发下，$T_{queue}$ 成为 TTFT 的主导因素。

---

## 三、推理引擎层优化（核心战场）

### 3.1 连续批处理 (Continuous Batching)

vLLM 的核心创新，彻底打破 Static Batching 的木桶效应。只要 Batch 中有任何一个请求完成 Decode 腾出 Slot，调度器立刻将排队的请求插入，并执行其 Prefill。

```
Continuous Batching (动态插空):
───────────────────────────────────────────────────────────── 时间
Req A (短):  [ P ] [ D D D D ]
Req B (长):         [ P ==== ] [ D D D ]
Req C (中):              [ P === ] [ D D D ]
Req D (短):                   [ P ] [ D D D ]
             ▲
       所有请求几乎同时开始 Prefill，TTFT 显著降低
```
**效果**：$T_{queue}$ 趋近于 0，GPU 利用率从 30% 提升至 80%+。

### 3.2 块级预填充 (Chunked Prefill)

Continuous Batching 解决了排队问题，但**单个超长请求仍会阻塞整个 GPU**（因为 Prefill 不可中断）。

```
Chunked Prefill 机制:
───────────────────────────────────────────────────────────── 时间
Req Long (32k): [ P1 ] [ P2 ] [ P3 ] [ P4 ] [ P5 ] [ P6 ] ...
Req Short (A):        [ P == ] [ D D ]
Req Short (B):               [ P == ] [ D D ]
             ▲
       长 Prompt 被切块 (Chunk)，与短请求的 Decode 交替执行
```

- **原理**：将长 Prompt 拆分为固定大小的 Chunk（如 512 Token）。每次迭代只计算一个 Chunk，并与 Decode 请求交替执行。
- **效果**：长尾延迟（P99 TTFT）下降 60-80%。

### 3.3 KV Cache 复用：Prefix Caching & RadixAttention

在实际业务中（如多轮对话、Agent 工具调用），大量请求共享相同的系统 Prompt。

```
RadixAttention 缓存树 (Trie):
                [Root]
               /      \
           [SysPrompt]
           /         \
      [User_A]      [User_B]
      /    \        /    \
 [History] [New] [History] [New]
   ▲          ▲      ▲
   └──────────┴──────┘
     直接复用已计算的 KV Cache
     T_prefill ≈ T_load_cache (< 5ms)
```

**效果**：命中前缀的请求，$T_{prefill}$ 可降至极低，因为跳过了繁重的矩阵乘法。

---

## 四、系统架构层优化（资源调度）

### 4.1 PD 分离架构 (Prefill-Decode Disaggregation)

2024-2025 年工业界最热门的架构演进。将 Prefill（计算密集）和 Decode（访存密集）拆分到不同实例。

```
┌──────────────────────────────────────────────────────────────┐
│                        API Gateway                           │
│  (路由逻辑: Prompt 长度 > 8k 走 Prefill 集群，否则走 Decode)   │
└───────────────┬────────────────────────────┬─────────────────┘
                │                            │
     (长 Prompt / 计算密集)          (短 Prompt / 访存密集)
                │                            │
                ▼                            ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │    Prefill Cluster   │      │    Decode Cluster    │
    │   (H100 × 4, 高算力)  │      │   (L40S × 8, 高显存) │
    │                      │      │                      │
    │  [ P1 ] [ P2 ] [ P3] │      │  [ D1 ] [ D2 ] [ D3] │
    │  [ P1 ] [ P2 ] [ P3] │      │  [ D1 ] [ D2 ] [ D3] │
    │  [ P1 ] [ P2 ] [ P3] │      │  [ D1 ] [ D2 ] [ D3] │
    └──────────┬───────────┘      └──────────▲───────────┘
               │                            │
               │     KV Cache Transfer      │
               └────────────────────────────┘
                  (RDMA / P2P / Shared Mem)
```

**优势**：
1.  **资源匹配**：Prefill 吃算力，Decode 吃显存带宽。不再让昂贵的 H100 被 Decode 的低带宽利用率浪费。
2.  **隔离干扰**：长 Prompt 的 Prefill 不再抢占 Decode 的 GPU 时间片。
3.  **弹性扩缩容**：可根据业务峰谷独立扩容 Prefill 或 Decode 节点。

**代表项目**：DistServe, Splitwise, Mooncake, vLLM (PD Disaggregation)。

### 4.2 Flash Attention 2/3 技术

传统 Attention 实现需要多次读写 HBM，Flash Attention 通过 **IO-Aware** 分块计算，将中间结果保留在 SRAM 中。

| 版本 | 核心优化 | 适用硬件 | Prefill 提速 |
|------|---------|---------|-------------|
| **Flash Attention 1** | 分块计算，减少 HBM 读写 | A100/V100 | 1.5x - 2x |
| **Flash Attention 2** | 优化线程块调度，减少同步 | A100/H100 | 2x - 3x |
| **Flash Attention 3** | Hopper TMA 异步加载指令 | H100/H200 | 3x - 5x |

---

## 五、常见坑点与避坑指南

| # | 坑点 | 现象 | 根因分析 | 解决方案 | 严重程度 |
|---|------|------|---------|---------|---------|
| 1 | **长 Prompt 阻塞全局** | P99 TTFT 飙升，短请求排队 > 2s | 静态 Batch 或未开启 Chunked Prefill | 开启 `--enable-chunked-prefill`，设置合理 Chunk 大小 | 🔴 致命 |
| 2 | **显存碎片化** | 请求被 OOM 拒绝或频繁触发 Swap | 传统 KV Cache 分配非连续 | 使用 vLLM/SGLang 的 PagedAttention | 🔴 致命 |
| 3 | **TP 并行度过大** | 算力利用率低，通信占比 > 30% | AllReduce 同步延迟抵消计算收益 | 控制 TP≤4（A100）或 ≤8（H100） | 🟠 高 |
| 4 | **KV Cache 未清理** | 显存泄漏，运行数小时后 OOM | 异常请求或超时连接未释放 Cache | 配置 TTL + 定期 GC，启用 Paged Cache | 🟡 中 |
| 5 | **网关缓冲流式响应** | 首字延迟高，但后续输出极快 | 网关等待整个响应打包后才下发 | 开启 SSE 实时透传，禁用响应缓冲 | 🟠 高 |

---

## 六、总结与实战建议

### 6.1 分层落地路线图

| 场景 | 推荐技术栈 | 预期 TTFT | 实施成本 |
|------|-----------|----------|---------|
| **轻量级/原型** | vLLM + FlashAttn2 + PagedAttention | 500ms ~ 1s | 低（开箱即用） |
| **高并发生产** | vLLM/SGLang + Chunked Prefill + RadixAttention | 200ms ~ 500ms | 中（需调参） |
| **极致低延迟/长文本** | PD 分离 + Speculative Prefill + H100/L40S 异构 | < 200ms (128k) | 高（架构改造） |

### 6.2 给架构师的 3 条核心建议
1.  **先开 PagedAttention 和 Continuous Batching**：这是 vLLM/SGLang 的默认能力，不花一分钱硬件成本，可解决 80% 的排队和碎片问题。
2.  **长文本必开 Chunked Prefill**：不要相信"GPU 算力够就能硬扛长 Prompt"。Chunk 是保证 P99 延迟不雪崩的唯一工程手段。
3.  **TTFT 与 Throughput 是 trade-off**：过度优化 TTFT（如极小 Chunk）会牺牲吞吐。根据业务 SLA 设定合理的 Chunk Size。

---

*文档版本：v2.0 (图文并茂版) | 作者：小伟 | 日期：2026-07-29*
