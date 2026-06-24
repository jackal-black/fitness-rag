# Fitness RAG — 项目总结文档

> 用途：每次改简历或面试前，把这份文档丢给 LLM，它能基于真实信息给你建议和模拟面试。
> 最后更新：2025-06-25

---

## 一、项目概述

| 项目 | 说明 |
|------|------|
| 名称 | Fitness RAG — 健身知识问答系统 |
| 定位 | 基于检索增强生成（RAG）的领域问答系统 |
| 仓库 | https://github.com/jackal-black/fitness-rag |
| 状态 | 已完成核心功能，可稳定运行 |

---

## 二、技术栈

| 组件 | 选型 | 为什么选它 |
|------|------|-----------|
| 检索 | scikit-learn TfidfVectorizer + rank-bm25 | 零模型下载，无需 GPU，秒级索引 |
| 融合 | Reciprocal Rank Fusion (RRF) | 简单有效的多路结果合并算法 |
| 框架 | FastAPI | 轻量高性能，原生支持 SSE |
| LLM | OpenAI / DeepSeek / Ollama（通过 .env 配置） | OpenAI 兼容 API，一键切换 |
| 前端 | 纯 HTML + JS | 无框架依赖，一个文件搞定 |
| 部署 | Docker + docker-compose | 容器化标准方案 |
| CI | GitHub Actions | 每次 push 自动跑 eval |
| 依赖 | scikit-learn, rank-bm25, fastapi, uvicorn, openai, pypdf, python-dotenv | 总计约 7 个依赖 |

---

## 三、系统架构

```
用户输入
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│ 查询改写     │     │  知识库（data/）   │
│ (多轮 -> 单轮)│     │  .md / .pdf 文件  │
└──────┬──────┘     └────────┬─────────┘
       │                     │
       ▼                     ▼
┌──────────────────────────────────────┐
│           检索器（Hybrid）              │
│  ┌────────┐  ┌────────┐  ┌────────┐  │
│  │ TF-IDF │  │  BM25  │  │ RRF    │  │
│  │ 关键词  │  │ 词频+  │  │ 融合   │  │
│  │ 字符ng │  │ 长度归一│  │ 排序   │  │
│  └────────┘  └────────┘  └────────┘  │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│           Prompt 构建                  │
│   检索结果 + 对话历史 → 结构化 Prompt   │
└────────────────┬─────────────────────┘
                 │
         ┌───────┴───────┐
         ▼                ▼
   ┌──────────┐    ┌──────────┐
   │ 纯检索输出 │    │ LLM 生成  │
   │ (显示原文) │    │ (DeepSeek)│
   └──────────┘    └──────────┘
```

### 数据流

```
ingest.py：文件 → chunk → TF-IDF向量 + BM25索引 → pickle 持久化
query.py：用户问题 → (TF-IDF + BM25) → RRF融合 → top-5 结果
main.py：结果 → 拼 prompt → LLM回答 → SSE 流式返回
```

---

## 四、核心设计决策

### 4.1 为什么不用 Embedding 模型？

| 因素 | 决策 |
|------|------|
| 知识库规模 | 仅 3 个文件、9 个文本块，关键词足够区分 |
| 启动速度 | Embedding 需下载 80MB ONNX 模型，TF-IDF 秒级 |
| 离线能力 | TF-IDF + BM25 完全离线，Embedding 需首次下载 |
| 结论 | 当前规模 TF-IDF/BM25 是最优解，≥50 文件再升级 |

### 4.2 为什么不用 LangChain / LlamaIndex？

- 目的是展示对底层原理的理解（面试加分）
- TF-IDF、BM25、RRF 融合均自主实现
- 不引入框架级依赖，项目更轻量

### 4.3 为什么选 BM25 作为最优策略？

- TF-IDF：只考虑词频 × 逆文档频率
- BM25：额外加了文档长度归一化和词频饱和
- 中文场景下，字符 2-gram + BM25 对短文本检索更友好

---

## 五、评估结果

### 5.1 测试集

- 20 道手工标注题目
- 覆盖 3 个文件：squat-technique.md（8道）、nutrition-for-muscle-growth.md（6道）、strength-training-basics.md（6道）
- 包含 2 道跨文件综合题
- 评估方式：每道题标注期望召回的源文件名

### 5.2 基线数据

| 策略 | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR | NDCG@5 |
|------|-----------|-----------|-----------|-----|--------|
| TF-IDF（基线） | 85% | 100% | 100% | 0.9250 | 2.0682 |
| BM25 🏆 | **90%** | 100% | 100% | **0.9500** | **2.1493** |
| Hybrid (RRF) | 85% | 100% | 100% | 0.9250 | 2.0822 |

**结论：** BM25 在当前数据上最优，HR@1 90%。所有策略在 top-3 内均能召回相关文档。

### 5.3 评估命令

```bash
python eval/run.py --retriever tfidf    # 评估 TF-IDF
python eval/run.py --retriever bm25     # 评估 BM25
python eval/run.py --retriever hybrid   # 评估 Hybrid
python eval/run.py --html               # 生成 HTML 报告
```

---

## 六、API 接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 返回 Web UI 或 API 信息 |
| `/health` | GET | 健康检查（返回 index_ready, llm_ready） |
| `/ask` | POST | 单轮问答 |
| `/ask/stream` | POST | 流式问答（SSE） |
| `/chat` | POST | 多轮对话（支持 session_id） |
| `/sessions/{id}` | DELETE | 删除会话 |
| `/docs` | GET | Swagger 文档 |

### 请求示例

```json
POST /ask
{
  "question": "深蹲膝盖痛怎么办",
  "top_k": 5,
  "llm": true,
  "retriever": "hybrid"
}
```

---

## 七、简历用法

### 7.1 项目描述（详细版）

> **Fitness RAG — 健身知识问答系统**
>
> 基于检索增强生成（RAG）的健身领域问答系统，支持混合检索、多轮对话和流式输出。
>
> - 实现 **TF-IDF + BM25 双路检索** + Reciprocal Rank Fusion 融合排序，在 20 道领域测试题上 Hit Rate@1 达 90%，MRR 0.950
> - 设计 **Session 管理 + 滑动窗口历史**的多轮对话引擎，支持上下文追问
> - 基于 **SSE 流式输出**实现逐 token 实时显示，内建 **Eval 管线**量化 Hit Rate/MRR/NDCG
> - 使用 FastAPI 构建 RESTful API + Web UI，Docker 容器化部署，**GitHub Actions CI** 自动运行评估
> - 技术栈：Python, scikit-learn, FastAPI, DeepSeek/OpenAI API, Docker

### 7.2 精简版（放技能栏）

> 自建 RAG 系统，TF-IDF + BM25 双路检索 + RRF 融合，HR@1 达 90%。支持多轮对话、SSE 流式、Eval 管线。FastAPI + Docker + GitHub Actions。

### 7.3 英文版（投外企）

> **Fitness RAG — Domain-specific Q&A System**
>
> - Implemented hybrid search (TF-IDF + BM25) with Reciprocal Rank Fusion, achieving **90% Hit Rate@1** and **0.950 MRR** on 20 domain test questions
> - Designed session-based multi-turn dialogue with sliding window context
> - Built SSE streaming response and eval pipeline (Hit Rate, MRR, NDCG)
> - Containerized with Docker, automated CI via GitHub Actions
> - Stack: Python, scikit-learn, FastAPI, DeepSeek/OpenAI API, Docker

---

## 八、面试 QA 库

### Q1：为什么 BM25 比 TF-IDF 好？

> TF-IDF 的评分公式是 TF × IDF，只考虑词频和逆文档频率。BM25 在此基础上加了两个关键改进：
> 1. **词频饱和**：一个词出现 10 次和 20 次，对相关性的增益不是线性的，BM25 用 k1 参数抑制高频词的过度影响
> 2. **文档长度归一化**：长文档更容易出现更多关键词，BM25 通过 b 参数对长文档做惩罚
>
> 在健身问答这种短文本场景下，BM25 的 HR@1 比 TF-IDF 高 5 个百分点（85% → 90%）。

### Q2：RAG 和 Fine-tuning 有什么区别？

> **RAG**：检索外部知识 + 生成回答。适合知识频繁更新、需要引用来源的场景。不需要训练，成本低。
>
> **Fine-tuning**：让模型学习特定领域的表达方式和知识。适合输出风格固定、不需要外部知识的场景。需要标注数据、训练成本高。
>
> 健身问答场景适合 RAG，因为：
> - 知识可以随时新增（新的训练方法、营养指南）
> - 用户需要看到来源才能信任回答
> - 不需要模型"记住"特定的输出格式

### Q3：你的 Eval 怎么设计的？

> 我手工标注了 20 道测试题，每道题预设了期望召回的源文件名。评估指标：
> - **Hit Rate@K**：top-k 结果中是否包含期望文件
> - **MRR**：第一个相关文档排名的倒数均值
> - **NDCG**：考虑排序位置的归一化累计增益
>
> 这样能定量对比不同检索策略（TF-IDF vs BM25 vs Hybrid）的效果差异。

### Q4：为什么不直接用 GPT/DeepSeek 问？

> 通用模型的问题：
> 1. **时效性**：不知道最新研究成果
> 2. **权威性**：不知道回答来自哪个来源
> 3. **一致性**：同一个问题不同时间问，答案可能不同
>
> RAG 的优势是**可控**——回答基于我选的特定资料，来源可追溯，也方便替换为机构内部知识库。

### Q5：知识库扩大会有什么问题？怎么解决？

> TF-IDF/BM25 超过 50 个文件后效果下降，因为关键词匹配在大量文档中噪声增加。解决方案是升级到 Embedding 语义检索，做三路融合（TF-IDF + BM25 + Embedding），再上层加 Cross-encoder Reranker。

### Q6：哪些代码是你自己写的，哪些是调库？

> 核心逻辑全部自己实现：
> - TF-IDF 调用 sklearn，但中文分词的 2-gram 策略是自己设计的
> - BM25 调 rank-bm25 库，但中文分词器是自己写的
> - RRF 融合算法自己实现
> - Prompt 模板自己设计
> - Eval 管线全部自己写
> - Web UI 纯手写 HTML + JS
>
> 调库部分：FastAPI（框架）、uvicorn（服务器）、openai（LLM 调用）

---

## 九、后续优化路线图

| 优先级 | 优化项 | 触发条件 | 工作量 |
|--------|--------|---------|--------|
| P0 | 扩充知识库至 20+ 文件 | 立即可以做 | 1-2 小时 |
| P0 | 按 Markdown 标题分块 | 知识库 > 10 文件时 | 20 行代码 |
| P1 | 查询改写（多轮优化） | 多轮对话效果不满时 | 15 行代码 |
| P1 | 加一路 Embedding 检索 | 知识库 > 50 文件时 | 50 行代码 + 80MB 模型 |
| P2 | Cross-encoder Reranker | HR@1 < 80% 时 | 30 行代码 |
| P2 | 溯源高亮 | 用户体验不满时 | HTML + CSS 改 |

---

## 十、项目文件索引

```
fitness-rag/
├── ingest.py          # 索引构建（核心文件约 140 行）
├── query.py           # 检索 + LLM + 流式（核心文件约 200 行）
├── main.py            # FastAPI 服务（约 160 行）
├── eval/
│   ├── run.py         # 评估管线（约 150 行）
│   └── questions.json # 20 道测试题
├── data/              # 知识库源文件
├── static/index.html  # Web UI（纯 HTML+JS）
├── .github/workflows/ci.yml  # CI 流水线
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

---

## 十一、对话模板（发简历时用）

### 投递附言

> 您好，附件是我的简历。其中 Fitness RAG 项目是我自建的一个领域问答系统，实现了 TF-IDF + BM25 混合检索、多轮对话、流式输出，并搭建了评估管线量化效果。GitHub 链接：https://github.com/jackal-black/fitness-rag
>
> 如果对这个项目感兴趣，我可以详细给你讲技术细节。

### 面试开场自我介绍

> 我之前做了一个健身 RAG 问答系统，核心是实现了 TF-IDF 和 BM25 双路检索，用 RRF 融合排序，在 20 道测试题上 Hit Rate@1 达到 90%。支持多轮会话和流式输出，用 FastAPI 搭的接口，Docker 部署，GitHub Actions 自动跑评估。
>
> 这个项目让我对 RAG 的检索策略、评估方法和工程化有了比较完整的实践。后面如果想用在实际产品中，可以升级 Embedding 检索和对知识库进行更细颗粒度的管理。
