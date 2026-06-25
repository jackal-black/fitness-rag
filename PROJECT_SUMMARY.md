# Fitness RAG — 项目总结文档

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


