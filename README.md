# Fitness RAG — 健身知识问答系统

基于检索增强生成（RAG）的健身知识问答系统，支持中文。使用 TF-IDF + BM25 双路检索，支持多轮对话和流式输出。

项目演示：[Web Demo](https://your-demo-url) · [API 文档](http://localhost:8000/docs)

## 功能特性

| 特性 | 说明 |
|------|------|
| 🔍 Hybrid Search | TF-IDF + BM25 双路检索 + RRF 融合排序 |
| 💬 多轮对话 | Session 管理 + 滑动窗口历史，支持追问 |
| ⚡ 流式输出 | SSE 实时推送，逐 token 显示 |
| 📊 Eval 管线 | 20 道测试题，量化 Hit Rate/MRR/NDCG |
| 🐳 Docker | 容器化一键部署 |
| 🔌 API | RESTful + Swagger 文档 |

## 效果对比

| 策略 | Hit Rate@1 | MRR |
|------|-----------|-----|
| TF-IDF（基线） | 85% | 0.925 |
| **BM25** 🏆 | **90%** | **0.950** |
| Hybrid (RRF) | 85% | 0.925 |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 LLM（可选）
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY

# 3. 放入健身资料到 data/（支持 .md 和 .pdf）

# 4. 建立索引
python ingest.py data/*.md

# 5. 问答
python query.py "深蹲膝盖痛怎么办"              # 纯检索
python query.py --llm "减脂期蛋白质怎么吃"       # LLM 回答
python query.py --llm --retriever bm25 "训练频率" # 指定检索策略

# 6. 启动 API
uvicorn main:app --reload
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/ask` | POST | 单轮问答 |
| `/ask/stream` | POST | 流式问答（SSE） |
| `/chat` | POST | 多轮对话（传 session_id） |
| `/sessions/{id}` | DELETE | 删除会话 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 文档 |

### POST /chat（多轮对话）

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"深蹲的标准动作是什么"}'

# 返回 session_id，下次带上它就能延续对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"那膝盖内扣怎么纠正","session_id":"66f22496"}'
```

## Docker 部署

```bash
docker-compose up --build
```

## 项目结构

```
fitness-rag/
├── data/               # 健身资料
│   ├── strength-training-basics.md
│   ├── nutrition-for-muscle-growth.md
│   └── squat-technique.md
├── eval/               # 评估管线
│   ├── questions.json  #   20 道测试题
│   └── run.py          #   Hit Rate/MRR/NDCG
├── index/              # 向量索引（自动生成）
├── ingest.py           # 索引构建（TF-IDF + BM25）
├── query.py            # 问答引擎（Hybrid Search + 流式）
├── main.py             # FastAPI 服务
├── Dockerfile          # 容器部署
├── docker-compose.yml
└── requirements.txt
```

## 技术栈

| 组件 | 选型 |
|------|------|
| 检索 | scikit-learn TF-IDF + rank-bm25 |
| 融合 | Reciprocal Rank Fusion (RRF) |
| 框架 | FastAPI |
| LLM | OpenAI / DeepSeek / Ollama |
| 部署 | Docker + docker-compose |
| 评估 | Hit Rate, MRR, NDCG |
