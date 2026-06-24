# 🏋️ Fitness RAG — 健身知识问答系统

[![CI](https://github.com/jackal-black/fitness-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/jackal-black/fitness-rag/actions/workflows/ci.yml)

基于检索增强生成（RAG）的健身知识问答系统。支持 Hybrid Search、多轮对话、流式输出。

## 功能

| 特性 | 说明 |
|------|------|
| 🔍 **Hybrid Search** | TF-IDF + BM25 双路检索 + Reciprocal Rank Fusion 排序 |
| 💬 **多轮对话** | Session 管理 + 滑动窗口历史，支持追问 |
| ⚡ **流式输出** | SSE 实时推送，先字延迟 < 500ms |
| 📊 **评估管线** | 20 道测试题量化 Hit Rate / MRR / NDCG |
| 🖥️ **Web UI** | 浏览器直接交互，支持流式显示 |
| 🐳 **Docker** | 一键容器化部署 |

## 检索效果对比

| 策略 | Hit Rate@1 | MRR |
|------|-----------|-----|
| TF-IDF（基线） | 85% | 0.925 |
| BM25 | **90%** | **0.950** |
| Hybrid (RRF) | 85% | 0.925 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 建立索引
python ingest.py data/*.md

# 启动 Web 服务
uvicorn main:app --reload

# 打开浏览器访问
open http://localhost:8000
```

### Docker 部署

```bash
docker-compose up --build
```

## 项目结构

```
fitness-rag/
├── ingest.py              # 索引构建（TF-IDF + BM25）
├── query.py               # 问答引擎（检索 + LLM + 流式）
├── main.py                # FastAPI 服务（REST + Web UI）
├── eval/                  # 评估管线
│   ├── questions.json     #   20 道领域测试题
│   └── run.py             #   Hit Rate / MRR / NDCG 计算
├── data/                  # 健身知识库（可替换为你的资料）
├── static/                # Web UI
│   └── index.html         #   交互式问答界面
├── .github/workflows/     # CI 流水线
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 技术栈

- **检索**：scikit-learn TF-IDF + rank-bm25 + RRF 融合
- **框架**：FastAPI + SSE 流式
- **LLM**：OpenAI / DeepSeek / Ollama（通过 .env 配置）
- **评估**：Hit Rate@K, Mean Reciprocal Rank, NDCG
- **部署**：Docker + docker-compose + GitHub Actions CI

## 自定义知识库

将你自己的健身资料（.md / .pdf）放入 `data/` 目录，重新索引：

```bash
python ingest.py data/*.md --reset
```

支持中英文，无需下载任何 NLP 模型。
