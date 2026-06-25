 # 🏋️ Fitness RAG — 健身知识问答系统

 [![CI](https://github.com/jackal-black/fitness-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/jackal-blac
k/fitness-rag/actions/workflows/ci.yml)

 基于检索增强生成（RAG）的健身知识问答系统。支持 5 种检索策略、多轮对话、流式输出。

 ## 功能

 | 特性 | 说明 |
 |------|------|
 | 🔍 **5 种检索策略** | TF-IDF / BM25 / Embedding（语义）/ Hybrid 双路 / Hybrid3 三路 |
 | 💬 **多轮对话** | Session 管理 + 滑动窗口历史，支持追问 |
 | ⚡ **流式输出** | SSE 实时推送，首字延迟 < 500ms |
 | 📊 **评估管线** | 20 道测试题量化 Hit Rate / MRR / NDCG |
 | 🖥️ **Web UI** | 浏览器直接交互，支持流式显示 |
 | 🐳 **Docker** | 一键容器化部署 |

 ## 检索效果对比

 | 策略 | 类型 | Hit Rate@1 | MRR |
 |------|------|-----------|-----|
 | TF-IDF | 关键词（字符 n-gram） | 85% | 0.925 |
 | BM25 🏆 | 词频 + 长度归一 | **90%** | **0.950** |
 | Embedding | 语义（all-MiniLM-L6-v2） | — | — |
 | Hybrid | TF-IDF + BM25 双路融合 | 85% | 0.925 |
 | Hybrid3 | TF-IDF + BM25 + Embedding 三路融合 | — | — |

 > 当前知识库 3 个文件，关键词检索已足够。架构预留了 Embedding 接口，知识库扩大后可无缝升级。

 ## 快速开始

 ```bash
 # 1. 安装依赖
 pip install -r requirements.txt

 # 2. （可选）安装 Embedding 模型，启用语义检索
 pip install sentence-transformers
 $env:HF_ENDPOINT = "https://hf-mirror.com"
 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

 # 3. 建立索引
 python ingest.py data/*.md --reset

 # 4. 问答
 python query.py "深蹲膝盖痛怎么办"                      # 默认 Hybrid 检索
 python query.py --retriever embedding "膝关节疼痛"       # 语义检索
 python query.py --retriever hybrid3 "膝关节疼痛"         # 三路融合
 python query.py --llm "减脂期蛋白质怎么吃"               # LLM 回答

 # 5. 启动 Web UI
 uvicorn main:app --reload
 # 浏览器打开 http://localhost:8000

 Docker 部署

 bash
 docker-compose up --build

 API

 接口        方法 说明
 ─────────── ──── ───────────────
 /ask        POST 单轮问答
 /ask/stream POST 流式问答（SSE）
 /chat       POST 多轮对话
 /health     GET  健康检查
 /docs       GET  Swagger 文档

 /ask  请求示例：

 json
 {
   "question": "深蹲膝盖痛怎么办",
   "top_k": 5,
   "llm": true,
   "retriever": "hybrid"
 }

 retriever  可选： tfidf  /  bm25  /  embedding  /  hybrid （双路）/  hybrid3 （三路）

 项目结构

 fitness-rag/
 ├── ingest.py              # 索引构建（TF-IDF + BM25 + Embedding）
 ├── query.py               # 问答引擎（5 种检索策略 + LLM + 流式）
 ├── main.py                # FastAPI 服务（REST + Web UI）
 ├── eval/                  # 评估管线（20 道测试题）
 ├── data/                  # 健身知识库（可替换）
 ├── static/index.html      # Web UI
 ├── .github/workflows/     # CI 流水线
 ├── Dockerfile
 ├── docker-compose.yml
 └── requirements.txt

 技术栈

 · 检索：scikit-learn TF-IDF + rank-bm25 + sentence-transformers + RRF
 · 框架：FastAPI + SSE 流式
 · LLM：OpenAI / DeepSeek / Ollama
 · 评估：Hit Rate@K, MRR, NDCG
 · 部署：Docker + docker-compose + GitHub Actions CI

 自定义知识库

 bash
 # 把你的资料放进 data/，重新索引即可
 python ingest.py data/*.md --reset

支持 .md 和 .pdf 格式。
