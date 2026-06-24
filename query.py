"""
Fitness RAG — 问答引擎 (Hybrid: TF-IDF + BM25 + RRF Fusion)

用法:
  python query.py "深蹲的正确姿势是什么"
  python query.py "蛋白质每天吃多少"
  python query.py --llm "你的问题"   (需要 OPENAI_API_KEY in .env)
  python query.py --retriever tfidf / bm25 / hybrid
"""

import argparse
import os
import pickle
import re
from pathlib import Path

# 加载 .env 文件（如果存在）
from dotenv import load_dotenv
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = "index"
INDEX_FILE = "fitness_index.pkl"
TOP_K = 5
RRF_K = 60  # RRF 常数


def tokenize_chinese(text: str) -> list[str]:
    """和 ingest.py 保持一致的分词"""
    tokens = []
    for word in re.findall(r"[a-zA-Z_]+", text):
        if len(word) > 1:
            tokens.append(word.lower())
    clean = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(len(clean) - 1):
        tokens.append(clean[i:i+2])
    return tokens


def load_index():
    base = Path(__file__).parent
    path = base / INDEX_DIR / INDEX_FILE
    if not path.exists():
        print(f"❌ 索引文件不存在: {path}")
        print("   请先运行: python ingest.py data/*.md")
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_tfidf(vectorizer, matrix, query: str, top_k: int = TOP_K):
    """TF-IDF 检索"""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    return [(idx, float(scores[idx])) for idx in top_indices]


def retrieve_bm25(bm25, tokenized_corpus, query: str, top_k: int = TOP_K):
    """BM25 检索"""
    tokens = tokenize_chinese(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[::-1][:top_k]
    return [(idx, float(scores[idx])) for idx in top_indices]


def retrieve_embedding(embeddings, query: str, top_k: int = TOP_K):
    """Embedding 语义检索"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_vec = model.encode([query])
        scores = cosine_similarity(q_vec, embeddings).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(idx, float(scores[idx])) for idx in top_indices]
    except Exception as e:
        print(f"  ⚠️ Embedding 检索失败: {e}")
        return []


def retrieve(query: str, top_k: int = TOP_K, retriever: str = "hybrid"):
    """
    统一检索入口。

    retriever 参数:
      - "tfidf":      仅 TF-IDF（关键词）
      - "bm25":       仅 BM25（词频+长度归一）
      - "embedding":  仅 Embedding（语义）
      - "hybrid":     TF-IDF + BM25 双路融合（默认）
      - "hybrid3":    TF-IDF + BM25 + Embedding 三路融合
    """
    data = load_index()
    if data is None:
        return None

    chunks = data["chunks"]
    metas = data["metadatas"]

    if retriever == "tfidf":
        results = retrieve_tfidf(data["vectorizer"], data["matrix"], query, top_k)
    elif retriever == "bm25":
        results = retrieve_bm25(data["bm25"], data["tokenized_corpus"], query, top_k)
    elif retriever == "embedding":
        if "embeddings" not in data:
            print("  ⚠️ 索引中没有 Embedding，请运行 python ingest.py 重新建索引")
            return None
        results = retrieve_embedding(data["embeddings"], query, top_k)
    elif retriever == "hybrid3":
        # 三路融合
        if "embeddings" not in data:
            print("  ⚠️ 索引中没有 Embedding，请运行 python ingest.py 重新建索引")
            return None
        tfidf_results = retrieve_tfidf(data["vectorizer"], data["matrix"], query, top_k * 2)
        bm25_results = retrieve_bm25(data["bm25"], data["tokenized_corpus"], query, top_k * 2)
        emb_results = retrieve_embedding(data["embeddings"], query, top_k * 2)
        rrf_scores = {}
        for rank, (idx, _) in enumerate(tfidf_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(emb_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(idx, score) for idx, score in sorted_indices]
    else:
        # 默认双路 hybrid
        tfidf_results = retrieve_tfidf(data["vectorizer"], data["matrix"], query, top_k * 2)
        bm25_results = retrieve_bm25(data["bm25"], data["tokenized_corpus"], query, top_k * 2)
        rrf_scores = {}
        for rank, (idx, _) in enumerate(tfidf_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(idx, score) for idx, score in sorted_indices]

    formatted = []
    for idx, score in results:
        formatted.append({
            "text": chunks[idx],
            "source": metas[idx]["source"],
            "score": score,
        })

    return formatted


def build_prompt(query: str, contexts: list[dict]) -> str:
    ctx_lines = []
    for i, ctx in enumerate(contexts, 1):
        ctx_lines.append(
            f"[{i}] (来源: {ctx['source']}, 相似度: {ctx['score']:.3f})\n{ctx['text']}"
        )

    ctx_block = "\n\n---\n\n".join(ctx_lines)

    prompt = f"""你是一个专业的健身教练助手。请基于以下参考资料回答问题。
如果参考资料不足以回答，请如实说"资料中没有涉及这个方面"，不要编造。

参考资料：
{ctx_block}

问题：{query}

回答要求：
1. 直接回答问题，简洁清晰
2. 每个关键观点末尾标注来源编号，如 [1][2]
3. 如果引用了具体数值或方法，确保来自参考资料
4. 最后列出本次回答引用的来源文件

回答："""
    return prompt


def answer_openai(prompt: str) -> str:
    from openai import OpenAI
    import httpx

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return "❌ 请设置 OPENAI_API_KEY 环境变量才能使用 LLM 回答。"

    http_client = httpx.Client(verify=True)
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def answer_openai_stream(prompt: str):
    """流式生成回答，逐个 token 产出"""
    from openai import OpenAI
    import httpx

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        yield "❌ 请设置 OPENAI_API_KEY"
        return

    http_client = httpx.Client(verify=True)
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


def show_context(contexts: list[dict]):
    print("\n📎 检索到的参考资料:")
    print("-" * 60)
    for i, ctx in enumerate(contexts, 1):
        preview = ctx["text"][:120].replace("\n", " ")
        print(f"  [{i}] {ctx['source']} (相似度: {ctx['score']:.3f})")
        print(f"      {preview}…")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="健身RAG — 问答")
    parser.add_argument("question", nargs="*", help="你的问题")
    parser.add_argument("--llm", action="store_true", help="使用 LLM 生成回答")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"检索结果数 (默认 {TOP_K})")
    parser.add_argument("--retriever", choices=["tfidf", "bm25", "embedding", "hybrid", "hybrid3"], default="hybrid",
                        help="检索策略: tfidf/bm25/embedding/hybrid(双路)/hybrid3(三路)")
    parser.add_argument("--no-context", action="store_true", help="不显示检索上下文")
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else input("💪 你的问题: ")
    if not question.strip():
        print("❌ 问题不能为空")
        return

    print(f"\n🔍 检索 ({args.retriever}): {question}")
    results = retrieve(question, args.top_k, retriever=args.retriever)

    if results is None:
        return

    if not args.no_context:
        show_context(results)

    if args.llm:
        print("\n🤖 生成回答中…")
        prompt = build_prompt(question, results)
        answer = answer_openai(prompt)
        print("\n" + "=" * 60)
        print("💬 回答:")
        print(answer)
        print("=" * 60)
    else:
        best = results[0]
        print(f"\n📄 最佳匹配 ({best['source']}, 相似度: {best['score']:.3f}):")
        print(best["text"][:500])


if __name__ == "__main__":
    main()
