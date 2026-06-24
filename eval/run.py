"""
RAG 评估管线：Hit Rate@K / MRR / NDCG / Faithfulness

用法:
  python eval/run.py                          # 评估纯检索
  python eval/run.py --llm                    # 评估 LLM 回答忠实度（需配置 .env）
  python eval/run.py --html                   # 输出可视化报告
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from query import retrieve, build_prompt, answer_openai


def load_questions(path=None):
    base = Path(__file__).parent
    if path is None:
        path = base / "questions.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hit_rate(results, expected_sources, k):
    """top-k 中是否包含预期来源文件"""
    retrieved_sources = {r["source"] for r in results[:k]}
    return any(src in retrieved_sources for src in expected_sources)


def mrr(results, expected_sources):
    """第一个相关文档的倒数排名"""
    for rank, r in enumerate(results, 1):
        if r["source"] in expected_sources:
            return 1.0 / rank
    return 0.0


def ndcg(results, expected_sources, k):
    """归一化折损累计增益"""
    dcg = 0.0
    for i, r in enumerate(results[:k]):
        rel = 1.0 if r["source"] in expected_sources else 0.0
        if i == 0:
            dcg += rel
        else:
            dcg += rel / math.log2(i + 1)

    # IDCG: 理想情况下的 DCG
    ideal_rel = [1.0] * min(len(expected_sources), k) + [0.0] * max(0, k - len(expected_sources))
    idcg = 0.0
    for i, rel in enumerate(ideal_rel):
        if i == 0:
            idcg += rel
        else:
            idcg += rel / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def check_faithfulness(question, answer, contexts, llm_checker=True):
    """
    检查 LLM 回答的忠实度：
    - 回答中的关键主张是否能在参考文档中找到支持
    - 使用 LLM 自评（如果 llm_checker=True）
    """
    if not answer or answer.startswith("❌"):
        return 0.0

    if not llm_checker:
        return None  # 跳过

    ctx_block = "\n".join([f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(contexts[:3])])

    prompt = f"""判断以下回答是否忠实于给出的参考资料。
回答中的每个重要主张都应该能在参考资料中找到支持。
只回答一个数字：1（完全忠实）或 0（存在幻觉/编造）。

参考资料：
{ctx_block}

用户问题：{question}

AI回答：{answer}

评分（1 或 0）："""

    try:
        result = answer_openai(prompt)
        # 提取数字
        for word in result.strip().split():
            if word in ("1", "0"):
                return float(word)
        return None
    except Exception:
        return None


def run_retrieval_eval(questions, top_k=5, retriever="hybrid"):
    """检索评估"""
    results_detail = []

    total_hr = {k: 0.0 for k in [1, 3, 5]}
    total_mrr = 0.0
    total_ndcg = {k: 0.0 for k in [1, 3, 5]}

    for q in questions:
        qid = q["id"]
        exp = q["expected_sources"]
        query = q["question"]

        docs = retrieve(query, top_k=top_k, retriever=retriever)
        if docs is None:
            print(f"  ❌ {qid}: 索引为空")
            continue

        hr5 = hit_rate(docs, exp, 5)
        hr3 = hit_rate(docs, exp, 3)
        hr1 = hit_rate(docs, exp, 1)
        reciprocal = mrr(docs, exp)
        ndcg5 = ndcg(docs, exp, 5)
        ndcg3 = ndcg(docs, exp, 3)
        ndcg1 = ndcg(docs, exp, 1)

        total_hr[1] += hr1
        total_hr[3] += hr3
        total_hr[5] += hr5
        total_mrr += reciprocal
        total_ndcg[1] += ndcg1
        total_ndcg[3] += ndcg3
        total_ndcg[5] += ndcg5

        top_sources = [r["source"] for r in docs[:3]]
        hit = "✅" if hr5 else "❌"

        results_detail.append({
            "id": qid,
            "question": query,
            "hit": hr5,
            "mrr": round(reciprocal, 4),
            "ndcg@5": round(ndcg5, 4),
            "top_sources": top_sources,
            "expected": exp,
        })

        print(f"  {hit} {qid}: {query[:30]}... HR@5={hr5}, MRR={reciprocal:.3f}")

    n = len(questions)
    metrics = {
        "num_questions": n,
        "Hit Rate @1": round(total_hr[1] / n, 4),
        "Hit Rate @3": round(total_hr[3] / n, 4),
        "Hit Rate @5": round(total_hr[5] / n, 4),
        "MRR": round(total_mrr / n, 4),
        "NDCG @1": round(total_ndcg[1] / n, 4),
        "NDCG @3": round(total_ndcg[3] / n, 4),
        "NDCG @5": round(total_ndcg[5] / n, 4),
    }

    return metrics, results_detail


def run_faithfulness_eval(questions, sample_size=None):
    """LLM 忠实度评估（取子集，因为要调 LLM）"""
    if sample_size and sample_size < len(questions):
        import random
        sampled = random.sample(questions, sample_size)
    else:
        sampled = questions

    scores = []
    for q in sampled:
        query = q["question"]
        print(f"  🔍 {q['id']} 评估忠实度...")

        docs = retrieve(query, top_k=3)
        if not docs:
            continue

        prompt = build_prompt(query, docs)
        answer = answer_openai(prompt)
        score = check_faithfulness(query, answer, docs)

        if score is not None:
            scores.append(score)
            icon = "✅" if score == 1 else "❌"
            print(f"    {icon} Faithfulness: {score}")

    if scores:
        return round(sum(scores) / len(scores), 4)
    return None


def print_report(metrics, faithfulness=None):
    """打印评估报告"""
    print("\n" + "=" * 50)
    print("📊 RAG 评估报告")
    print("=" * 50)
    print(f"测试题数: {metrics['num_questions']}")
    print()
    print("检索指标:")
    for k in [1, 3, 5]:
        print(f"  Hit Rate @{k}:  {metrics[f'Hit Rate @{k}']:.2%}")
    print(f"  MRR:           {metrics['MRR']:.4f}")
    for k in [1, 3, 5]:
        print(f"  NDCG @{k}:      {metrics[f'NDCG @{k}']:.4f}")

    if faithfulness is not None:
        print(f"\nFaithfulness:   {faithfulness:.2%}")

    print("=" * 50)


def generate_html_report(metrics, details, faithfulness=None):
    """生成 HTML 报告"""
    rows = ""
    for d in details:
        hit_icon = "✅" if d["hit"] else "❌"
        sources = ", ".join(d["top_sources"])
        expected = ", ".join(d["expected"])
        rows += f"""
        <tr>
            <td>{d['id']}</td>
            <td>{d['question'][:40]}…</td>
            <td>{hit_icon}</td>
            <td>{d['mrr']:.3f}</td>
            <td>{d['ndcg@5']:.3f}</td>
            <td style="font-size:0.85em">{sources}</td>
            <td style="font-size:0.85em">{expected}</td>
        </tr>"""

    faithfulness_html = f"<p><strong>Faithfulness:</strong> {faithfulness:.2%}</p>" if faithfulness is not None else ""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>RAG 评估报告</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }}
h1, h2 {{ color: #58a6ff; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #30363d; padding: 8px 12px; text-align: left; }}
th {{ background: #161b22; color: #8b949e; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 20px 0; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
.card .value {{ font-size: 1.8em; font-weight: bold; color: #58a6ff; }}
.card .label {{ font-size: 0.85em; color: #8b949e; }}
.positive {{ color: #3fb950; }}
.negative {{ color: #f85149; }}
</style></head>
<body>
<h1>📊 RAG 评估报告</h1>
<p><strong>测试题数:</strong> {metrics['num_questions']}</p>

<h2>整体指标</h2>
<div class="metrics">
    <div class="card"><div class="value">{metrics['Hit Rate @5']:.1%}</div><div class="label">Hit Rate @5</div></div>
    <div class="card"><div class="value">{metrics['Hit Rate @3']:.1%}</div><div class="label">Hit Rate @3</div></div>
    <div class="card"><div class="value">{metrics['Hit Rate @1']:.1%}</div><div class="label">Hit Rate @1</div></div>
    <div class="card"><div class="value">{metrics['MRR']:.4f}</div><div class="label">MRR</div></div>
    <div class="card"><div class="value">{metrics['NDCG @5']:.4f}</div><div class="label">NDCG @5</div></div>
    <div class="card"><div class="value">{metrics['NDCG @3']:.4f}</div><div class="label">NDCG @3</div></div>
</div>
{faithfulness_html}

<h2>逐题明细</h2>
<table>
<thead><tr><th>ID</th><th>问题</th><th>Hit@5</th><th>MRR</th><th>NDCG@5</th><th>Top来源</th><th>期望来源</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body></html>"""

    report_path = Path(__file__).parent / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="RAG Eval")
    parser.add_argument("--llm", action="store_true", help="评估 LLM 忠实度")
    parser.add_argument("--html", action="store_true", help="生成 HTML 报告")
    parser.add_argument("--faithfulness-samples", type=int, default=5, help="忠实度评估采样数")
    parser.add_argument("--retriever", choices=["tfidf", "bm25", "hybrid"], default="hybrid",
                        help="检索策略")
    args = parser.parse_args()

    print("📋 加载测试集...")
    questions = load_questions()
    print(f"   共 {len(questions)} 道题\n")

    print("🔍 运行检索评估...")
    metrics, details = run_retrieval_eval(questions, retriever=args.retriever)

    faithfulness = None
    if args.llm:
        print(f"\n🤖 运行忠实度评估 (采样 {args.faithfulness_samples} 道)...")
        faithfulness = run_faithfulness_eval(questions, args.faithfulness_samples)

    print_report(metrics, faithfulness)

    if args.html:
        report_path = generate_html_report(metrics, details, faithfulness)
        print(f"\n📄 HTML 报告已生成: {report_path}")


if __name__ == "__main__":
    main()
