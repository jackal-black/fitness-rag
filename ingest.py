"""
Fitness RAG — 文档索引工具 (TF-IDF + BM25)
用法: python ingest.py data/*.md
"""

import argparse
import glob
import os
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_DIR = "index"
INDEX_FILE = "fitness_index.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def read_file(path: str) -> tuple[str, str]:
    ext = Path(path).suffix.lower()
    name = Path(path).name

    if ext == ".md":
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), name

    elif ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise RuntimeError("请安装 pypdf: pip install pypdf")
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, name

    else:
        raise ValueError(f"不支持的文件类型: {ext}")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            cut = max(
                chunk.rfind("。"),
                chunk.rfind("\n\n"),
                chunk.rfind(". "),
                chunk.rfind("\n"),
            )
            if cut > chunk_size // 2:
                end = start + cut + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 20]


def tokenize_chinese(text: str) -> list[str]:
    """中文分词：字符级 2-gram + 保留英文单词"""
    tokens = []
    # 提取英文单词
    for word in re.findall(r"[a-zA-Z_]+", text):
        if len(word) > 1:
            tokens.append(word.lower())
    # 中文 2-gram（去掉标点空格）
    clean = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(len(clean) - 1):
        tokens.append(clean[i:i+2])
    return tokens


def main():
    parser = argparse.ArgumentParser(description="健身RAG — 索引文档 (TF-IDF + BM25)")
    parser.add_argument("files", nargs="+", help="要索引的文件")
    parser.add_argument("--reset", action="store_true", help="清空已有索引")
    args = parser.parse_args()

    files = []
    for pattern in args.files:
        files.extend(glob.glob(pattern, recursive=True))
    files = sorted(set(files))
    if not files:
        print("❌ 没有找到匹配的文件")
        return

    print(f"📄 发现 {len(files)} 个文件")

    all_chunks = []
    all_metas = []

    for fpath in files:
        print(f"  📖 {Path(fpath).name}")
        content, source = read_file(fpath)
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metas.append({"source": source})

    if not all_chunks:
        print("❌ 没有提取到有效文本")
        return

    print(f"🧩 共 {len(all_chunks)} 个文本块")

    # ── 1. TF-IDF ──
    print("  🔧 训练 TF-IDF…")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        analyzer="char",
        ngram_range=(2, 4),
    )
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    # ── 2. BM25 ──
    print("  🔧 训练 BM25…")
    tokenized_corpus = [tokenize_chinese(c) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # ── 保存（基于脚本所在目录） ──
    base_dir = Path(__file__).parent
    index_dir = base_dir / INDEX_DIR
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / INDEX_FILE

    with open(index_path, "wb") as f:
        pickle.dump({
            "vectorizer": vectorizer,
            "matrix": tfidf_matrix,
            "bm25": bm25,
            "tokenized_corpus": tokenized_corpus,
            "chunks": all_chunks,
            "metadatas": all_metas,
        }, f)

    print(f"✅ 完成！索引 {len(all_chunks)} 个文本块，保存至 {index_path}")
    print(f"   TF-IDF 词表: {len(vectorizer.get_feature_names_out())} | BM25 已就绪")


if __name__ == "__main__":
    main()
