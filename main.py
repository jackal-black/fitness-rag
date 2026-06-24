"""
Fitness RAG — FastAPI 服务 (Hybrid Search + 多轮对话)
启动: uvicorn main:app --reload
"""

import os
import time
import uuid
from pathlib import Path
from typing import Optional

# 加载 .env 文件（如果存在）
from dotenv import load_dotenv
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from query import retrieve, build_prompt, answer_openai, answer_openai_stream

app = FastAPI(title="Fitness RAG", version="1.2.0")

# ── 多轮对话 session 管理 ──
# 格式: { session_id: [{"role": "user"/"assistant", "content": str}, ...] }
_sessions: dict[str, list[dict]] = {}
_SESSION_TTL = 3600  # 1 小时过期
_MAX_HISTORY = 6     # 保留最近 3 轮（6 条消息）


def _clean_expired_sessions():
    """清理过期 session"""
    now = time.time()
    expired = [sid for sid, (_, ts) in _sessions.items()
               if isinstance(_sessions[sid], tuple) and now - ts > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]


def _get_or_create_session(session_id: str | None) -> tuple[str, list[dict]]:
    """获取或创建会话，返回 (session_id, history)"""
    if session_id and session_id in _sessions:
        history, _ = _sessions[session_id]
        return session_id, history

    sid = session_id or str(uuid.uuid4())[:8]
    _sessions[sid] = ([], time.time())
    return sid, []


def _update_session(session_id: str, history: list[dict]):
    """更新会话并更新时间戳"""
    _clean_expired_sessions()
    _sessions[session_id] = (history[-_MAX_HISTORY:], time.time())


def build_chat_prompt(question: str, contexts: list[dict], history: list[dict]) -> str:
    """带对话历史的 prompt 构建"""
    # 构建历史文本
    history_lines = []
    for msg in history[-4:]:  # 最近 2 轮
        role = "用户" if msg["role"] == "user" else "助手"
        history_lines.append(f"{role}: {msg['content']}")

    history_block = "\n".join(history_lines) if history_lines else "（无对话历史）"

    # 构建上下文引用
    ctx_lines = []
    for i, ctx in enumerate(contexts, 1):
        ctx_lines.append(
            f"[{i}] (来源: {ctx['source']}, 相似度: {ctx['score']:.3f})\n{ctx['text']}"
        )
    ctx_block = "\n\n---\n\n".join(ctx_lines)

    prompt = f"""你是一个专业的健身教练助手。请基于以下参考资料和对话历史回答问题。

对话历史：
{history_block}

参考资料：
{ctx_block}

当前问题：{question}

回答要求：
1. 直接回答问题，简洁清晰
2. 每个关键观点末尾标注来源编号，如 [1][2]
3. 如果用户追问上轮话题，先参考对话历史理解上下文
4. 如果参考资料不足以回答，请如实说"资料中没有涉及这个方面"
5. 最后列出本次回答引用的来源文件

回答："""
    return prompt


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    llm: Optional[bool] = False
    retriever: str = "hybrid"  # tfidf / bm25 / hybrid


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: int = 5
    retriever: str = "hybrid"


class SourceItem(BaseModel):
    source: str
    similarity: float
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    sources: list[SourceItem]


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[SourceItem]


@app.get("/")
def root():
    return {"message": "Fitness RAG API", "docs": "/docs", "version": "1.2.0"}


@app.get("/health")
def health():
    api_key = bool(os.getenv("OPENAI_API_KEY"))
    idx_path = Path(__file__).parent / "index" / "fitness_index.pkl"
    ok = idx_path.exists()
    return {
        "status": "ok" if ok else "no index",
        "index_ready": ok,
        "llm_ready": api_key or False,
        "llm_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "active_sessions": len(_sessions),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    results = retrieve(req.question, req.top_k, retriever=req.retriever)
    if results is None:
        raise HTTPException(status_code=400, detail="知识库为空。请先运行 python ingest.py data/*.md")

    sources = [
        SourceItem(source=r["source"], similarity=round(r["score"], 4), snippet=r["text"][:200])
        for r in results
    ]

    answer = None
    if req.llm:
        prompt = build_prompt(req.question, results)
        answer = answer_openai(prompt)

    return AskResponse(question=req.question, answer=answer, sources=sources)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """流式回答（SSE）"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    results = retrieve(req.question, req.top_k, retriever=req.retriever)
    if results is None:
        raise HTTPException(status_code=400, detail="知识库为空")

    prompt = build_prompt(req.question, results)

    def generate():
        for token in answer_openai_stream(prompt):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    多轮对话。传入 session_id 则延续对话，不传则新建。
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 检索
    results = retrieve(req.question, req.top_k, retriever=req.retriever)
    if results is None:
        raise HTTPException(status_code=400, detail="知识库为空。请先运行 python ingest.py data/*.md")

    sources = [
        SourceItem(source=r["source"], similarity=round(r["score"], 4), snippet=r["text"][:200])
        for r in results
    ]

    # 多轮对话
    session_id, history = _get_or_create_session(req.session_id)

    prompt = build_chat_prompt(req.question, results, history)
    answer = answer_openai(prompt)

    # 更新历史
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})
    _update_session(session_id, history)

    return ChatResponse(
        session_id=session_id,
        question=req.question,
        answer=answer,
        sources=sources,
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """删除会话"""
    if session_id in _sessions:
        del _sessions[session_id]
    return {"message": f"Session {session_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
