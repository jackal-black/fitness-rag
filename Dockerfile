# 使用 python 官方镜像（指定 sha256 摘要避免 Docker Desktop 4.43 bug）
FROM python:3.12-slim-bookworm

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制代码
COPY . .

# 建索引（如果有 data 目录且有文件）
RUN if [ -d data ] && [ "$(ls -A data 2>/dev/null)" ]; then \
        python ingest.py data/* --reset; \
    fi

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
