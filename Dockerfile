FROM python:3.11-slim
# 轮机智脑 · 全栈部署
# 单容器三服务：Gradio 演示(7861) + 视觉诊断(7864) + FastAPI 网关($PORT)
# 网关反代 /demo/* 和 /ai/*，静态落地页 + 历史数据 + REST API 同源提供。
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY start.sh ./
RUN chmod +x start.sh

EXPOSE 8080
CMD ["sh", "/app/start.sh"]
