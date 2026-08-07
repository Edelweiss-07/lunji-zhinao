FROM python:3.11-slim

# ---------- 基础工具 + Nginx + 中文字体 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fontconfig \
    tzdata \
    nginx \
    gettext-base \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv > /dev/null

ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

WORKDIR /app

# ---------- Python 依赖 ----------
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ---------- Playwright（agent_vision_monitor 截图需要）----------
RUN python3 -m playwright install chromium --with-deps

# ---------- 应用代码 ----------
COPY app/agent_vision_monitor.py ./
COPY app/api_server.py ./
COPY app/visualizer_new_美化版.py ./
COPY app/prompts.py ./
COPY app/kb_loader.py ./
COPY app/static/ ./static/
COPY app/kb_data/ ./kb_data/
COPY app/history_data/ ./history_data/

# ---------- 容器配置 ----------
COPY start.sh ./
RUN chmod +x start.sh

# ---------- 完整服务栈 ----------
# start.sh：Nginx → /demo/*(Gradio 7861) + /ai/*(Agent 7864) + /*(FastAPI 静态)
# Render 注入 PORT 环境变量
EXPOSE 8080

CMD ["sh", "/app/start.sh"]