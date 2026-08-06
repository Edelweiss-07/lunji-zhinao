FROM python:3.11-slim

# ---------- 基础工具 + 中文字体 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    curl \
    ca-certificates \
    gettext-base \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv > /dev/null

# 让 Python 日志立即 flush（不缓存），Render Logs 页面才能实时看到 traceback
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ---------- 应用代码 ----------
COPY requirements.txt ./
COPY app/ ./
COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ---------- Python 依赖 ----------
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ---------- Playwright + Chromium + 系统依赖 ----------
# 用 install --with-deps 一次性让 Playwright 装齐所有系统库（libgtk-3-0, libgdk-pixbuf-2.0-0 等）
# 之前手动 apt-get 装的 libnss3 那批可能漏了 libgtk，导致 chromium 启动崩溃
RUN python3 -m playwright install --with-deps chromium

# HF 通过 $PORT 访问（默认 7860，HF 会自动覆盖该环境变量）
EXPOSE 7860
ENV PORT 7860

CMD ["/start.sh"]