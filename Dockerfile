FROM python:3.11-slim

# ---------- 基础工具 + 中文字体 ----------
# （不再装 nginx / gettext-base：反代由 FastAPI 自己用 httpx 做，少一个进程少一堆坑）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fontconfig \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv > /dev/null

# 让 Python 日志立即 flush（不缓存），Render Logs 页面才能实时看到 traceback
ENV PYTHONUNBUFFERED=1
# 容器时区设中国（用户在中国，visualizer 用 datetime.now() 无时区，
# 不设的话 Render(UTC) 上的时间戳会比用户本地慢 8 小时）
ENV TZ=Asia/Shanghai

WORKDIR /app

# ---------- 应用代码 ----------
COPY requirements.txt ./
COPY app/ ./
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ---------- Python 依赖 ----------
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ---------- Playwright + Chromium + 系统依赖 ----------
# 用 install --with-deps 一次性让 Playwright 装齐所有系统库（libgtk-3-0, libgdk-pixbuf-2.0-0 等）
# 之前手动 apt-get 装的 libnss3 那批可能漏了 libgtk，导致 chromium 启动崩溃
RUN python3 -m playwright install --with-deps chromium

# Render 会注入 PORT（默认 10000）。api_server 读 PORT 后 uvicorn 直接绑它。
# 我们不再走 nginx，所以没有端口 redirect 问题，也不需要 envsubst。
EXPOSE 10000

CMD ["/start.sh"]