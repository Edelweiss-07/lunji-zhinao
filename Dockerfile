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
COPY public/ ./public/
COPY serve.py ./

# ---------- Python 依赖 ----------
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ---------- 纯静态托管 ----------
# 站点资源已在 public/，serve.py 用 FastAPI StaticFiles 直接托管，无需 Gradio/Playwright 等重依赖。
# Render 注入 PORT 环境变量，uvicorn 绑定 0.0.0.0:$PORT。
EXPOSE 8080

CMD ["sh", "-c", "uvicorn serve:app --host 0.0.0.0 --port ${PORT:-8080}"]