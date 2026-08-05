FROM python:3.11-slim

# ---------- 系统依赖 ----------
# nginx：作为唯一对外大门，反向代理三个内部服务
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Playwright (chromium) 运行所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 libpango-1.0-0 libcairo2 libatspi2.0-0 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 应用代码 ----------
COPY requirements.txt ./
COPY app/ ./
COPY nginx.conf.template /etc/nginx/nginx.conf.template
COPY start.sh /start.sh
RUN chmod +x /start.sh

# ---------- Python 依赖 ----------
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# ---------- 安装浏览器（视觉诊断截图用） ----------
RUN python3 -m playwright install chromium

# HF 通过 $PORT 访问（默认 7860，HF 会自动覆盖该环境变量）
EXPOSE 7860
ENV PORT 7860

CMD ["/start.sh"]
