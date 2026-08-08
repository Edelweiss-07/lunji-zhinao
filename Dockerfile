FROM python:3.11-slim
# 轮机智脑 · 公网落地页壳(隧道模式)
# Render 只托管静态落地页，真实后端跑在用户本地，
# 通过 cloudflared 隧道暴露，落地页用 ?b1 ?b2 ?b4 连接。
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY serve.py ./
COPY public/ ./public/
EXPOSE 8080
CMD ["sh", "-c", "python3 serve.py"]
