#!/bin/bash
set -e

# HF 注入的端口（默认 7860），用 envsubst 写进 nginx 配置
export PORT="${PORT:-7860}"
envsubst '${PORT}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

echo ">>> 启动后端服务..."

# 1) 视觉诊断智能体（7864）：截图 + 调用学校 DSR1 + 暴露 /ai 接口
python3 /app/agent_vision_monitor.py &

# 2) Gradio 轮机智脑演示（7861，经 nginx /demo 暴露）
python3 /app/visualizer_new_美化版.py &

# 3) FastAPI 网关（7862，落地页 + 静态资源 + 历史）
python3 /app/api_server.py &

# 等网关就绪（最多 30s）
for i in $(seq 1 30); do
  if curl -s -o /dev/null http://127.0.0.1:7862/ ; then
    echo ">>> 网关就绪"
    break
  fi
  sleep 1
done

echo ">>> 启动 nginx 作为唯一对外大门（端口 ${PORT}）"
# 前台运行，保持容器存活
nginx -g 'daemon off;'
