#!/bin/bash
set -e

# ============================================================
# HF/Render 注入 PORT（默认 7860），envsubst 替换 nginx 配置里的 ${PORT}
# ============================================================
export PORT="${PORT:-7860}"
echo ">>> [1/6] PORT=${PORT}"
echo ">>> [1/6] nginx 模板 -> /etc/nginx/nginx.conf"
envsubst '${PORT}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf
echo ">>> [1/6] /etc/nginx/nginx.conf 生成完毕（内容如下，方便 Render Logs 排查）"
sed -n '1,80p' /etc/nginx/nginx.conf | grep -E "location|listen|proxy_pass" | head -30

# ============================================================
# 依次启动三个后端服务，每个都等就绪
# 这样 nginx 启动时 upstream 一定在监听，不会 502
# ============================================================
echo ">>> [2/6] 启动 Gradio 演示服务（7861）..."
python3 /app/visualizer_new_美化版.py > /tmp/gradio.log 2>&1 &
GRADIO_PID=$!
echo "    Gradio PID=$GRADIO_PID"

echo ">>> [3/6] 启动 FastAPI 网关（7862，落地页+静态资源+历史）..."
python3 /app/api_server.py > /tmp/api.log 2>&1 &
API_PID=$!
echo "    API PID=$API_PID"

# 等 7862 就绪（7861 也要先就绪，但 7861 Gradio 启动慢，单独等）
echo ">>> [4/6] 等待 7862 就绪..."
for i in $(seq 1 60); do
  if curl -s -o /dev/null --max-time 2 http://127.0.0.1:7862/ ; then
    echo "    ✓ 7862 已就绪 (用时 ${i}s)"
    break
  fi
  sleep 1
done

# 验证 7861 Gradio 也启动了（这步关键：之前 /demo/ 失效的根因之一是 7861 没起来）
echo ">>> [5/6] 验证 7861 Gradio 是否在监听..."
for i in $(seq 1 60); do
  if curl -s -o /dev/null --max-time 2 http://127.0.0.1:7861/ ; then
    echo "    ✓ 7861 Gradio 已就绪 (用时 ${i}s)"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "    ⚠ 7861 Gradio 60s 内未就绪！查看日志："
    tail -30 /tmp/gradio.log
  fi
  sleep 1
done

# 7864 必须在 7862 起来之后再启动（否则 capture_panel 第一次 fetch /static/cooling-system.html 会失败）
echo ">>> [6/6] 启动视觉诊断智能体（7864）..."
python3 /app/agent_vision_monitor.py > /tmp/agent.log 2>&1 &
AGENT_PID=$!
echo "    7864 PID=$AGENT_PID"

# 给 7864 几秒钟启动 HTTP server
sleep 3
if curl -s -o /dev/null --max-time 2 http://127.0.0.1:7864/health ; then
  echo "    ✓ 7864 已就绪"
else
  echo "    ⚠ 7864 启动后 /health 无响应（仍正常，monitor_loop 是异步的）"
fi

echo ""
echo "================================================================="
echo "  所有服务已启动："
echo "    7861 Gradio  (PID=$GRADIO_PID)"
echo "    7862 网关    (PID=$API_PID)"
echo "    7864 诊断    (PID=$AGENT_PID)"
echo "  nginx 即将启动在 :$PORT"
echo "================================================================="
echo ""

# 前台跑 nginx 保持容器存活
nginx -g 'daemon off;'