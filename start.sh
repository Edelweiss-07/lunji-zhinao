#!/bin/bash
# ============================================================
# Render 容器启动脚本（干掉 nginx 后的极简版）
#
# 顺序：
#   1) 启动 Gradio (7861)   — /demo/*  上游
#   2) 启动 Agent  (7864)   — /ai/*    上游
#   3) 等 7861、7864 都就绪 → 启动 FastAPI (7862 -> $PORT) 前台跑
#
# FastAPI 内部用 httpx 反代到 7861/7864，单进程搞定一切。
# Render 注入 PORT=10000，api_server 读 PORT env 后 uvicorn 绑它。
# ============================================================
set -e

PORT="${PORT:-10000}"
echo ">>> [boot] PORT=$PORT (Render 会注入)"
echo ">>> [boot] 启动 Gradio (7861) — 演示页上游"

cd /app
python3 /app/visualizer_new_美化版.py > /tmp/gradio.log 2>&1 &
GRADIO_PID=$!
echo "    Gradio PID=$GRADIO_PID"

echo ">>> [boot] 启动视觉诊断 (7864) — /ai/* 上游"
python3 /app/agent_vision_monitor.py > /tmp/agent.log 2>&1 &
AGENT_PID=$!
echo "    Agent PID=$AGENT_PID"

# 等 7861 就绪
echo ">>> [boot] 等 7861 Gradio..."
for i in $(seq 1 90); do
  if curl -s -o /dev/null --max-time 2 http://127.0.0.1:7861/ ; then
    echo "    ✓ 7861 就绪 (${i}s)"
    break
  fi
  if [ $i -eq 90 ]; then
    echo "    ⚠ 7861 90s 内未就绪，tail 日志："
    tail -40 /tmp/gradio.log
  fi
  sleep 1
done

# 等 7864 就绪
echo ">>> [boot] 等 7864 Agent..."
for i in $(seq 1 30); do
  if curl -s -o /dev/null --max-time 2 http://127.0.0.1:7864/health ; then
    echo "    ✓ 7864 就绪 (${i}s)"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "    ⚠ 7864 30s 内未就绪，tail 日志："
    tail -40 /tmp/agent.log
  fi
  sleep 1
done

echo ""
echo "================================================================="
echo "  FastAPI 网关（反代 + 落地页 + 历史）即将启动在 0.0.0.0:$PORT"
echo "    /           → 落地页 (static/index.html)"
echo "    /demo/*     → Gradio 7861"
echo "    /ai/*       → Agent  7864 (剥前缀)"
echo "    /ai         → 静态 cooling-diagnosis.html"
echo "================================================================="
echo ""

export PORT
exec python3 /app/api_server.py