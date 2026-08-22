#!/bin/bash
# ============================================================
# 轮机智脑 · Render 容器启动脚本
#
# 顺序：
#   1) 启动视觉诊断 Agent (7864) — /ai/* 上游
#   2) 启动 Gradio 演示 (7861)    — /demo/* 上游
#   3) 等两者就绪 → FastAPI 网关 ($PORT) 前台运行
#
# FastAPI 网关用 httpx 反代 7861/7864（SSE 带心跳保活）。
# Render 注入 PORT（默认 10000），api_server 读 PORT env 绑定。
# ============================================================
set -e

PORT="${PORT:-10000}"
echo ">>> [boot] PORT=$PORT (Render 会注入)"

cd /app/app
mkdir -p data history_data output

echo ">>> [boot] 启动视觉诊断 Agent (7864) — /ai/* 上游"
python3 agent_vision_monitor.py > /tmp/agent.log 2>&1 &
AGENT_PID=$!
echo "    Agent PID=$AGENT_PID"

echo ">>> [boot] 启动 Gradio 演示 (7861) — /demo/* 上游"
python3 "visualizer_new_美化版.py" > /tmp/gradio.log 2>&1 &
GRADIO_PID=$!
echo "    Gradio PID=$GRADIO_PID"

# 等 7861 就绪（免费层冷启动较慢，放宽到 150s）
echo ">>> [boot] 等 7861 Gradio..."
for i in $(seq 1 150); do
  if curl -s -o /dev/null --max-time 2 http://127.0.0.1:7861/ ; then
    echo "    ✓ 7861 就绪 (${i}s)"
    break
  fi
  if [ "$i" -eq 150 ]; then
    echo "    ⚠ 7861 150s 内未就绪，tail 日志："
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
  if [ "$i" -eq 30 ]; then
    echo "    ⚠ 7864 30s 内未就绪（网关照常启动，/ai 暂不可用），tail 日志："
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
echo "    /history    → 历史数据浏览"
echo "================================================================="
echo ""

export PORT
exec python3 api_server.py
