# -*- coding: utf-8 -*-
"""
轮机智脑 — 云端合并服务入口（Render 单端口部署）

路由总表：
  /            → portal.html（Gradio 外壳：标签页 + iframe）
  /app/        → Gradio 主应用（总览/监控/视觉识别/数据历史）
  /history     → 历史数据可视化页
  /api/*       → 历史数据 & AI 对话 API（来自 api_server）
  /static/*    → 静态资源（portal/history 页面及配图）

本地调试：python app.py  →  http://localhost:10000/
Render ：自动注入 PORT 环境变量
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import uvicorn

from api_server import app                # FastAPI：API 路由 + /static + / + /history
from visualizer_main import (             # Gradio 主界面（美化版）+ 样式资源
    create_ui, GRADIO_CSS, GRADIO_THEME, NAV_BRIDGE_JS,
)

print("🚢 轮机智脑云端服务启动中…")
print("   [1/2] 构建 Gradio 主界面")
demo = create_ui()
demo.queue(default_concurrency_limit=3)

print("   [2/2] 挂载 Gradio 到 /app")
# Gradio 6：css/js/theme 必须在挂载时显式传入（不会继承 Blocks 构造函数）
app = gr.mount_gradio_app(
    app, demo, path="/app",
    css=GRADIO_CSS,
    js=NAV_BRIDGE_JS,
    theme=GRADIO_THEME,
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("=" * 60)
    print(f"✅ 服务就绪: http://localhost:{port}/")
    print(f"   Gradio 主界面: http://localhost:{port}/app/")
    print(f"   历史数据:      http://localhost:{port}/history")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
