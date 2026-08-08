# ⚓ 轮机智脑 · 远洋船舶轮机智能运维中枢

面向 MAN B&W 12K98ME-C7 大型低速二冲程柴油机的智能运维平台：数字孪生 + 大语言模型 + 实时数据驱动的全生命周期健康管理与智能决策闭环。

## 仓库结构

```
lunji-zhinao/
├── render.yaml          # Render Blueprint（一键部署两个服务）
├── site/                # 静态门户着陆页 → Render Static Site
│   ├── index.html       # 海洋主题着陆页（Three.js 3D 游鱼 / GSAP 动效）
│   ├── images/ libs/ fish_data.js *.mp4
└── server/              # 合并 Web 服务 → Render Web Service（单端口）
    ├── app.py           # FastAPI 装配入口（启动命令：python app.py）
    ├── visualizer_main.py   # Gradio 主界面（总览/监控/视觉识别/数据历史）
    ├── visualizer_core.py   # 核心逻辑（api_server 依赖）
    ├── api_server.py        # 历史数据 & AI 对话 API
    ├── kb_loader.py / prompts.py / kb_data/   # 知识库 RAG
    ├── history_data/        # 历史诊断记录 JSON
    └── static/              # portal / history / 监控页面
```

## 云端路由

| URL | 内容 |
|---|---|
| `https://lunji-zhinao-portal.onrender.com/` | 着陆页（静态站） |
| `https://lunji-zhinao-server.onrender.com/` | Gradio 外壳（portal） |
| `https://lunji-zhinao-server.onrender.com/app/` | Gradio 主应用 |
| `https://lunji-zhinao-server.onrender.com/history` | 历史数据可视化 |
| `https://lunji-zhinao-server.onrender.com/api/list` | 历史数据 API |

## 本地运行

```bash
cd server
pip install -r requirements.txt
# 配置密钥（不配置也能跑，AI 功能会提示未配置）
set SCHOOL_API_KEY=sk-xxx
set DSV4_KEY=sk-xxx
set DSR1_API_KEY=sk-xxx
python app.py        # http://localhost:10000/
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `SCHOOL_API_KEY` / `DSV4_KEY` / `DSR1_API_KEY` | LLM API 密钥（必填，在 Render 后台设置） |
| `SCHOOL_API_BASE` / `DSR1_API_BASE` | API 端点（默认学校端点） |
| `OLLAMA_BASE` | 本地 Ollama 地址（云端留空） |
| `PORT` | 服务端口（Render 自动注入，本地默认 10000） |

## Render 部署

1. 推送本仓库到 GitHub
2. Render Dashboard → **New → Blueprint** → 选择本仓库
3. 按提示填写 3 个 API 密钥（`sync: false` 占位）
4. Apply，等待构建完成

免费套餐说明：Web 服务 15 分钟无流量会休眠，冷启动约 30-60 秒；静态门户不休眠。
