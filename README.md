# 轮机智脑 · 智能体视觉诊断（Hugging Face Spaces 版）

船舶发动机智能监测演示：落地页（含 3D 鱼）→ 轮机智脑 Gradio 演示 → 冷却系统视觉诊断（DSR1 多模态 + KB 校验 + RAG）。
本仓库可直接部署到 Hugging Face Spaces，获得一个**公开链接**，任何人（评委/同学）打开即可访问。

## 架构（单容器，nginx 反代）

```
外网用户
   │  https://你的空间.hf.space   (HF 注入 $PORT，默认 7860)
   ▼
nginx（唯一大门）
   ├─ /demo/*  ──► Gradio 演示服务 (7861)        visualizer_new_美化版.py
   ├─ /ai/*    ──► 视觉诊断智能体 (7864)          agent_vision_monitor.py
   └─ /*       ──► FastAPI 网关 (7862)            api_server.py
                 （落地页 / 静态资源 / 历史 / 3D 鱼）
```

所有前端请求都走同一域名 + 路径前缀，因此浏览器代码里**没有写死 127.0.0.1**，公网可访问。

## 部署步骤（你来做，约 5 分钟）

1. 注册免费 HF 账号：https://huggingface.co/join
2. 新建 Space：https://huggingface.co/new-space
   - 填 Space 名称（如 `marine-engine-ai`）
   - **SDK 选 `Docker`**
   - 可见性选 **Public**（别人才能访问）
3. 把本目录（`轮机智脑-hf/`）下**所有文件**上传到 Space 仓库：
   - 方式 A（推荐）：`git clone` 仓库后把文件放进去 `git add -A && git commit && git push`
   - 方式 B：在 Space 网页直接拖拽上传 `Dockerfile / requirements.txt / start.sh / nginx.conf.template / app/`
4. 等 HF 自动构建（首次会下载依赖 + Playwright chromium，约 3–8 分钟）。构建完页面顶部显示 **Running** 即成功。
5. 公开地址即 `https://你的用户名-空间名.hf.space`，发给评委即可。

## 可选：学校 API 密钥（Secret）

代码默认用写在文件里的学校 API 地址/密钥。若想改用 Space Secret（不把密钥暴露进公开仓库），
在 Space **Settings → Secrets** 添加：

- `SCHOOL_API_BASE`  （默认 `https://chat.cqjtu.edu.cn/ds/api/v1`）
- `SCHOOL_API_KEY_LLM`
- `SCHOOL_API_KEY_DSR1`

不填则使用代码内默认值（你说学校 API 外网可用，故可不填）。

## 已知事项 / 注意

- **数据仍为模拟**：传感器由 `marine-sensor-simulator` 生成，比赛演示请标注为 demo，非真实船舶数据。
- **Gradio 版本**：本地开发用 6.x，HF 装公开最新版。若 Gradio 接口有差异导致演示页异常，需在 `requirements.txt` 锁定版本后再部署。
- **告警推送（Server酱，/alert）**：HF 上未运行 `alert_bridge.py`，该推送功能不可用，不影响主流程（视觉诊断、落地页、演示均正常）。
- **截图显示**：诊断页截图路径已改为可移植（相对 `SCREENSHOT_DIR`），每次检测后保留最近 50 张。
- **视觉诊断依赖学校 API**：若学校 API 在外网偶发不可达，诊断结论会缺失（页面会显示“未连接”），落地页与演示不受影响。
