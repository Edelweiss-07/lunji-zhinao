"""
轮机智脑 · 纯静态托管服务

把 public/ 目录（index.html + libs/ + images/ + fish_data.js）作为静态站点托管。
Render 注入 PORT 环境变量，绑定 0.0.0.0:$PORT。
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

app = FastAPI(title="轮机智脑 | 远洋船舶轮机智能运维中枢", docs_url=None, redoc_url=None)

# 挂载 public/ 为站点根：
#   /                 -> index.html
#   /fish_data.js     -> 3D 鱼模型内联数据（38MB）
#   /libs/*           -> Three.js / GLTFLoader
#   /images/*         -> 页面图片资源
app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="site")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
