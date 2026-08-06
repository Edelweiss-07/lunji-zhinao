"""完整 E2E 验证 — 起 mock 7861 Gradio / mock 7864 Agent,跑真实 api_server,验 HTTP + WS 反代"""
import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path

DEPLOY = Path(r"D:\船舶智能体\轮机智脑-hf")
sys.path.insert(0, str(DEPLOY / "app"))

import httpx
import websockets

# 装 aiohttp 给 mock 用
try:
    from aiohttp import web
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "aiohttp"], check=True)
    from aiohttp import web


# ============================================================
# Mock 上游服务
# ============================================================
async def run_mock_7861():
    """aiohttp mock: HTTP / 返 Gradio HTML; WS /queue/join 接受+回显"""
    gradio_html = b"""<!DOCTYPE html>
<html><head><title>Mock Gradio</title></head>
<body>Mock Gradio at 7861</body></html>"""

    async def http_handler(request):
        return web.Response(body=gradio_html, content_type="text/html")

    async def ws_handler(request):
        ws = web.WebSocketResponse(protocols=["gradio-protocol"])
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                await ws.send_json({"echo": msg.data, "from": "mock_7861"})
        return ws

    # SSE 端点（Gradio 6.x 实际用 SSE，不是 WS）
    async def sse_handler(request):
        # Gradio 6.x 客户端用 EventSource 连这个端点接收流式数据
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        for i in range(3):
            await asyncio.sleep(0.3)
            await resp.write(b"data: {\"msg\": \"hello\"}\n\n")
        await resp.write(b"data: [DONE]\n\n")
        return resp

    # 单个 dispatch 函数手动按 path 分发（aiohttp 路由 var path 不可靠）
    async def dispatch(request):
        path = request.path
        if path == "/queue/join":
            return await ws_handler(request)
        elif path == "/queue/data":
            return await sse_handler(request)
        else:
            return web.Response(body=gradio_html, content_type="text/html")

    app = web.Application()
    # catch-all 路由
    app.router.add_get("/{path:.*}", dispatch)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 7861).start()
    print("[mock 7861] up on 127.0.0.1:7861 (HTTP+WS)")
    await asyncio.Future()


async def run_mock_7864():
    """BaseHTTPRequestHandler mock: /health + /latest"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse
    import threading

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                body = json.dumps({"status": "ok", "port": 7864, "diagnosing": False}).encode()
            elif path == "/latest":
                body = json.dumps({"id": "mock-001", "time": "2026-08-06 14:00:00"}).encode()
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a): pass

    httpd = HTTPServer(("127.0.0.1", 7864), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print("[mock 7864] up on 127.0.0.1:7864")
    await asyncio.Future()


async def run_api_server():
    """子进程跑真实 api_server.py,绑 PORT=7862"""
    env = {**os.environ, "PORT": "7862", "PYTHONPATH": str(DEPLOY / "app")}
    proc = subprocess.Popen(
        [sys.executable, str(DEPLOY / "app" / "api_server.py")],
        env=env, cwd=str(DEPLOY / "app"),
        stdout=sys.stdout, stderr=subprocess.STDOUT,
    )
    for i in range(30):
        try:
            async with httpx.AsyncClient() as c:
                r = await c.get("http://127.0.0.1:7862/", timeout=2)
                if r.status_code == 200:
                    print(f"[api_server] ready in {i}s")
                    return proc
        except Exception:
            pass
        await asyncio.sleep(1)
    raise RuntimeError("api_server 30s 内未就绪")


async def test_http():
    print("\n========== HTTP 反代测试 ==========")
    async with httpx.AsyncClient(timeout=10) as c:
        # / 落地页
        r = await c.get("http://127.0.0.1:7862/")
        print(f"  GET /            -> {r.status_code}  size={len(r.content)}")
        assert r.status_code == 200, f"/ 返回 {r.status_code}"

        # /demo/ 反代到 mock 7861
        r = await c.get("http://127.0.0.1:7862/demo/")
        ok = "Mock Gradio" in r.text
        print(f"  GET /demo/       -> {r.status_code}  size={len(r.content)}  has 'Mock Gradio'={ok}")
        assert r.status_code == 200 and ok

        # /ai/health 反代到 mock 7864 (剥前缀)
        r = await c.get("http://127.0.0.1:7862/ai/health")
        print(f"  GET /ai/health   -> {r.status_code}  body={r.text[:80]}")
        assert r.status_code == 200 and r.json().get("status") == "ok"

        # /ai/latest 反代
        r = await c.get("http://127.0.0.1:7862/ai/latest")
        print(f"  GET /ai/latest   -> {r.status_code}  body={r.text[:80]}")
        assert r.status_code == 200 and r.json().get("id") == "mock-001"

        # /demo 301 重定向
        r = await c.get("http://127.0.0.1:7862/demo", follow_redirects=False)
        print(f"  GET /demo        -> {r.status_code}  location={r.headers.get('location')}")
        assert r.status_code == 301 and r.headers.get("location") == "/demo/"

    print("✅ HTTP 全部通过")


async def test_sse():
    print("\n========== SSE 流式反代测试（Gradio 6.x 实际通信方式）==========")
    url = "http://127.0.0.1:7862/demo/queue/data"
    print(f"GET {url} 期望: text/event-stream + 流式 data 事件")
    received_events = []
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            async with c.stream("GET", url) as r:
                print(f"  status={r.status_code}  content-type={r.headers.get('content-type')}")
                assert r.status_code == 200
                assert "event-stream" in r.headers.get("content-type", ""), f"content-type={r.headers.get('content-type')}"
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        received_events.append(data)
                        print(f"  event: {data}")
                        if data == "[DONE]":
                            # 立即 break,不让 httpx context manager 触发 aread
                            # 用 aclose() 跳过未读完的 body
                            await r.aclose()
                            break
    except httpx.RemoteProtocolError as e:
        # SSE 流正常结束时 chunked 编码被截断,httpx context 关闭时报错
        # 只要收到事件就 OK
        print(f"  (stream closed by upstream: {type(e).__name__})")
    assert len(received_events) >= 3, f"期望至少 3 条 event, 实际 {len(received_events)}"
    print(f"✅ SSE 流式反代通过,收到 {len(received_events)} 条 event")


async def main():
    t1 = asyncio.create_task(run_mock_7861())
    t2 = asyncio.create_task(run_mock_7864())
    await asyncio.sleep(2)  # 等 mock
    api_proc = await run_api_server()
    try:
        await test_http()
        await test_sse()
        print("\n🎉 全部通过！HTTP + SSE 流式反代均正常")
    finally:
        api_proc.terminate()
        api_proc.wait()
        t1.cancel(); t2.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass