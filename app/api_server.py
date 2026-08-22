"""
Marine Engine AI — FastAPI Backend
Replaces Gradio UI with RESTful API for new frontend.
"""
import sys
import os
import json
import asyncio
from pathlib import Path

# Add local_agent to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
import uvicorn
import httpx
import uuid
from datetime import datetime

# Import original visualizer functions
from visualizer import (
    SYSTEM_TABS, SYSTEM_META, KB_BASELINE, SessionData,
    classify_intent, extract_data_from_message,
    call_llm_stream, call_school_llm_stream, call_dsv4_stream,
    build_enhanced_chart,
    INTENT_TO_KB_KEY, INTENT_PROMPTS, DATA_HEAVY_INTENTS, KB_INTENTS,
    MAIN_AGENT_PROMPT, list_saved_sessions,
    delete_session, save_session, get_retriever,
    build_system_cards_html, build_overview_content_html,
)

app = FastAPI(title="Marine Engine AI API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session and retriever
session = SessionData()
retriever = get_retriever()

# ─── Anomaly Tracker ────────────────────────────────────────────────────────────
ANOMALY_FILE = Path(__file__).parent / "data" / "anomalies.json"



def _chat_pipeline(query: str, history: list, selected_system: str):
    """Core chat pipeline (async generator for SSE)."""
    if not query.strip():
        yield f"data: {json.dumps({'type': 'empty'})}\n\n"
        return

    history = history or []
    intent = classify_intent(query)
    kb_key = INTENT_TO_KB_KEY.get(intent)
    context = ""
    if kb_key:
        context = retriever.retrieve_for_intent(query, kb_key, top_k=5)

    prompt_template = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["其他"])
    user_prompt = prompt_template.format(query=query, context=context)

    if intent in DATA_HEAVY_INTENTS:
        gen = call_dsv4_stream(MAIN_AGENT_PROMPT, user_prompt)
        model_label = "DS V4 Pro"
    elif intent in KB_INTENTS:
        gen = call_school_llm_stream(MAIN_AGENT_PROMPT, user_prompt)
        model_label = "DS V3"
    else:
        gen = call_llm_stream(MAIN_AGENT_PROMPT, user_prompt)
        model_label = "本地qwen"

    # Send intent + model info
    yield f"data: {json.dumps({'type': 'meta', 'model': model_label, 'intent': intent}, ensure_ascii=False)}\n\n"

    # Stream LLM response
    full_response = ""
    for chunk in gen:
        full_response += chunk
        yield f"data: {json.dumps({'type': 'token', 'text': chunk}, ensure_ascii=False)}\n\n"

    # Extract data points
    points = extract_data_from_message(query)
    chart_data = {}
    if points:
        for pt in points:
            session.add(pt["load"], pt["name"], pt["value"], query)
        # Build chart update
        params = SYSTEM_TABS.get(selected_system, [])
        fig = build_enhanced_chart(selected_system, params, session, None, "all")
        chart_data = {"system": selected_system, "points": len(points)}

    yield f"data: {json.dumps({'type': 'done', 'chart': chart_data, 'full_response': full_response[:200]}, ensure_ascii=False)}\n\n"


@app.get("/api/chat")
async def chat_stream(
    query: str = Query(...),
    system: str = Query(default="排气系统"),
):
    """SSE chat endpoint."""
    def generate():
        yield from _chat_pipeline(query, [], system)
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat_stream_post(data: dict):
    """SSE chat endpoint (POST). 兼容 query / message 两种字段名。"""
    query = data.get("query") or data.get("message") or ""
    selected_system = data.get("system", "排气系统")
    history = data.get("history", [])

    def generate():
        yield from _chat_pipeline(query, history, selected_system)
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/systems")
async def get_systems():
    """Get all system tabs with metadata."""
    systems = {}
    for name, params in SYSTEM_TABS.items():
        meta = SYSTEM_META.get(name, {})
        systems[name] = {
            "params": params,
            "icon": meta.get("icon", ""),
            "color": meta.get("color", "#3B82F6"),
            "label": meta.get("label", name),
        }
    return {"systems": systems}


@app.get("/api/overview")
async def get_overview():
    """Get overview page data."""
    overview_html = build_overview_content_html()
    cards_html = build_system_cards_html("排气系统")
    return {
        "html": overview_html,
        "cards": cards_html,
        "session_points": len(session.points),
        "session_params": len(session.get_params()),
    }


@app.get("/api/chart/{system_name}")
async def get_chart(
    system_name: str,
    highlight_load: int = Query(default=75),
    time_range: str = Query(default="all"),
):
    """Get chart data for a specific system."""
    if system_name not in SYSTEM_TABS:
        raise HTTPException(status_code=404, detail="System not found")

    params = SYSTEM_TABS[system_name]
    fig = build_enhanced_chart(system_name, params, session, highlight_load, time_range)

    return {
        "system": system_name,
        "params": params,
        "chart_json": fig.to_json() if hasattr(fig, 'to_json') else None,
    }


@app.get("/api/history")
async def get_history(type_filter: str = Query(default="all")):
    """List saved history sessions."""
    try:
        files = list_saved_sessions(type_filter)
        return {"files": files}
    except Exception as e:
        return {"files": [], "error": str(e)}


@app.get("/api/list")
async def get_history_list(type_filter: str = Query(default="all")):
    """List saved history sessions with rich metadata."""
    try:
        import json as _json
        history_dir = Path(__file__).parent / "history_data"
        files = sorted(history_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        result = []
        for f in files:
            try:
                data = _json.loads(f.read_text(encoding="utf-8"))
                _type = data.get("type", "text")
                if type_filter != "all" and _type != type_filter:
                    continue
                points = data.get("points") or []
                param_names = list(dict.fromkeys(p.get("param", "") for p in points if p.get("param")))
                result.append({
                    "filename": f.name,
                    "label": data.get("label", f.stem),
                    "timestamp": data.get("timestamp", ""),
                    "type": _type,
                    "point_count": data.get("point_count", len(points)),
                    "parameters": param_names,
                    "parameter_count": len(param_names)
                })
            except Exception:
                continue
        return result
    except Exception as e:
        return []


@app.get("/api/history/{filename}")
@app.get("/api/data")
async def load_history(filename: str = "", file: str = Query(default="")):
    """Load a specific history session."""
    try:
        import json
        target = filename or file
        if not target:
            raise HTTPException(status_code=400, detail="Missing filename")
        history_dir = Path(__file__).parent / "history_data"
        filepath = history_dir / target
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # If points not at top level, extract from messages
        if "points" not in data and "messages" in data:
            pts = []
            for msg in data["messages"]:
                if msg["role"] == "assistant":
                    for c in msg.get("content", []):
                        try:
                            inner = json.loads(c) if isinstance(c, str) else c
                            if "points" in inner:
                                pts.extend(inner["points"])
                        except:
                            pass
            data["points"] = pts
        # Inject KB baselines for params present in points (for radar/heatmap)
        if data.get("points") and not data.get("kb_baselines"):
            from visualizer import KB_BASELINE
            data["kb_baselines"] = {
                p["param"]: KB_BASELINE[p["param"]]
                for p in data["points"]
                if p.get("param") in KB_BASELINE
            }
        data["filename"] = target
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Admin token guard (optional) ────────────────────────────────────────────
# 设置环境变量 ADMIN_TOKEN 后，破坏性接口要求请求头 X-Admin-Token 匹配；
# 未设置则放行（本地开发不受影响）。线上建议在 Render Secrets 配置。
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


async def _require_admin(request: Request):
    if ADMIN_TOKEN and request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="X-Admin-Token invalid or missing")


@app.delete("/api/history/{filename}")
async def remove_history(filename: str, request: Request):
    """Delete a history session."""
    await _require_admin(request)
    try:
        remaining, msg = delete_session(filename)
        return {"remaining": list(remaining) if remaining else [], "message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session")
async def get_session():
    """Get current session data."""
    params = session.get_params()
    total_points = len(session.points)
    return {
        "params": params,
        "total_points": total_points,
        "points": session.to_markdown(),
    }


@app.delete("/api/session")
async def clear_session(request: Request):
    """Clear current session data."""
    await _require_admin(request)
    session.clear()
    return {"status": "cleared"}


# ─── Anomaly endpoints ──────────────────────────────────────────────────────

# Serve static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================
# 反向代理：/demo/* -> Gradio 7861, /ai/* -> Agent 7864
# 干掉 nginx，单进程 FastAPI 自己做反代（避免 nginx 配置/端口/daemon off 各种坑）
# ============================================================
_HOP_BY_HOP = {"host", "content-length", "connection", "transfer-encoding", "upgrade"}


async def _reverse_proxy(request: Request, upstream_base: str, path_override: str = None):
    """把请求转发到上游服务。

    关键设计：用 stream 模式发起请求，从响应头读取 Content-Type 判断是否是 SSE。
    不能先用 client.request() 预读——SSE 永远不结束，那一行会无限期挂着。

    - SSE/event-stream 响应：用 StreamingResponse 流式透传（24h 长 timeout）
    - 其他响应：直接 aiter_raw 读完一次性返回（避免 HTTP/1.0 + stream 兼容问题）

    path_override: 若指定，用它替换原始路径（用于剥前缀场景）
    """
    upstream_path = path_override if path_override is not None else request.url.path
    upstream_url = f"{upstream_base}{upstream_path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}

    # 长 timeout：read 24h，connect 5s。SSE 会一直挂着
    timeout = httpx.Timeout(86400.0, connect=5.0)

    # 一次性建立 stream 请求，从响应头判断 content-type
    client = httpx.AsyncClient(timeout=timeout)
    try:
        req = client.build_request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            content=body,
        )
        upstream_resp = await client.send(req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        return JSONResponse(
            {"error": "upstream not ready", "upstream": upstream_base, "path": upstream_path},
            status_code=503,
        )
    except Exception as e:
        await client.aclose()
        return JSONResponse({"error": str(e), "upstream": upstream_base}, status_code=502)

    content_type = upstream_resp.headers.get("content-type", "").lower()
    is_streaming = (
        "event-stream" in content_type
        or (
            upstream_resp.headers.get("transfer-encoding", "").lower() == "chunked"
            and "json" not in content_type
            and not upstream_resp.headers.get("content-length")
        )
    )

    response_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in _HOP_BY_HOP | {"content-length"}
    }

    if is_streaming:
        # SSE 长连接：流式透传 + 心跳保活，靠 background task 在断开时清理
        # 心跳：每 30s 无数据就发一个 SSE comment 包，避免 Cloudflare 免费层 100s 超时掐断
        return StreamingResponse(
            _sse_with_heartbeat(upstream_resp),
            status_code=upstream_resp.status_code,
            headers=response_headers,
            background=BackgroundTask(_close_stream, upstream_resp, client),
        )

    # 非 SSE：读完一次性返回，但要保留 client 直到读完才能正确 aclose
    async def _collect_and_close():
        try:
            await upstream_resp.aclose()
        finally:
            await client.aclose()

    return StreamingResponse(
        _collect_body(upstream_resp),
        status_code=upstream_resp.status_code,
        headers=response_headers,
        background=BackgroundTask(_collect_and_close),
    )


async def _collect_body(resp):
    """一次性读完整 body 给非 SSE 响应。"""
    async for chunk in resp.aiter_raw():
        yield chunk


async def _close_stream(resp, client):
    """SSE 断开时清理 httpx 资源。"""
    try:
        await resp.aclose()
    finally:
        await client.aclose()


async def _sse_with_heartbeat(resp):
    """SSE 流式透传 + 心跳保活。

    Cloudflare 免费层会在 100s 无数据传输时主动掐断长连接，
    Gradio 6.x 检测到断开会弹 "Connection lost / Attempting reconnection..."。
    这里每 30s 检测一次：若上游 30s 内没发数据，就主动发一个 SSE comment
    包（`: heartbeat\\n\\n`，Gradio 客户端会忽略），让连接保持活跃。

    实现要点：后台读协程持续从上游 aiter_raw() 读数据放入队列，主生成器只
    对「队列等待」设 30s 超时——读操作本身永不被 cancel（cancel 会直接
    断开 httpx 到上游的连接，导致后续数据丢失）。
    """
    import asyncio
    queue: asyncio.Queue = asyncio.Queue()

    async def _reader():
        try:
            async for chunk in resp.aiter_raw():
                await queue.put(("data", chunk))
        except StopAsyncIteration:
            pass
        except Exception:
            pass
        finally:
            await queue.put(("done", None))

    read_task = asyncio.ensure_future(_reader())
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # 30s 内上游无新数据，发心跳保活，连接不中断
                yield b": heartbeat\n\n"
                continue
            if kind == "done":
                break
            yield payload
    finally:
        if not read_task.done():
            read_task.cancel()


# （注：Gradio 6.x 实际用 HTTP POST /queue/join + GET /queue/data (SSE/EventSource)
# 通信，不是 WebSocket。之前加的 WS 反代路由已删除——直接走 HTTP 反代即可。）


@app.api_route("/demo/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_demo(full_path: str, request: Request):
    """Reverse proxy /demo/* -> Gradio 7861（剥 /demo 前缀，因 Gradio 6.x 不设 root_path）."""
    new_path = "/" + full_path if full_path else "/"
    return await _reverse_proxy(request, "http://127.0.0.1:7861", path_override=new_path)


@app.get("/demo")
async def redirect_demo():
    """Redirect bare /demo to /demo/."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/demo/", status_code=301)


@app.api_route("/ai/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_ai(full_path: str, request: Request):
    """Reverse proxy /ai/* -> Agent 7864（剥 /ai 前缀，因为 7864 内部路由不带 /ai）."""
    # 剥掉 /ai 前缀：/ai/health -> /health
    new_path = "/" + full_path if full_path else "/"
    return await _reverse_proxy(request, "http://127.0.0.1:7864", path_override=new_path)


@app.get("/ai")
async def serve_ai_index():
    """Serve cooling-diagnosis.html for bare /ai."""
    diag = static_dir / "cooling-diagnosis.html"
    if diag.exists():
        return FileResponse(str(diag), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return JSONResponse({"error": "cooling-diagnosis.html not found"}, status_code=404)


@app.get("/")
async def serve_gateway():
    """Serve the marketing landing page (with 开始演示/历史数据 buttons)."""
    idx = static_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    # 兜底：landing 不存在则返回 history（兼容旧部署）
    hist = static_dir / "history.html"
    if hist.exists():
        return FileResponse(str(hist), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "Marine Engine AI API", "docs": "/docs"}


@app.get("/dashboard")
async def serve_dashboard():
    """Serve the 2D dashboard."""
    dash = static_dir / "dashboard.html"
    if dash.exists():
        return FileResponse(str(dash), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "Dashboard not found — check static/ dir"}


@app.get("/history")
async def serve_history():
    """Serve the history data browser."""
    hist = static_dir / "history.html"
    if hist.exists():
        return FileResponse(str(hist), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "History page not found — check static/ dir"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7862"))
    print("=" * 60)
    print(f"Marine Engine AI — FastAPI Backend (with reverse proxy)")
    print(f"  Listening: 0.0.0.0:{port}")
    print(f"  Gateway:   /              (落地页+静态)")
    print(f"  Demo:      /demo/*        -> Gradio :7861")
    print(f"  AI:        /ai/*          -> Agent  :7864")
    print(f"  Docs:      /docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
