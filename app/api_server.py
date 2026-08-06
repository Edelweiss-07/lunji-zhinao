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

from fastapi import FastAPI, Query, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx
import websockets
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
    """SSE chat endpoint (POST)."""
    query = data.get("query", "")
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


@app.delete("/api/history/{filename}")
async def remove_history(filename: str):
    """Delete a history session."""
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
async def clear_session():
    """Clear current session data."""
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
    """把请求流式转发到上游服务。
    path_override: 若指定,用它替换原始路径(用于剥前缀场景)
    """
    upstream_path = path_override if path_override is not None else request.url.path
    upstream_url = f"{upstream_base}{upstream_path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}

    try:
        # Gradio SSE 是长连接，不发数据会一直挂着；read timeout 设 24h 避免被反代切断
        async with httpx.AsyncClient(timeout=httpx.Timeout(86400.0, connect=5.0)) as client:
            upstream_resp = await client.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                content=body,
            )
    except httpx.ConnectError:
        return JSONResponse(
            {"error": "upstream not ready", "upstream": upstream_base, "path": upstream_path},
            status_code=503,
        )
    except Exception as e:
        return JSONResponse({"error": str(e), "upstream": upstream_base}, status_code=502)

    # 过滤 hop-by-hop headers 再回传
    response_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in _HOP_BY_HOP | {"content-length"}
    }
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=response_headers,
    )


# ============================================================
# WebSocket 反代（Gradio 用 WS 通信，HTTP 反代拿不掉 Connection/Upgrade）
# 在 ASGI middleware 层拦截 WS 握手，做双向转发
# ============================================================
@app.websocket("/demo/{full_path:path}")
async def proxy_demo_ws(websocket: WebSocket, full_path: str):
    """WS 反代 /demo/* -> Gradio 7861（保留 /demo 前缀）."""
    await _ws_proxy(websocket, "ws://127.0.0.1:7861/demo/" + full_path)


@app.websocket("/ai/{full_path:path}")
async def proxy_ai_ws(websocket: WebSocket, full_path: str):
    """WS 反代 /ai/* -> Agent 7864（剥 /ai 前缀）."""
    await _ws_proxy(websocket, "ws://127.0.0.1:7864/" + full_path)


async def _ws_proxy(websocket: WebSocket, upstream_url: str):
    """用 starlette WebSocket 接客户端，websockets 库连上游，双向转发."""
    await websocket.accept()
    # 透传 subprotocol（Gradio 用 "gradio-protocol"）
    subprotocols = websocket.headers.get("sec-websocket-protocol")
    subprotocol_list = [s.strip() for s in subprotocols.split(",")] if subprotocols else None
    selected_subprotocol = subprotocol_list[0] if subprotocol_list else None

    try:
        async with websockets.connect(
            upstream_url,
            subprotocols=subprotocol_list,
            ping_interval=None,  # 关闭 ping（防止长连接被误判）
        ) as upstream:
            # 告知客户端选了哪个 subprotocol
            if selected_subprotocol and upstream.subprotocol != selected_subprotocol:
                pass  # 实际握手已经在 connect 里完成，无需手动通知

            async def client_to_upstream():
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg["type"] == "websocket.disconnect":
                            break
                        if msg.get("text") is not None:
                            await upstream.send(msg["text"])
                        elif msg.get("bytes") is not None:
                            await upstream.send(msg["bytes"])
                except Exception:
                    pass

            async def upstream_to_client():
                try:
                    async for msg in upstream:
                        if isinstance(msg, str):
                            await websocket.send_text(msg)
                        else:
                            await websocket.send_bytes(msg)
                except Exception:
                    pass

            # 任一方向断开就结束
            done, pending = await asyncio.wait(
                [asyncio.create_task(client_to_upstream()),
                 asyncio.create_task(upstream_to_client())],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
    except Exception as e:
        # WS 已 accept，错误就只能关闭
        try:
            await websocket.close(code=1011, reason=str(e)[:100])
        except Exception:
            pass


@app.api_route("/demo/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_demo(full_path: str, request: Request):
    """Reverse proxy /demo/* -> Gradio 7861（保留 /demo 前缀，Gradio root_path=/demo）."""
    return await _reverse_proxy(request, "http://127.0.0.1:7861")


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
