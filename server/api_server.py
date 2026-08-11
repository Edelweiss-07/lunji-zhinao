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

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
from datetime import datetime

# Import original visualizer functions
from visualizer_core import (
    SYSTEM_TABS, SYSTEM_META, KB_BASELINE, SessionData,
    classify_intent, extract_data_from_message,
    call_llm_stream, call_school_llm_stream, call_dsv4_stream,
    build_enhanced_chart,
    INTENT_TO_KB_KEY, INTENT_PROMPTS, DATA_HEAVY_INTENTS, KB_INTENTS,
    MAIN_AGENT_PROMPT, list_saved_sessions,
    save_session, get_retriever,
    build_system_cards_html, build_overview_content_html,
    _safe_history_path,
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
        filepath = _safe_history_path(target)
        if filepath is None:
            raise HTTPException(status_code=400, detail="Invalid filename")
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
        # KB_BASELINE already imported from visualizer_core at module top
        if data.get("points") and not data.get("kb_baselines"):
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
    filepath = _safe_history_path(filename)
    if filepath is None or not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        filepath.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    remaining = list_saved_sessions()
    return {"remaining": remaining, "message": f"已删除 {filename}"}


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

    # 着陆页资源（相对路径 images/ libs/ 及根级文件 → 挂到根路径，与 index.html 相对引用对应）
    for _sub in ("images", "libs"):
        _d = static_dir / _sub
        if _d.exists():
            app.mount(f"/{_sub}", StaticFiles(directory=str(_d)), name=f"site-{_sub}")
    for _f in ("fish_data.js", "ocean_background.mp4", "footer_sea.mp4", "favicon.svg", "icons.svg"):
        _fp = static_dir / _f
        if _fp.exists():
            app.get(f"/{_f}")(
                lambda _fp=_fp: FileResponse(str(_fp), headers={"Cache-Control": "no-cache"})
            )


@app.get("/")
async def serve_landing():
    """Serve the marketing landing page（海洋主题着陆页 index.html，同本地首页）。"""
    idx = static_dir / "index.html"
    if idx.exists():
        return FileResponse(str(idx), headers={"Cache-Control": "no-cache"})
    return {"message": "Marine Engine AI API", "docs": "/docs"}


@app.get("/portal")
async def serve_portal():
    """Serve the Gradio portal shell (tabs + iframe → /app/)."""
    portal = static_dir / "portal.html"
    if portal.exists():
        return FileResponse(str(portal), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "Portal not found — check static/ dir"}


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
    port = int(os.environ.get("PORT", 7862))
    print("=" * 60)
    print("Marine Engine AI — FastAPI Backend")
    print(f"  API: http://localhost:{port}")
    print(f"  Portal: http://localhost:{port}/")
    print(f"  History: http://localhost:{port}/history")
    print(f"  Docs: http://localhost:{port}/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=port)
