# -*- coding: utf-8 -*-
"""
轮机智脑 · 冷却系统 智能体监控服务（云端版）

取代本地 agent_vision_monitor.py(7864)：不再依赖本地截图与本地 Gradio，
全部逻辑跑在 Render 服务进程内，通过同源路由 /agent/* 对外提供服务。

数据来源（优先级）：
  1. 浏览器 DOM 真实值 —— cooling-system.html 每 1s POST /agent/__state
  2. 云端模拟器 —— 移植自 cooling-system.html 的负载漂移 + 传感器噪声 +
     故障注入逻辑，无人打开页面时也能持续产出诊断

诊断流程：
  Python 按 KB 基准插值判定各参数状态（正常/超容差/严重超差）→
  DSR1 直连云端 API 撰写工况评估文（未配置密钥时用本地模板兜底）→
  写 latest_diagnosis.json + history/<id>.json + history/index.json
"""
import os
import base64
import json
import math
import random
import re
import threading
import time
import datetime
import traceback
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI

# ===== 配置 =====
DATA_DIR = Path(__file__).parent / "agent_data"
HIST_DIR = DATA_DIR / "history"
LATEST = DATA_DIR / "latest_diagnosis.json"
INDEX = HIST_DIR / "index.json"

INTERVAL = int(os.environ.get("AGENT_INTERVAL", "300"))
DOM_STATE_TTL = 60          # 浏览器 DOM 状态超过 60s 视为过期，回落到模拟器
HISTORY_KEEP = 50

DSR1_API_BASE = os.environ.get("DSR1_API_BASE", "https://chat.cqjtu.edu.cn/ds/api/v1")
DSR1_API_KEY = os.environ.get("DSR1_API_KEY", "")
DSR1_MODEL = os.environ.get("DSR1_MODEL", "doubao-2.0-pro")
_dsr1 = OpenAI(base_url=DSR1_API_BASE, api_key=DSR1_API_KEY) if DSR1_API_KEY else None

router = APIRouter(prefix="/agent")

# ===== KB 基准（与 cooling-system.html / 本地版完全一致） =====
COOLING_BASELINE = {
    "淡水进水温度":       {"unit": "℃", "tolerance": 5, "values": {25: 17,   50: 19,   75: 22,   90: 26,   100: 32,   110: 34}},
    "缸套水出水温度":     {"unit": "℃", "tolerance": 3, "values": {25: 86.3, 50: 88.4, 75: 89.9, 90: 92.8, 100: 95.2, 110: 99.1}},
    "冷却淡水出水温度":   {"unit": "℃", "tolerance": 2, "values": {25: 83.7, 50: 79.6, 75: 78.8, 90: 78.4, 100: 78.8, 110: 78.5}},
    "活塞冷却油出口温度": {"unit": "℃", "tolerance": 2, "values": {25: 48.3, 50: 54.1, 75: 57.6, 90: 58.8, 100: 59.1, 110: 59.8}},
}
LOADS = [25, 50, 75, 90, 100, 110]

FAULT_SCENARIOS = {
    "normal":       {"offsets": {}, "fail": []},
    "cyl_high":     {"offsets": {"缸套水出水温度": 7}, "fail": []},
    "pco_high":     {"offsets": {"活塞冷却油出口温度": 5}, "fail": []},
    "coolin_high":  {"offsets": {"淡水进水温度": 12}, "fail": []},
    "coolout_high": {"offsets": {"冷却淡水出水温度": 5}, "fail": []},
    "sensor_fail":  {"offsets": {}, "fail": ["活塞冷却油出口温度"]},
    "multi":        {"offsets": {"缸套水出水温度": 7, "活塞冷却油出口温度": 5, "淡水进水温度": 12}, "fail": []},
}


def get_baseline(name, load):
    v = COOLING_BASELINE[name]["values"]
    if load <= 25:
        return v[25]
    if load >= 110:
        return v[110]
    for i in range(len(LOADS) - 1):
        if LOADS[i] <= load <= LOADS[i + 1]:
            t = (load - LOADS[i]) / (LOADS[i + 1] - LOADS[i])
            return round(v[LOADS[i]] + t * (v[LOADS[i + 1]] - v[LOADS[i]]), 2)
    return v[100]


# ===== 云端模拟器（移植自 cooling-system.html updateLoad/updateSensors） =====
class _Sim:
    def __init__(self):
        self.lock = threading.Lock()
        self.load = 85.0
        self.phase = 0.0
        self.fault = "normal"

    def set_fault(self, key):
        with self.lock:
            self.fault = key if key in FAULT_SCENARIOS else "normal"

    def tick(self):
        with self.lock:
            self.phase += 0.015
            self.load = 81 + math.sin(self.phase) * 16 + math.sin(self.phase * 2.3) * 4 + random.uniform(-1, 1)
            self.load = max(25.0, min(110.0, self.load))
            load, phase, sc = self.load, self.phase, FAULT_SCENARIOS.get(self.fault, FAULT_SCENARIOS["normal"])
            sensors = {}
            for name in COOLING_BASELINE:
                if name in sc["fail"]:
                    sensors[name] = 0
                    continue
                base = get_baseline(name, load)
                noise = random.uniform(-0.8, 0.8)
                drift = math.sin(phase * 0.7 + len(name)) * 0.5
                off = sc["offsets"].get(name, 0)
                sensors[name] = round(base + noise + drift + off, 2)
            return {"load": round(load, 2), "sensors": sensors, "fault": self.fault}


_sim = _Sim()


# ===== 云端真实截图（headless Chromium 渲染 cooling-system.html 并截屏） =====
_last_capture_ok = None  # None=未尝试 True/False=最近一次结果


def _playwright_ok():
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False


def _prune_shots(shots_dir: Path, keep: int = 10):
    try:
        files = sorted(shots_dir.glob("capture_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            old.unlink(missing_ok=True)
    except Exception:
        pass


def capture_panel(load: float, fault: str):
    """headless Chromium 打开冷却面板页面真实截图；失败返回 None（回退无截图模式）。"""
    global _last_capture_ok
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _last_capture_ok = False
        return None
    shots_dir = DATA_DIR / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = shots_dir / f"capture_{ts}.png"
    port = os.environ.get("PORT", "10000")
    url = (f"http://127.0.0.1:{port}/static/cooling-system.html"
           f"?mode=manual&load={load}&fault={fault}")
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=[
                "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
                "--disable-extensions", "--hide-scrollbars",
                "--single-process", "--disable-software-rasterizer",
            ])
            try:
                ctx = browser.new_context(viewport={"width": 1400, "height": 1086})
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2500)
                page.screenshot(path=str(path))
            finally:
                try:
                    browser.close()
                except Exception:
                    pass
        if path.exists() and path.stat().st_size > 0:
            _prune_shots(shots_dir)
            _last_capture_ok = True
            print(f"[agent] 云端截图 OK {path.name} size={path.stat().st_size}", flush=True)
            return path
    except Exception as e:
        print(f"[agent] 云端截图失败（回退无截图模式）: {type(e).__name__}: {str(e)[:200]}", flush=True)
    _last_capture_ok = False
    return None

# ===== 浏览器 DOM 真实值缓存 =====
_dom_state = None
_dom_lock = threading.Lock()

# ===== 诊断调度状态 =====
_run_lock = threading.Lock()
_diagnosing = True
_diag_lock = threading.Lock()
_thread_started = False
_start_lock = threading.Lock()


def _collect_readings():
    """优先取新鲜 DOM 真实值，否则用云端模拟器。"""
    global _dom_state
    now = time.time()
    with _dom_lock:
        dom = dict(_dom_state) if _dom_state else None
    if dom and isinstance(dom.get("load"), (int, float)) and now - dom.get("_recv_ts", 0) <= DOM_STATE_TTL:
        return float(dom["load"]), dom.get("sensors", {}), "dom"
    sim = _sim.tick()
    return sim["load"], sim["sensors"], "sim"


def _judge(load, raw_sensors):
    """按 KB 基准判定状态，组装 sensors + concerns（逻辑与本地版 parse_result 一致）。"""
    sensors, concerns = {}, []
    for name, bl in COOLING_BASELINE.items():
        raw = raw_sensors.get(name)
        if raw is None or raw == "":
            sensors[name] = {"value": None, "unit": bl["unit"], "system": "冷却", "status": "未知"}
            concerns.append(f"{name}：无读数")
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            sensors[name] = {"value": None, "unit": bl["unit"], "system": "冷却", "status": "未知"}
            concerns.append(f"{name}：读数无效（{raw}）")
            continue
        base = get_baseline(name, load)
        delta = round(v - base, 2)
        tol = bl["tolerance"]
        if abs(delta) <= tol:
            st = "正常"
        elif abs(delta) <= tol * 2:
            st = "超容差"
        else:
            st = "严重超差"
        sensors[name] = {"value": round(v, 2), "unit": bl["unit"], "system": "冷却", "status": st}
        if st != "正常":
            concerns.append(
                f"{name}：实测 {v}{bl['unit']}，基准 {base}{bl['unit']}，"
                f"偏差 {'+' if delta >= 0 else ''}{delta}{bl['unit']}（{st}）"
            )
    statuses = [s["status"] for s in sensors.values()]
    if any(s == "严重超差" for s in statuses):
        overall = "异常"
    elif any(s in ("超容差", "未知") for s in statuses):
        overall = "关注"
    else:
        overall = "正常"
    return sensors, concerns, overall


def _assessment_text(load, sensors, concerns, overall, img_path=None):
    """DSR1 撰写工况评估文（有截图时带图真视觉）；未配置密钥或失败时本地模板兜底。"""
    rows = []
    for name, s in sensors.items():
        rows.append(f"- {name}：{s['value']}{s['unit']}（{s['status']}，基准 {get_baseline(name, load)}{s['unit']}）")
    table = "\n".join(rows)
    issues = "\n".join("- " + c for c in concerns) if concerns else "无"
    fallback = (
        f"当前主机负载 {load}%，冷却系统四参数与 KB 基准比对结论：{overall}。"
        + ("主要关注点：" + "；".join(concerns) + "。建议按容差判定结果排查对应传感器与冷却回路。"
           if concerns else "各参数偏差均在容差范围内，冷却系统工况正常，维持常规监测即可。")
    )
    if _dsr1 is None:
        return fallback
    user_text = (
        f"主机负载：{load}%\n四参数实测（含状态判定）：\n{table}\n\n"
        f"超差关注点：\n{issues}\n\n整体判定：{overall}。请写工况评估文。"
    )
    try:
        if img_path and Path(img_path).exists():
            b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": user_text + "\n\n附图是刚从云端 headless 浏览器截取的冷却系统监测面板实时截图，请结合截图内容撰写。"},
            ]
        else:
            content = user_text
        resp = _dsr1.chat.completions.create(
            model=DSR1_MODEL,
            temperature=0.4,
            max_tokens=800,
            messages=[
                {"role": "system", "content": (
                    "你是远洋船舶资深轮机长，精通 MAN B&W 12K98ME-C7 大型低速二冲程柴油机的冷却系统运维。"
                    "根据给定的实时测量值与 KB 基准比对结果，用中文写一段 120-200 字的工况评估文："
                    "先给整体结论，再点出异常参数的可能原因（如缸套水温偏高→冷却器结垢/温控阀故障），"
                    "最后给一句处置建议。不要输出 JSON、表格或标题，直接输出正文。"
                )},
                {"role": "user", "content": content},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return re.sub(r"```[a-z]*\n?|```", "", text).strip() or fallback
    except Exception as e:
        print(f"[agent] DSR1 评估失败，使用本地模板兜底: {e}", flush=True)
        return fallback


def _save(diag):
    DATA_DIR.mkdir(exist_ok=True)
    HIST_DIR.mkdir(exist_ok=True)
    LATEST.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    (HIST_DIR / f"{diag['id']}.json").write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        idx = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else []
    except Exception:
        idx = []
    ssum = [
        {"name": n, "value": diag["sensors"].get(n, {}).get("value"),
         "unit": diag["sensors"].get(n, {}).get("unit", "℃"),
         "status": diag["sensors"].get(n, {}).get("status", "未知")}
        for n in COOLING_BASELINE
    ]
    idx.insert(0, {
        "id": diag["id"], "time": diag["time"], "system": "冷却系统", "load": diag["load"],
        "status": diag["status"], "overall_status": diag["assessment"]["overall_status"],
        "anomaly_count": len(diag["anomalies"]), "concern_count": len(diag["assessment"]["concerns"]),
        "screenshot": diag.get("screenshot"), "sensors_summary": ssum,
    })
    INDEX.write_text(json.dumps(idx[:HISTORY_KEEP], ensure_ascii=False, indent=2), encoding="utf-8")


def run_once():
    with _run_lock:
        load, raw_sensors, source = _collect_readings()
        fault = "normal"
        if source == "dom":
            fault = _dom_state.get("fault", "normal") if _dom_state else "normal"
        else:
            fault = _sim.fault
        img = capture_panel(load, fault)
        sensors, concerns, overall = _judge(load, raw_sensors)
        now = datetime.datetime.now()
        diag = {
            "id": now.strftime("%Y%m%d_%H%M%S"),
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "load": load,
            "screenshot": img.name if img else None,
            "sensors": sensors,
            "assessment": {
                "system": "冷却系统",
                "overall_status": overall,
                "assessment": _assessment_text(load, sensors, concerns, overall, img_path=img),
                "concerns": concerns,
            },
            "anomalies": [],
            "diagnosis": None,
            "vision_method": "vision" if img else source,   # vision=云端截图+DSR1 / dom=读DOM / sim=云端模拟器
            "kb_sources": ["温度监测", "负载指数"],
            "status": {"正常": "normal", "关注": "warn", "异常": "abnormal"}[overall],
            "next_check": (now + datetime.timedelta(seconds=INTERVAL)).strftime("%H:%M:%S"),
            "interval_sec": INTERVAL,
        }
        _save(diag)
        print(f"[agent] {diag['time']} diag OK  method={diag['vision_method']}  load={load}%  {overall}", flush=True)
        return diag


def _scheduler():
    time.sleep(3)
    while True:
        with _diag_lock:
            active = _diagnosing
        if active:
            try:
                run_once()
            except Exception:
                traceback.print_exc()
        time.sleep(INTERVAL if active else 5)


def start_agent():
    global _thread_started
    with _start_lock:
        if _thread_started:
            return
        _thread_started = True
    threading.Thread(target=_scheduler, daemon=True, name="agent-monitor").start()
    print(f"[agent] 云端诊断智能体已启动（间隔 {INTERVAL}s，DSR1={'已配置' if _dsr1 else '未配置(模板兜底)'}）", flush=True)


# ===== HTTP 接口（同源挂载，前缀 /agent） =====
class DomState(BaseModel):
    load: float
    sensors: dict
    mode: str = "auto"
    fault: str = "normal"
    frozen: bool = False
    ts: int = 0


@router.post("/__state")
def receive_state(st: DomState):
    global _dom_state
    payload = st.model_dump()
    payload["_recv_ts"] = time.time()
    with _dom_lock:
        _dom_state = payload
    return {"ok": True}


@router.get("/latest")
def latest():
    if LATEST.exists():
        return JSONResponse(json.loads(LATEST.read_text(encoding="utf-8")))
    return {
        "status": "waiting",
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assessment": {"system": "冷却系统", "overall_status": "等待", "assessment": "尚未产生评估", "concerns": []},
        "sensors": {}, "vision_method": "sim", "interval_sec": INTERVAL,
    }


@router.get("/history")
def history():
    try:
        arr = json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else []
    except Exception:
        arr = []
    return {"records": arr, "total": len(arr)}


@router.get("/history/{hid}")
def history_one(hid: str):
    safe = os.path.basename(hid)
    f = HIST_DIR / f"{safe}.json"
    if f.exists():
        return JSONResponse(json.loads(f.read_text(encoding="utf-8")))
    return JSONResponse({"ok": False, "error": "history not found: " + hid}, status_code=404)


@router.get("/screenshot")
@router.get("/screenshot/{sid}")
def screenshot(sid: str = None):
    name = None
    if sid:
        f = HIST_DIR / (os.path.basename(sid) + ".json")
        if f.exists():
            try:
                name = json.loads(f.read_text(encoding="utf-8")).get("screenshot")
            except Exception:
                pass
    elif LATEST.exists():
        try:
            name = json.loads(LATEST.read_text(encoding="utf-8")).get("screenshot")
        except Exception:
            pass
    if name:
        fp = DATA_DIR / "screenshots" / os.path.basename(name)
        if fp.exists():
            return FileResponse(str(fp), media_type="image/png", headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "screenshot not found"}, status_code=404)


@router.get("/health")
def health():
    with _diag_lock:
        active = _diagnosing
    return {
        "ok": True,
        "service": "agent_monitor_cloud",
        "interval_sec": INTERVAL,
        "dsr1": bool(_dsr1),
        "playwright": _playwright_ok(),
        "last_capture_ok": _last_capture_ok,
        "dom_state_fresh": bool(_dom_state and time.time() - _dom_state.get("_recv_ts", 0) <= DOM_STATE_TTL),
        "latest_exists": LATEST.exists(),
        "diagnosing": active,
    }


@router.get("/start_diagnosis")
def start_diagnosis():
    global _diagnosing
    with _diag_lock:
        _diagnosing = True
    threading.Thread(target=run_once, daemon=True).start()
    return {"ok": True, "diagnosing": True, "msg": "诊断已开始"}


@router.get("/stop_diagnosis")
def stop_diagnosis():
    global _diagnosing
    with _diag_lock:
        _diagnosing = False
    return {"ok": True, "diagnosing": False, "msg": "诊断已暂停"}


@router.get("/trigger")
def trigger():
    try:
        return {"ok": True, "diag": run_once()}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/set_fault")
def set_fault(key: str = "normal"):
    _sim.set_fault(key)
    return {"ok": True, "fault": _sim.fault,
            "note": "仅影响云端模拟器读数；浏览器打开面板时以页面注入的故障为准"}
