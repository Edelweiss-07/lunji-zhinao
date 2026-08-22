"""
Marine Engine AI — 轮机负载可视化监控面板 (Enhanced)
深色主题 · 多维度仪表盘 · 增强图表 · 交互式监控
"""
import json
import re
import sys
import os
from pathlib import Path
import base64
from datetime import datetime, timedelta
from collections import defaultdict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ["PYTHONIOENCODING"] = "utf-8"

import gradio as gr
from openai import OpenAI
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from kb_loader import get_retriever, INTENT_KB_MAP
from prompts import (
    INTENT_CLASSIFY_PROMPT, INTENT_PROMPTS, INTENT_TO_KB_KEY, MAIN_AGENT_PROMPT,
)

# ═══════════════════════════════════════════════════════════════
# API Config（密钥全部从环境变量读取）
# ═══════════════════════════════════════════════════════════════

SCHOOL_API_BASE = os.environ.get("SCHOOL_API_BASE", "https://chat.cqjtu.edu.cn/ds/api/v1")
SCHOOL_API_KEY = os.environ.get("SCHOOL_API_KEY", "")
SCHOOL_MODEL = os.environ.get("SCHOOL_MODEL", "deepseek-v3-2-251201")
school_client = OpenAI(base_url=SCHOOL_API_BASE, api_key=SCHOOL_API_KEY) if SCHOOL_API_KEY else None

DSV4_KEY = os.environ.get("DSV4_KEY", "")
DSV4_MODEL = os.environ.get("DSV4_MODEL", "deepseek-chat")
dsv4_client = OpenAI(base_url=SCHOOL_API_BASE, api_key=DSV4_KEY) if DSV4_KEY else None

DSR1_API_BASE = os.environ.get("DSR1_API_BASE", "https://chat.cqjtu.edu.cn/ds/api/v1")
DSR1_API_KEY = os.environ.get("DSR1_API_KEY", "")
DSR1_MODEL = os.environ.get("DSR1_MODEL", "doubao-2.0-pro")
dsr1_client = OpenAI(base_url=DSR1_API_BASE, api_key=DSR1_API_KEY) if DSR1_API_KEY else None

DATA_HEAVY_INTENTS = {
    "轮机负载温差摄氏度变化监控",
    "轮机系统油耗监控",
    "涡轮增压器转速监测",
    "负载参数记录系统",
}
KB_INTENTS = {"船舶分类", "应急响应", "培训需要", "轮机保养和维护", "故障维修"}

# ═══════════════════════════════════════════════════════════════
# KB Baseline Data (12K98ME-C7 Hull 1957 AA2877)
# ═══════════════════════════════════════════════════════════════

LOADS = [25, 50, 75, 90, 100, 110]

KB_BASELINE = {
    "排气温度": {
        "unit": "℃", "system": "排气", "tolerance": 15,
        "values": {25: 224.8, 50: 262.7, 75: 270.5, 90: 288.8, 100: 310.1, 110: 345.9},
    },
    "涡轮前排气温度": {
        "unit": "℃", "system": "排气", "tolerance": 25,
        "values": {25: 380.5, 50: 351.8, 75: 364.0, 90: 389.8, 100: 427.5, 110: 462.0},
    },
    "涡轮后排气温度": {
        "unit": "℃", "system": "排气", "tolerance": 15,
        "values": {25: 251.8, 50: 264.0, 75: 228.5, 90: 227.0, 100: 239.3, 110: 258.0},
    },
    "淡水进水温度": {
        "unit": "℃", "system": "冷却", "tolerance": 5,
        "values": {25: 17, 50: 19, 75: 22, 90: 26, 100: 32, 110: 34},
    },
    "缸套水出水温度": {
        "unit": "℃", "system": "冷却", "tolerance": 3,
        "values": {25: 86.3, 50: 88.4, 75: 89.9, 90: 92.8, 100: 95.2, 110: 99.1},
    },
    "冷却淡水出水温度": {
        "unit": "℃", "system": "冷却", "tolerance": 2,
        "values": {25: 83.7, 50: 79.6, 75: 78.8, 90: 78.4, 100: 78.8, 110: 78.5},
    },
    "活塞冷却油出口温度": {
        "unit": "℃", "system": "冷却", "tolerance": 2,
        "values": {25: 48.3, 50: 54.1, 75: 57.6, 90: 58.8, 100: 59.1, 110: 59.8},
    },
    "涡轮滑油进口温度": {
        "unit": "℃", "system": "滑油", "tolerance": 2,
        "values": {25: 40, 50: 42, 75: 42, 90: 42, 100: 42, 110: 42},
    },
    "涡轮滑油出口温度": {
        "unit": "℃", "system": "滑油", "tolerance": 5,
        "values": {25: 46, 50: 52, 75: 58, 90: 62, 100: 66, 110: 68},
    },
    "扫气温度": {
        "unit": "℃", "system": "扫气", "tolerance": 10,
        "values": {25: 45, 50: 97, 75: 145, 90: 175, 100: 192, 110: 210},
    },
    "扫气接收温度": {
        "unit": "℃", "system": "扫气", "tolerance": 5,
        "values": {25: 18, 50: 20, 75: 27, 90: 35, 100: 40, 110: 44},
    },
    "扫气压力": {
        "unit": "bar", "system": "扫气", "tolerance": 0.2,
        "values": {25: 0.31, 50: 0.95, 75: 1.85, 90: 2.47, 100: 2.86, 110: 3.12},
    },
    "最大爆发压力": {
        "unit": "bar", "system": "燃烧", "tolerance": 2,
        "values": {25: 74.2, 50: 112.5, 75: 143.6, 90: 150.4, 100: 150.0, 110: 150.0},
    },
    "压缩压力": {
        "unit": "bar", "system": "燃烧", "tolerance": 5,
        "values": {25: 49.7, 50: 75.7, 75: 107.8, 90: 120.3, 100: 132.1, 110: 141.7},
    },
    "增压器转速": {
        "unit": "rpm", "system": "增压器", "tolerance": 500,
        "values": {25: 3778, 50: 6585, 75: 8433, 90: 9383, 100: 9945, 110: 10457},
    },
    "增压器空气出口温度": {
        "unit": "℃", "system": "增压器", "tolerance": 5,
        "values": {25: 25.25, 50: 26.63, 75: 27.75, 90: 28.63, 100: 28.63, 110: 27.75},
    },
    "燃油消耗率(实测)": {
        "unit": "g/kWh", "system": "油耗", "tolerance": 5,
        "values": {25: 184.42, 50: 173.93, 75: 172.43, 90: 175.57, 100: 179.58, 110: 183.27},
    },
    "燃油消耗率(修正)": {
        "unit": "g/kWh", "system": "油耗", "tolerance": 3,
        "values": {25: 181.11, 50: 170.25, 75: 168.64, 90: 171.58, 100: 175.39, 110: 179.00},
    },
}

PARAM_ALIASES = {
    "排气温度": ["排气温度", "排温", "排气总管温度"],
    "涡轮前排气温度": ["涡轮前温度", "涡轮前排气温度", "增压器前排气温度", "涡轮前进气温度"],
    "涡轮后排气温度": ["涡轮后温度", "涡轮后排气温度", "增压器后排气温度"],
    "淡水进水温度": ["淡水进水温度", "冷却水进水温度", "缸套水进水温度"],
    "缸套水出水温度": ["缸套水出水温度", "缸套水出口温度", "排气阀淡水出口温度", "冷却水出水温度",
                       "缸套冷却水出水温度", "夹套水出水温度"],
    "冷却淡水出水温度": ["冷却淡水出水温度", "冷却水出口温度", "淡水冷却器出水温度"],
    "活塞冷却油出口温度": ["活塞冷却油出口温度", "活塞冷却油温度", "活塞冷却油出"],
    "涡轮滑油进口温度": ["涡轮滑油进口温度", "涡轮增压器滑油进口温度", "增压器滑油进"],
    "涡轮滑油出口温度": ["涡轮滑油出口温度", "涡轮增压器滑油出口温度", "增压器滑油出"],
    "扫气温度": ["扫气温度", "扫气空气温度", "进气温度"],
    "扫气接收温度": ["扫气接收温度", "扫气箱温度"],
    "扫气压力": ["扫气压力", "扫气空气压力", "进气压力", "增压压力"],
    "最大爆发压力": ["最大爆发压力", "爆发压力", "最高燃烧压力", "Pmax"],
    "压缩压力": ["压缩压力", "压缩终点压力", "Pcomp"],
    "增压器转速": ["增压器转速", "涡轮增压器转速", "增压器RPM"],
    "增压器空气出口温度": ["增压器空气出口温度", "增压器出口温度", "空冷器前温度"],
    "燃油消耗率(实测)": ["燃油消耗率", "油耗", "油耗率", "燃油消耗", "SFOC", "实测油耗"],
    "燃油消耗率(修正)": ["修正油耗", "修正后油耗", "ISO修正油耗"],
}

ALIAS_TO_PARAM = {}
for canonical, aliases in PARAM_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_PARAM[alias] = canonical

SYSTEM_TABS = {
    "排气系统": [k for k in KB_BASELINE if KB_BASELINE[k]["system"] == "排气"],
    "冷却系统": [k for k in KB_BASELINE if KB_BASELINE[k]["system"] == "冷却"],
    "滑油系统": [k for k in KB_BASELINE if KB_BASELINE[k]["system"] == "滑油"],
    "扫气系统": [k for k in KB_BASELINE if KB_BASELINE[k]["system"] == "扫气"],
    "燃烧参数": [k for k in KB_BASELINE if KB_BASELINE[k]["system"] == "燃烧"],
    "增压器":   [k for k in KB_BASELINE if KB_BASELINE[k]["system"] == "增压器"],
    "油耗":     [k for k in KB_BASELINE if KB_BASELINE[k]["system"] == "油耗"],
}

SYSTEM_META = {
    "排气系统": {"icon": "🌡️", "color": "#EF4444", "label": "排气"},
    "冷却系统": {"icon": "❄️",  "color": "#3B82F6", "label": "冷却"},
    "滑油系统": {"icon": "🛢️", "color": "#F59E0B", "label": "滑油"},
    "扫气系统": {"icon": "💨",  "color": "#10B981", "label": "扫气"},
    "燃烧参数": {"icon": "⚡",  "color": "#F97316", "label": "燃烧"},
    "增压器":   {"icon": "🔄",  "color": "#8B5CF6", "label": "增压器"},
    "油耗":     {"icon": "⛽",  "color": "#EC4899", "label": "油耗"},
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Dashboard palette (dark theme)
DASH_COLORS = {
    "bg_base":     "#0F172A",
    "bg_card":     "#1E293B",
    "bg_card_alt": "#1A2332",
    "border":      "#334155",
    "border_glow": "#3B82F6",
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted":  "#64748B",
    "accent_blue": "#3B82F6",
    "accent_green": "#22C55E",
    "accent_red":  "#EF4444",
    "accent_yellow": "#F59E0B",
}

# ═══════════════════════════════════════════════════════════════
# Session Data Store
# ═══════════════════════════════════════════════════════════════

class SessionData:
    """Accumulates extracted sensor data points during a session."""
    def __init__(self):
        self.points: list[dict] = []
        self.start_time = datetime.now()

    def add(self, load: float, param: str, value: float, query: str = ""):
        self.points.append({
            "load": load,
            "param": param,
            "value": value,
            "time": datetime.now().strftime("%H:%M:%S"),
            "dt": datetime.now(),
            "query": query[:60],
        })

    def clear(self):
        self.points = []
        self.start_time = datetime.now()

    def get_params(self) -> list[str]:
        seen = []
        for p in self.points:
            if p["param"] not in seen:
                seen.append(p["param"])
        return seen

    def points_for(self, param: str, time_range: str = "all") -> list[dict]:
        pts = [p for p in self.points if p["param"] == param]
        if time_range != "all":
            cutoff = self._cutoff_for(time_range)
            pts = [p for p in pts if p.get("dt", datetime.now()) >= cutoff]
        return sorted(pts, key=lambda x: x["load"])

    def _cutoff_for(self, time_range: str) -> datetime:
        now = datetime.now()
        mapping = {"1h": timedelta(hours=1), "6h": timedelta(hours=6),
                   "24h": timedelta(hours=24), "7d": timedelta(days=7)}
        return now - mapping.get(time_range, timedelta(days=365))

    def latest_load(self) -> float | None:
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.get("dt", datetime.min))["load"]

    def to_markdown(self) -> str:
        if not self.points:
            return "*暂无数据点*"
        params = self.get_params()
        lines = [f"**已采集 {len(self.points)} 个数据点，{len(params)} 个参数**\n"]
        for param in params:
            pts = self.points_for(param)
            vals = ", ".join(
                f"{p['load']}%→{p['value']}{KB_BASELINE.get(param,{}).get('unit','')}"
                for p in pts
            )
            lines.append(f"- **{param}**: {vals}")
        return "\n".join(lines)

session = SessionData()

# ═══════════════════════════════════════════════════════════════
# DSR1 Data Extraction (unchanged)
# ═══════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """你是船舶轮机传感器数据提取器。从用户消息中提取负载百分比和传感器参数值。

可用参数列表（别名匹配即可）：
- 排气温度、涡轮前温度、涡轮后温度
- 缸套水出水温度、冷却淡水出水温度、淡水进水温度、活塞冷却油出口温度
- 涡轮滑油进口温度、涡轮滑油出口温度
- 扫气温度、扫气接收温度、扫气压力
- 最大爆发压力、压缩压力
- 增压器转速、增压器空气出口温度
- 燃油消耗率(实测)、燃油消耗率(修正)

规则：
1. 负载必须是百分比数字（如50→50%），如果用户说"现在75%负载"就提取75
2. 如果用户说"从X%到Y%"，取Y（目标/较高负载）作为load值
3. 温度单位默认℃、压力默认bar、转速默认rpm、油耗默认g/kWh
4. 如果用户提到了多个参数，全部提取
5. 只提取用户明确提到的值，不要推测

输出纯JSON（不要markdown代码块）。如果用户给出了多组负载-数值对应关系（如"从75%到100%，温度从24到64"），请提取为多个数据点：
{"points": [{"load": 75, "params": [{"name": "淡水进水温度", "value": 24}]}, {"load": 100, "params": [{"name": "淡水进水温度", "value": 64}]}]}
如果只有一个负载值，包在数组中即可：
{"points": [{"load": 75, "params": [{"name": "缸套水出水温度", "value": 90}]}]}"""

def extract_data_from_message(query: str) -> dict | None:
    """Use school DeepSeek V3 to extract structured sensor data."""
    raw = None
    try:
        resp = school_client.chat.completions.create(
            model=SCHOOL_MODEL,
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.1, max_tokens=512, timeout=30,
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"[Extract] V3 failed: {e}")
    if not raw:
        try:
            resp = dsr1_client.chat.completions.create(
                model=DSR1_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.1, max_tokens=512, timeout=15,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[Extract] DSR1 also failed: {e}")
            return None
    if not raw:
        return None

    print(f"[Extract] Raw response ({len(raw)} chars): {raw[:300]}")

    raw_clean = raw.strip()
    if raw_clean.startswith("```"):
        raw_clean = re.sub(r'^```(?:json)?\s*\n?', '', raw_clean)
        raw_clean = re.sub(r'\n?```\s*$', '', raw_clean)
    try:
        data = json.loads(raw_clean)
    except json.JSONDecodeError:
        start = raw_clean.find('{')
        if start == -1:
            return None
        depth = 0
        end = -1
        for i, ch in enumerate(raw_clean[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        try:
            data = json.loads(raw_clean[start:end])
        except json.JSONDecodeError:
            return None

    points = data.get("points")
    if not points:
        load = data.get("load")
        params = data.get("params", [])
        if params and load is not None:
            points = [{"load": load, "params": params}]
    if not points:
        return None

    result_points = []
    for pt in points:
        load = pt.get("load")
        if load is None:
            pct_match = re.findall(r'(\d+(?:\.\d+)?)\s*%', query)
            if pct_match:
                idx = points.index(pt)
                if idx < len(pct_match):
                    load = float(pct_match[idx])
                else:
                    load = float(pct_match[-1])
        if load is None:
            continue
        for p in pt.get("params", []):
            name = p.get("name", "")
            value = p.get("value")
            if not name or value is None:
                continue
            matched = ALIAS_TO_PARAM.get(name)
            if not matched:
                for alias, canonical in ALIAS_TO_PARAM.items():
                    if alias in name or name in alias:
                        matched = canonical
                        break
            if matched:
                result_points.append({"load": float(load), "name": matched, "value": float(value)})

    return result_points if result_points else None

# ═══════════════════════════════════════════════════════════════
# LLM Calls (unchanged)
# ═══════════════════════════════════════════════════════════════

def call_school_llm_stream(system_prompt, user_message):
    """DS V3（学校端点，降级链锚点）。"""
    if school_client is None:
        yield "⚠️ 学校 API 未配置（SCHOOL_API_KEY 为空），请配置后使用"
        return
    try:
        stream = school_client.chat.completions.create(
            model=SCHOOL_MODEL, temperature=0.7, max_tokens=4096, stream=True,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}],
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"❌ 学校API调用失败: {e}"

def call_dsv4_stream(system_prompt, user_message):
    """DS V4 Pro（数据密集意图），失败降级到 DS V3。"""
    if dsv4_client is None:
        yield "[V4 Pro 未配置，降级 DS V3]\n"
        yield from call_school_llm_stream(system_prompt, user_message)
        return
    try:
        stream = dsv4_client.chat.completions.create(
            model=DSV4_MODEL, temperature=0.7, max_tokens=4096, stream=True,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}],
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"[V4 Pro 不可用，降级 DS V3]\n"
        yield from call_school_llm_stream(system_prompt, user_message)

def call_dsr1_chat_stream(system_prompt, user_message):
    """DSR1（doubao，闲聊/图片），失败降级到 DS V3。"""
    if dsr1_client is None:
        yield "⚠️ DSR1 模型未配置（DSR1_API_KEY 为空），请配置后使用"
        return
    try:
        stream = dsr1_client.chat.completions.create(
            model=DSR1_MODEL, temperature=0.7, max_tokens=4096, stream=True,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}],
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"[DSR1 不可用，降级 DS V3]\n"
        yield from call_school_llm_stream(system_prompt, user_message)

def classify_intent(query: str) -> str:
    try:
        resp = school_client.chat.completions.create(
            model=SCHOOL_MODEL, temperature=0.3, max_tokens=256,
            messages=[{"role":"system","content":INTENT_CLASSIFY_PROMPT},{"role":"user","content":query}],
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        return "其他"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            try: data = json.loads(match.group())
            except json.JSONDecodeError: return "其他"
        else: return "其他"
    name = data.get("classificationName", "其他")
    valid = list(INTENT_PROMPTS.keys())
    if name in valid: return name
    for vi in valid:
        if vi in name or name in vi: return vi
    return "其他"

# ═══════════════════════════════════════════════════════════════
# Chart Generation (Enhanced)
# ═══════════════════════════════════════════════════════════════

def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"


def build_empty_chart() -> go.Figure:
    """Return an empty chart placeholder."""
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=480,
        title=dict(text="请选择历史文件查看", font=dict(color="#556677", size=16)),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def build_enhanced_chart(
    system_name: str,
    param_list: list[str],
    session_data: SessionData,
    highlight_load: int | None = None,
    time_range: str = "all",
) -> go.Figure:
    """Build an enhanced Plotly figure for one system with KB baseline, tolerance bands,
    load highlighting, and rich hover info."""

    fig = go.Figure()
    has_any_data = False

    for i, param in enumerate(param_list):
        color = COLORS[i % len(COLORS)]
        kb = KB_BASELINE.get(param, {})
        unit = kb.get("unit", "")
        tolerance = kb.get("tolerance", 0)
        kb_values = kb.get("values", {})

        pts = session_data.points_for(param, time_range)
        if not pts:
            continue

        has_any_data = True

        # ── KB Baseline (dashed line) ──
        if kb_values:
            x_kb = list(kb_values.keys())
            y_kb = list(kb_values.values())
            fig.add_trace(go.Scatter(
                x=x_kb, y=y_kb, mode="lines+markers",
                name=f"{param} [KB基准]",
                line=dict(dash="dot", width=1.5, color=color),
                marker=dict(size=5, symbol="cross-thin", color=color),
                opacity=0.40,
                legendgroup=param,
                hovertemplate=(
                    f"<b>KB基准</b> {param}<br>"
                    f"值: %{{y:.1f}}{unit}<br>"
                    f"负载: %{{x}}%<extra></extra>"
                ),
            ))

            # ── Tolerance band ──
            y_upper = [v + tolerance for v in y_kb]
            y_lower = [v - tolerance for v in y_kb]
            fig.add_trace(go.Scatter(
                x=x_kb + x_kb[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.10),
                line=dict(width=0),
                name=f"{param} ±{tolerance}{unit}",
                showlegend=False,
                legendgroup=param,
                hoverinfo="skip",
            ))

        # ── Actual data points ──
        x_act = [p["load"] for p in pts]
        y_act = [p["value"] for p in pts]
        texts = [f"{p['time']}: {p['query']}" for p in pts]

        # Color markers: red if out of tolerance
        out_colors = []
        for lx, vy in zip(x_act, y_act):
            if lx in kb_values:
                lo = kb_values[lx] - tolerance
                hi = kb_values[lx] + tolerance
                out_colors.append("#EF4444" if vy < lo or vy > hi else color)
            else:
                out_colors.append(color)

        fig.add_trace(go.Scatter(
            x=x_act, y=y_act, mode="lines+markers",
            name=f"{param} [实测]",
            line=dict(width=2.8, color=color, shape="spline", smoothing=0.4),
            marker=dict(
                size=11, symbol="circle",
                color=out_colors,
                line=dict(width=1.5, color="#FFFFFF"),
            ),
            legendgroup=param,
            text=texts,
            hovertemplate=(
                f"<b>实测</b> {param}<br>"
                f"当前值: %{{y:.1f}}{unit}<br>"
                f"负载点: %{{x}}%<br>"
                f"<extra>%{{text}}</extra>"
            ),
        ))

        # ── Add KB value annotations on hover for each actual point ──
        # (handled in unified hover via hovermode below)

    if not has_any_data:
        fig.add_annotation(
            text="<b>📡 暂无数据</b><br><sub>输入传感器参数后自动绘图</sub>",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=20, color="#64748B", family="Inter, sans-serif"),
        )

    # ── Highlight load vertical line ──
    if highlight_load is not None and highlight_load in LOADS:
        fig.add_vline(
            x=highlight_load,
            line=dict(color="#F59E0B", width=2, dash="dash"),
            annotation=dict(
                text=f"当前负载 {highlight_load}%",
                font=dict(color="#FF9F0A", size=12, family="Inter, sans-serif"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#F59E0B",
                borderwidth=1,
                borderpad=4,
                showarrow=False,
                yref="paper", y=0.98,
            ),
            opacity=0.9,
        )
        # Add a highlight band
        fig.add_vrect(
            x0=highlight_load - 2, x1=highlight_load + 2,
            fillcolor="rgba(245,158,11,0.06)",
            line_width=0,
            layer="below",
        )

    # ── Layout ──
    fig.update_layout(
        xaxis=dict(
            title=dict(text="负载 (%)", font=dict(color="#8899AA", size=13)),
            tickvals=LOADS,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            tickfont=dict(color="#8899AA"),
        ),
        yaxis=dict(
            title=dict(text="参数值", font=dict(color="#8899AA", size=13)),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            tickfont=dict(color="#8899AA"),
        ),
        margin=dict(l=55, r=30, t=50, b=55),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(color="#8899AA", size=11),
            bgcolor="rgba(12, 28, 48, 0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(12, 28, 48, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0C1C30",
            font=dict(color="#F0F3F5", size=12, family="Inter, sans-serif"),
            bordercolor="rgba(0,191,165,0.3)",
        ),
        font=dict(family="Inter, Noto Sans SC, sans-serif"),
        title=dict(
            text=f"{SYSTEM_META.get(system_name, {}).get('icon', '')} {system_name} · 实时趋势",
            font=dict(color="#F0F3F5", size=16),
            x=0.5,
            y=0.98,
        ),
    )

    # Update x-axis to always show LOADS ticks + highlight load
    fig.update_xaxes(tickmode="array", tickvals=LOADS)

    return fig


# ═══════════════════════════════════════════════════════════════
# Card HTML Builder
# ═══════════════════════════════════════════════════════════════

def build_system_cards_html(selected_system: str) -> str:
    """Generate HTML for the 7 system cards in the left sidebar."""
    cards = []
    for sys_name, params in SYSTEM_TABS.items():
        meta = SYSTEM_META.get(sys_name, {})
        icon = meta.get("icon", "📊")
        sys_color = meta.get("color", "#64748B")
        label = meta.get("label", sys_name)

        # Check if this system has any data points
        has_data = any(
            session.points_for(p) for p in params
        )

        # Compute latest deviation for the first param with data
        deviation_str = ""
        deviation_class = "dev-none"
        for param in params:
            pts = session.points_for(param)
            if pts:
                latest = pts[-1]
                kb = KB_BASELINE.get(param, {})
                kb_vals = kb.get("values", {})
                if latest["load"] in kb_vals:
                    baseline = kb_vals[latest["load"]]
                    dev_pct = (latest["value"] - baseline) / baseline * 100
                    abs_dev = abs(dev_pct)
                    if abs_dev <= 2:
                        deviation_class = "dev-ok"
                        arrow = ""
                    elif abs_dev <= 8:
                        deviation_class = "dev-warn"
                        arrow = "↑" if dev_pct > 0 else "↓"
                    else:
                        deviation_class = "dev-alert"
                        arrow = "↑" if dev_pct > 0 else "↓"
                    deviation_str = (
                        f'<span class="deviation {deviation_class}">'
                        f'{arrow}{abs_dev:.1f}%</span>'
                    )
                break

        is_active = "active" if sys_name == selected_system else ""
        data_dot = '<span class="data-dot active-dot"></span>' if has_data else '<span class="data-dot"></span>'

        card = f'''
        <div class="sys-card {is_active}" data-system="{sys_name}"
             style="border-left-color: {sys_color};">
            <div class="card-header">
                <span class="card-icon">{icon}</span>
                {data_dot}
            </div>
            <div class="card-body">
                <span class="card-label">{label}</span>
                {deviation_str}
            </div>
            <div class="card-glow" style="background: {sys_color};"></div>
        </div>'''
        cards.append(card)

    return "\n".join(cards)


def build_overview_content_html() -> str:
    """Generate HTML for the overview page content."""
    parts = []
    for sys_name, params in SYSTEM_TABS.items():
        meta = SYSTEM_META.get(sys_name, {})
        icon = meta.get("icon", "📊")
        color = meta.get("color", "#64748B")
        label = meta.get("label", sys_name)

        # Count data points for this system
        param_count = len(params)
        points_count = sum(len(session.points_for(p)) for p in params)

        # Check health
        has_alerts = False
        alert_params = []
        for param in params:
            pts = session.points_for(param)
            if pts:
                latest = pts[-1]
                kb = KB_BASELINE.get(param, {})
                kb_vals = kb.get("values", {})
                if latest["load"] in kb_vals:
                    baseline = kb_vals[latest["load"]]
                    if baseline != 0:
                        dev_pct = abs((latest["value"] - baseline) / baseline * 100)
                        if dev_pct > 5:
                            has_alerts = True
                            alert_params.append(param)

        alert_class = "alert" if has_alerts else "normal"
        alert_text = f"{len(alert_params)} 参数偏离" if has_alerts else "正常"

        parts.append(f'''
        <div class="overview-card" style="border-left: 4px solid {color};">
            <div class="overview-header">
                <span class="overview-icon">{icon}</span>
                <span class="overview-title">{label}</span>
                <span class="overview-status {alert_class}">{alert_text}</span>
            </div>
            <div class="overview-body">
                <div class="overview-stat">
                    <span class="stat-num">{param_count}</span>
                    <span class="stat-label">监测参数</span>
                </div>
                <div class="overview-stat">
                    <span class="stat-num">{points_count}</span>
                    <span class="stat-label">数据记录</span>
                </div>
            </div>
        </div>''')

    if not parts:
        return '<div class="overview-empty">暂无系统数据，请录入运行参数。</div>'
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Status Bar Builder
# ═══════════════════════════════════════════════════════════════

def build_status_bar_html() -> str:
    """Generate the top status bar HTML."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    load_val = session.latest_load()
    load_str = f"{load_val:.0f}%" if load_val is not None else "--"
    point_count = len(session.points)
    param_count = len(session.get_params())

    return f'''
    <div class="status-bar">
        <div class="status-left">
            <span class="status-brand">⚓ Marine Engine AI</span>
            <span class="status-divider">|</span>
            <span class="status-item">
                <span class="status-label">机型</span>
                <span class="status-value">12K98ME-C7</span>
            </span>
            <span class="status-divider">|</span>
            <span class="status-item">
                <span class="status-label">当前负载</span>
                <span class="status-value load-indicator">{load_str}</span>
            </span>
        </div>
        <div class="status-right">
            <span class="status-item">
                <span class="status-label">数据点</span>
                <span class="status-value">{point_count}</span>
            </span>
            <span class="status-divider">|</span>
            <span class="status-item">
                <span class="status-label">参数</span>
                <span class="status-value">{param_count}/18</span>
            </span>
            <span class="status-divider">|</span>
            <span class="status-item">
                <span class="status-label">时间</span>
                <span class="status-value">{now}</span>
            </span>
            <span class="status-divider">|</span>
            <span class="status-item">
                <span class="status-dot-green"></span>
                <span class="status-value">已连接</span>
            </span>
        </div>
    </div>'''


# ═══════════════════════════════════════════════════════════════
# Main Chat + Visualization Pipeline
# ═══════════════════════════════════════════════════════════════

def process_chat_and_viz(
    query: str, history: list[dict],
    selected_system: str, highlight_load: int, time_range: str,
):
    """
    Process user message:
    1. Classify intent → KB retrieval → LLM response (chat)
    2. DSR1 extract data → update session → regenerate chart
    """
    if not query.strip():
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=480,
        )
        yield (
            history,
            empty_fig,
            build_status_bar_html(),
            session.to_markdown(),
        )
        return

    history = history or []

    # ── Chat pipeline ──
    intent = classify_intent(query)
    kb_key = INTENT_TO_KB_KEY.get(intent)
    retriever = get_retriever()
    context = ""
    if kb_key:
        context = retriever.retrieve_for_intent(query, kb_key, top_k=5)

    prompt_template = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["其他"])
    user_prompt = prompt_template.format(query=query, context=context)

    if intent in DATA_HEAVY_INTENTS:
        use_dsv4, use_school = True, False
        model_label = "DS V4 Pro"
    elif intent in KB_INTENTS:
        use_dsv4, use_school = False, True
        model_label = "DS V3"
    else:
        use_dsv4, use_school = False, False
        model_label = "DSR1 闲聊"

    full_response = ""
    if use_dsv4:
        gen = call_dsv4_stream(MAIN_AGENT_PROMPT, user_prompt)
    elif use_school:
        gen = call_school_llm_stream(MAIN_AGENT_PROMPT, user_prompt)
    else:
        gen = call_dsr1_chat_stream(MAIN_AGENT_PROMPT, user_prompt)

    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": ""})

    # ── Interim chart (show current state during streaming) ──
    params = SYSTEM_TABS.get(selected_system, [])
    interim_fig = build_enhanced_chart(
        selected_system, params, session, highlight_load, time_range,
    )

    for chunk in gen:
        full_response += chunk
        history[-1] = {
            "role": "assistant",
            "content": f"**「{intent}」· {model_label}**\n\n{full_response}",
        }
        yield (
            history,
            interim_fig,
            build_status_bar_html(),
            session.to_markdown(),
        )

    # ── Data extraction ──
    extracted = []
    if intent in DATA_HEAVY_INTENTS:
        try:
            points = extract_data_from_message(query)
            if points:
                for pt in points:
                    session.add(pt["load"], pt["name"], pt["value"], query)
                extracted = points
        except Exception as e:
            print(f"[Extract] Error: {e}")

    # ── Final chart ──
    try:
        final_fig = build_enhanced_chart(
            selected_system, params, session, highlight_load, time_range,
        )
    except Exception as e:
        print(f"[Chart] Build error: {e}")
        final_fig = interim_fig

    if extracted:
        try:
            items = [f"{pt['name']}={pt['value']}(@{pt['load']}%)" for pt in extracted]
            note = f"\n\n📊 *已记录：{', '.join(items)}*"
            history[-1] = {"role": "assistant", "content": history[-1]["content"] + note}
        except Exception:
            pass

    yield (
        history,
        final_fig,
        build_status_bar_html(),
        session.to_markdown(),
    )


# ═══════════════════════════════════════════════════════════════
# Image Understanding Pipeline (DSR1 doubao multimodal)
# ═══════════════════════════════════════════════════════════════

IMAGE_SYSTEM_PROMPT = """你是"轮机智脑"的负载参数分析助手，专门处理轮机仪表盘、监控面板图片，核心任务是对比知识库基线数据判断设备负载状况。

当用户上传一张轮机相关图片时，请严格按以下流程处理：
1. 首先提取图片中所有可见的仪表读数和运行参数：负载率(%)、温度(℃)、压力(bar)、转速(rpm)、流量等。
2. 若系统提供的"知识库参考资料"中包含对应机型的基线数据，必须将提取到的实际读数与基线值逐一对比，判断是否在正常范围内。
3. 对于偏离基线的参数，给出异常程度判定和可能原因，引用知识库中的标准值作为判断依据。
4. 若图片信息不足以完整判断，列出已提取参数和缺失参数，指明哪些信息需要补充才能给出完整负载评估。

回答要求：
- 使用中文，条理清晰，必须使用 Markdown 小标题与列表。
- **优先引用知识库基线数据**，无知识库参考时标注"无基线对照，以下为通用经验判断"。
- 不确定处明确标注"需进一步确认"，避免无依据的猜测。
- 控制在合理篇幅内，重点突出。

【强制要求】在回答的最后，必须以如下格式输出一个 JSON 代码块，包含从图片中提取到的所有参数及其对应负载点。如果图片包含多个负载点的数据（如表格、多列数据），请提取为多个 points；如果只有一个负载点，也包在数组中。JSON代码块与前面的文字分析用空行隔开。

```json
{"points": [{"load": 75, "params": [{"name": "排气温度", "value": 280.5}, {"name": "扫气压力", "value": 1.85}]}]}
```

规则：
- load 为负载百分比数字，取 25, 50, 75, 90, 100, 110 中最近似值
- params 中的 name 使用以下可用参数列表中的中文名称（精确匹配）：排气温度、涡轮前排气温度、涡轮后排气温度、淡水进水温度、缸套水出水温度、冷却淡水出水温度、活塞冷却油出口温度、涡轮滑油进口温度、涡轮滑油出口温度、扫气温度、扫气接收温度、扫气压力、最大爆发压力、压缩压力、增压器转速、增压器空气出口温度、燃油消耗率(实测)、燃油消耗率(修正)
- value 为数值，不含单位
- 如果图片中确实无任何可读参数，points 置为空数组 []"""

def analyze_image_stream(image_path: str, user_question: str = ""):
    """Use school relay DSR1 (doubao-2.0-pro) multimodal model to analyze an uploaded image."""
    if not image_path:
        yield "⚠️ 请先上传一张图片。"
        return

    import base64

    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        yield f"❌ 图片读取失败: {e}"
        return

    # Infer mime type
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp"}
    mime = mime_map.get(ext, "image/jpeg")

    user_content = user_question.strip() or "请分析这张轮机相关图片，提取关键信息并给出专业解读。"

    # ── KB retrieval for image analysis ──
    # 图片分析场景下不依赖 classify_intent（分类不稳定），直接多KB联合检索
    IMAGE_KB_NAMES = ["温度监测", "负载指数", "查询知识库", "维护知识库"]
    kb_context = ""
    try:
        retriever = get_retriever()
        if retriever:
            # 图片分析用强化关键词：把用户问题拼接上轮机温度负载核心词，确保命中基线表格
            # 图片分析默认强制注入轮机核心关键词，不依赖用户输入
            base_kw = "负载 淡水进水 空冷器 温度 排气 推力瓦 滑油"
            extra = user_question.strip() if user_question.strip() else ""
            search_query = f"{base_kw} {extra}".strip()
            # 带预算的检索：去重 + 控制注入 prompt 的总字符量，避免 12×2000=24k 字符稀释上下文
            kb_context = retriever.retrieve_for_query(
                search_query, kb_names=IMAGE_KB_NAMES, top_k=12, max_chars=6000, with_source=True,
            )
            if kb_context:
                kb_context = "## 知识库参考资料\n" + kb_context
                print(f"[KB] 图片分析多KB联合检索命中 {len(kb_context)} 字符")
    except Exception as e:
        print(f"[KB] 图片分析知识库检索失败: {e}")

    # KB 数据注入 user message（与聊天管线一致），system prompt 保持不变
    if kb_context:
        user_content = kb_context + "\n\n---\n\n请根据以上知识库参考数据，分析以下图片：\n" + user_content
        # DEBUG: 写日志验证 KB 是否注入（相对路径，保证可移植；写入失败不影响主流程）
        try:
            with open(Path(__file__).parent / "viz_debug.log", "w", encoding="utf-8") as _f:
                _f.write(f"kb_context length: {len(kb_context)}\n")
                _f.write(f"kb_context preview: {kb_context[:500]}\n")
                _f.write(f"user_content length: {len(user_content)}\n")
        except Exception as _e:
            print(f"[KB] 写调试日志失败（忽略）: {_e}")

    try:
        stream = dsr1_client.chat.completions.create(
            model=DSR1_MODEL,
            temperature=0.6,
            max_tokens=2048,
            stream=True,
            messages=[
                {"role": "system", "content": IMAGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                        {"type": "text", "text": user_content},
                    ],
                },
            ],
        )
        first = True
        for chunk in stream:
            if chunk.choices[0].delta.content:
                if first:
                    yield "**「图片识别」· DSR1 学校中转**\n\n" + chunk.choices[0].delta.content
                    first = False
                else:
                    yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"❌ 图片分析失败: {e}"


# ═══════════════════════════════════════════════════════════════
# Image Analysis Visualization
# ═══════════════════════════════════════════════════════════════


def _extract_json_from_image_response(llm_response: str) -> dict | None:
    """Extract structured JSON from DSR1 image analysis response.
    Supports ```json code blocks and raw JSON objects. Returns parsed dict or None."""
    if not llm_response:
        return None
    raw = llm_response.strip()

    # ── Try fenced code block ──
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        block = m.group(1).strip()
    else:
        # ── Fallback: find outermost JSON object with load/points ──
        # Search for {"load" or {"points" pattern, then find its opening brace
        m = re.search(r'\{[^}]*"(?:load|points|params)"', raw)
        if not m:
            # Try finding just a {"points" or "load": anywhere
            m2 = re.search(r'"(?:points|load)"\s*:', raw)
            if not m2:
                return None
            # Scan backwards from this match to find the opening {
            pos = raw.rfind("{", 0, m2.start())
            if pos == -1:
                return None
        else:
            pos = m.start()

        depth = 0
        end = -1
        for i, ch in enumerate(raw[pos:], pos):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            return None
        block = raw[pos:end]

    # ── Parse JSON ──
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None

    # Validate structure
    if not isinstance(data, dict):
        return None
    if "points" in data or "load" in data or "params" in data:
        return data
    return None


def _normalize_image_params(raw_points: list[dict]) -> list[dict]:
    """Map DSR1-extracted parameter names to canonical KB names via ALIAS_TO_PARAM.
    Returns list of {"load": float, "name": str, "value": float} dicts."""
    result = []
    for pt in raw_points:
        load = pt.get("load")
        if load is None:
            continue
        try:
            load = float(load)
        except (TypeError, ValueError):
            continue
        for p in pt.get("params", []):
            name = p.get("name", "")
            value = p.get("value")
            if not name or value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            # Try exact alias match
            matched = ALIAS_TO_PARAM.get(name)
            if not matched:
                # Try substring matching
                for alias, canonical in ALIAS_TO_PARAM.items():
                    if alias in name or name in alias:
                        matched = canonical
                        break
            if matched and matched in KB_BASELINE:
                result.append({"load": load, "name": matched, "value": value})
    return result


def build_image_analysis_chart(extracted_points: list[dict]) -> go.Figure | None:
    """Build a Plotly line chart from image-extracted parameter data,
    styled like build_enhanced_chart with KB baselines and tolerance bands.
    Supports single or multiple load points. Returns None if no valid data."""
    if not extracted_points:
        return None

    # ── Group by parameter ──
    param_groups: dict[str, list[tuple[float, float]]] = {}
    for pt in extracted_points:
        param_groups.setdefault(pt["name"], []).append((pt["load"], pt["value"]))

    fig = go.Figure()
    has_data = False

    for i, (param, pts) in enumerate(param_groups.items()):
        kb = KB_BASELINE.get(param)
        if not kb:
            continue
        color = COLORS[i % len(COLORS)]
        unit = kb.get("unit", "")
        tolerance = kb.get("tolerance", 0)
        kb_values = kb.get("values", {})

        # ── KB Baseline (dashed) ──
        if kb_values:
            x_kb = list(kb_values.keys())
            y_kb = list(kb_values.values())
            fig.add_trace(go.Scatter(
                x=x_kb, y=y_kb, mode="lines+markers",
                name=f"{param} [KB基准]",
                line=dict(dash="dot", width=1.5, color=color),
                marker=dict(size=5, symbol="cross-thin", color=color),
                opacity=0.40,
                legendgroup=param,
                hovertemplate=f"<b>KB基准</b> {param}<br>值: %{{y:.1f}}{unit}<br>负载: %{{x}}%<extra></extra>",
            ))

            # ── Tolerance band ──
            y_upper = [v + tolerance for v in y_kb]
            y_lower = [v - tolerance for v in y_kb]
            fig.add_trace(go.Scatter(
                x=x_kb + x_kb[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.10),
                line=dict(width=0),
                name=f"{param} ±{tolerance}{unit}",
                showlegend=False,
                legendgroup=param,
                hoverinfo="skip",
            ))

        # ── Extracted data points ──
        pts_sorted = sorted(pts, key=lambda x: x[0])
        x_act = [p[0] for p in pts_sorted]
        y_act = [p[1] for p in pts_sorted]

        # Color markers by deviation from KB baseline
        out_colors = []
        for lx, vy in zip(x_act, y_act):
            if lx in kb_values:
                lo = kb_values[lx] - tolerance
                hi = kb_values[lx] + tolerance
                if vy < lo or vy > hi:
                    # 判定一律用绝对偏差（与知识库容差同单位），不用百分比
                    abs_dev = abs(vy - kb_values[lx])
                    if abs_dev > tolerance * 1.5:
                        out_colors.append("#ef4444")  # red — severe
                    else:
                        out_colors.append("#f59e0b")  # yellow — warning
                else:
                    out_colors.append("#22c55e")  # green — normal
            else:
                out_colors.append(color)

        mode = "lines+markers+text" if len(pts_sorted) >= 1 else "markers+text"
        marker_size = 14 if len(pts_sorted) == 1 else 11

        fig.add_trace(go.Scatter(
            x=x_act, y=y_act,
            mode=mode,
            name=f"{param} [图片提取]",
            line=dict(width=2.8, color=color, shape="spline", smoothing=0.4),
            marker=dict(
                size=marker_size, symbol="circle",
                color=out_colors,
                line=dict(width=1.5, color="#FFFFFF"),
            ),
            text=[f"{v:.1f}" for v in y_act],
            textposition="top center",
            textfont=dict(color="#1D1D1F", size=10, family="Inter, sans-serif"),
            legendgroup=param,
            hovertemplate=f"<b>图片提取</b> {param}<br>值: %{{y:.1f}}{unit}<br>负载: %{{x}}%<extra></extra>",
        ))
        has_data = True

    if not has_data:
        return None

    # ── Layout (dark theme matching dashboard) ──
    fig.update_layout(
        xaxis=dict(
            title=dict(text="负载 (%)", font=dict(color="#8899AA", size=13)),
            tickvals=LOADS,
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            tickfont=dict(color="#8899AA"),
        ),
        yaxis=dict(
            title=dict(text="参数值", font=dict(color="#8899AA", size=13)),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            tickfont=dict(color="#8899AA"),
        ),
        margin=dict(l=55, r=30, t=50, b=55),
        height=480,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(color="#8899AA", size=10),
            bgcolor="rgba(12, 28, 48, 0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(12, 28, 48, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0C1C30",
            font=dict(color="#F0F3F5", size=12, family="Inter, sans-serif"),
            bordercolor="rgba(0,191,165,0.3)",
        ),
        font=dict(family="Inter, Noto Sans SC, sans-serif"),
        title=dict(
            text="🖼️ 图片分析 · 参数趋势对比",
            font=dict(color="#1D1D1F", size=16),
            x=0.5,
            y=0.98,
        ),
    )
    fig.update_xaxes(tickmode="array", tickvals=LOADS)

    return fig


SECONDARY_EXTRACT_PROMPT = """你是轮机传感器数据提取器。从下方分析文字中提取所有参数名、负载点、数值。

输出纯JSON（不要markdown）：
{"points": [{"load": 25, "params": [{"name": "淡水进水温度", "value": 17}]}, {"load": 50, "params": [{"name": "淡水进水温度", "value": 20}]}]}

规则：
1. 每个负载点一个对象，包含该负载下的所有参数
2. 参数名使用以下标准名称之一：排气温度、涡轮前排气温度、涡轮后排气温度、淡水进水温度、缸套水出水温度、冷却淡水出水温度、活塞冷却油出口温度、涡轮滑油进口温度、涡轮滑油出口温度、扫气温度、扫气接收温度、扫气压力、最大爆发压力、压缩压力、增压器转速、增压器空气出口温度、燃油消耗率(实测)、燃油消耗率(修正)
3. 数值不含单位
4. 如果分析文字中有箭头序列（如"25%→50%→75%...17℃→20℃→23℃"），请对应提取每个负载-数值对
5. 只提取文字中明确提到的参数和数值"""


def _extract_points_from_analysis_text(analysis_text: str) -> list[dict] | None:
    """Call DeepSeek V3 to extract structured {load, name, value} points from DSR1 analysis text.
    Falls back to regex sequence parsing if API call fails."""
    if not analysis_text:
        return None

    # ── Primary: DeepSeek V3 extraction ──
    try:
        resp = school_client.chat.completions.create(
            model=SCHOOL_MODEL,
            messages=[
                {"role": "system", "content": SECONDARY_EXTRACT_PROMPT},
                {"role": "user", "content": analysis_text[:4000]},
            ],
            temperature=0.1, max_tokens=1024, timeout=20,
        )
        raw = resp.choices[0].message.content or ""
        print(f"[SecondaryExtract] Raw ({len(raw)} chars): {raw[:300]}")

        # Parse JSON from response
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = re.sub(r'^```(?:json)?\s*\n?', '', raw_clean)
            raw_clean = re.sub(r'\n?```\s*$', '', raw_clean)

        # Try direct parse
        try:
            data = json.loads(raw_clean)
        except json.JSONDecodeError:
            # Try balanced-brace extraction
            start = raw_clean.find('{')
            if start == -1:
                raise
            depth, end = 0, -1
            for i, ch in enumerate(raw_clean[start:], start):
                if ch == '{': depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end == -1:
                raise
            data = json.loads(raw_clean[start:end])

        points_data = data.get("points", [])
        if not points_data:
            load_val = data.get("load")
            params_val = data.get("params", [])
            if params_val and load_val is not None:
                points_data = [{"load": load_val, "params": params_val}]

        if points_data:
            normalized = _normalize_image_params(points_data)
            if normalized:
                print(f"[SecondaryExtract] V3 extracted {len(normalized)} points")
                return normalized
    except Exception as e:
        print(f"[SecondaryExtract] V3 failed: {e}")

    # ── Fallback: regex sequence parsing ──
    return _parse_arrow_sequences(analysis_text)


def _parse_arrow_sequences(text: str) -> list[dict] | None:
    """Parse arrow-separated sequences like '25%→50%→75%' mapping to '17℃→20℃→23℃'."""
    # Find load sequence: 负载变化序列：25%→50%→75%→90%→100%
    load_seq = re.search(
        r'负载[变序].*?[：:\s]*((?:\d+(?:\.\d+)?\s*%?\s*→\s*)+\d+(?:\.\d+)?)\s*%?',
        text
    )
    if not load_seq:
        return None

    loads = [float(x.strip().rstrip('%')) for x in load_seq.group(1).split('→')]
    print(f"[SeqParse] Loads: {loads}")

    # Find parameter sequences: 淡水进水温度序列：17℃→20℃→23℃→27℃→31℃
    param_seqs = re.findall(
        r'([一-鿿]{2,}(?:温度|压力|转速|率|流量|度)[一-鿿()]*?(?:序列)?)[：:\s]*((?:\d+(?:\.\d+)?\s*(?:℃|%|bar|rpm|°C|度)?\s*→\s*)+\d+(?:\.\d+)?)',
        text
    )

    result = []
    for raw_name, values_str in param_seqs:
        name = raw_name.replace('序列', '').strip()
        if '负载' in name:
            continue

        # Match canonical name
        matched = ALIAS_TO_PARAM.get(name)
        if not matched:
            for alias, canonical in ALIAS_TO_PARAM.items():
                if alias in name or name in alias:
                    matched = canonical
                    break
        if not matched or matched not in KB_BASELINE:
            continue

        values = [float(v.strip().rstrip('℃%barpm°C度')) for v in values_str.split('→')]

        # Zip loads with values
        for load, val in zip(loads, values):
            result.append({"load": load, "name": matched, "value": val})

        print(f"[SeqParse] {matched}: {dict(zip(loads, values))}")

    return result if result else None


def generate_image_analysis_viz(llm_response: str):
    """Parse LLM image analysis response, generate Plotly baseline comparison chart
    styled to match the real-time trend chart."""
    import numpy as np

    # ── Parse extracted params from LLM response ──
    # Standard format: 淡水进水温度: 18℃
    param_pattern = re.compile(
        r"[-*]?\s*(?:[a-zA-Z\u4ee00-\u9fff]+[：:])?\s*([\u4e00-\u9fff]+(?:\([^)]+\))?)[：:]\s*([\d.]+)\s*(?:℃|%|bar|rpm|°C|度)"
    )
    loose_pattern = re.compile(
        r"([\u4e00-\u9fff]{2,}(?:温度|压力|转速|率|流量)[\u4e00-\u9fff]*(?:\([^)]+\))?)[：:\s]+([\d.]+)\s*(?:℃|%|bar|rpm|°C|度)"
    )
    flexible_pattern = re.compile(
        r"([\u4e00-\u9fff]{2,}(?:温度|压力|转速|率|流量|度)[\u4e00-\u9fff()]*)[是为约]?(\d+(?:\.\d+)?)\s*(?:℃|%|bar|rpm|°C|度)"
    )
    # Arrow-separated sequence: 淡水进水温度实测值：18℃→20℃→23℃
    seq_pattern = re.compile(
        r"([\u4e00-\u9fff]{2,}(?:温度|压力|转速|率|流量|度)[\u4e00-\u9fff()]*(?:[值线]))[：:\s]*(\d+(?:\.\d+)?)\s*(?:℃|%|bar|rpm|°C|度)\s*→"
    )
    all_matches = {}
    for m in (param_pattern.findall(llm_response) +
              loose_pattern.findall(llm_response) +
              flexible_pattern.findall(llm_response) +
              seq_pattern.findall(llm_response)):
        name = m[0].strip()
        if "负载" in name or "序列" in name:
            continue
        all_matches[name] = float(m[1])

    # Map to canonical names
    extracted = {}
    for raw_name, value in all_matches.items():
        for canonical, aliases in PARAM_ALIASES.items():
            for alias in aliases:
                if alias in raw_name or raw_name in alias:
                    extracted[canonical] = value
                    break
            if canonical in extracted:
                break

    load_match = re.search(r"负载[率]?[：:\s]*(\d+)\s*%", llm_response)
    if not load_match:
        # Try sequence format: 负载变化序列：25%→50%→...
        load_match = re.search(r"负载[变序].*?[：:\s]*(\d+)\s*%\s*→", llm_response)
    load_level = int(load_match.group(1)) if load_match else 25

    # ── Build full-parameter reference data ──
    SYSTEM_ORDER = ["冷却", "排气", "滑油", "扫气", "燃烧", "增压器", "油耗"]
    system_colors = {
        "冷却": "#38bdf8", "排气": "#fb923c", "滑油": "#c084fc",
        "扫气": "#4ade80", "燃烧": "#facc15", "增压器": "#60a5fa", "油耗": "#f87171",
    }
    system_bg = {
        "冷却": "rgba(56,189,248,0.12)",
        "排气": "rgba(251,146,60,0.10)",
        "滑油": "rgba(192,132,252,0.10)",
        "扫气": "rgba(74,222,128,0.10)",
        "燃烧": "rgba(250,204,21,0.10)",
        "增压器": "rgba(96,165,250,0.10)",
        "油耗": "rgba(248,113,113,0.12)",
    }

    rows = []  # (system, param, actual, baseline, unit, tolerance)
    # Only include detected params, grouped by system order
    for system in SYSTEM_ORDER:
        for param, kb in KB_BASELINE.items():
            if kb.get("system") == system and load_level in kb.get("values", {}):
                actual = extracted.get(param)
                # Only include params that were actually detected
                if actual is not None:
                    baseline = kb["values"][load_level]
                    rows.append((system, param, actual, baseline, kb["unit"], kb.get("tolerance", 5)))

    # Debug: write to log file
    log_path = Path(__file__).parent / "viz_debug.log"
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"load_level={load_level}\nextracted={list(extracted.keys())}\nllm_response=\n{llm_response[:500]}\n")
    if not rows:
        return None

    n = len(rows)
    fig_height = max(420, n * 38)

    y_labels = []
    actual_vals = []
    baseline_vals = []
    tol_vals = []
    dev_pcts = []
    colors = []
    sys_names = []

    for sys_name, param, actual, baseline, unit, tol in rows:
        short = param.replace("(实测)", "").replace("(修正)", "")
        y_labels.append(f"{short} ({unit})")
        actual_vals.append(actual)
        baseline_vals.append(baseline)
        tol_vals.append(tol)
        sys_names.append(sys_name)
        if actual is not None:
            dev = (actual - baseline) / baseline * 100
            dev_pcts.append(dev)
            abs_dev = abs(dev)
            if abs_dev > tol * 1.5:
                colors.append("#ef4444")
            elif abs_dev > tol:
                colors.append("#f59e0b")
            else:
                colors.append("#22c55e")
        else:
            dev_pcts.append(None)
            colors.append("#475569")

    # ═══════════════════════════════════
    # Build Plotly figure (optimized for fast kaleido render)
    # ═══════════════════════════════════
    fig = go.Figure()

    y_idx = list(range(n))

    # ── Tolerance band (single thick-line trace per param, much faster than shapes) ──
    for i in range(n):
        lo = baseline_vals[i] - tol_vals[i]
        hi = baseline_vals[i] + tol_vals[i]
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[i, i],
            mode='lines',
            line=dict(width=16, color='rgba(34,197,94,0.16)'),
            showlegend=(i == 0),
            name='容差范围',
            legendgroup='tolerance',
            hoverinfo='skip',
        ))

    # ── Baseline reference dots ──
    fig.add_trace(go.Scatter(
        x=baseline_vals,
        y=y_idx,
        mode="markers",
        name="基线值",
        marker=dict(
            size=10, symbol="diamond-tall",
            color="#64748b",
            line=dict(width=1.5, color="#94a3b8"),
        ),
        hovertemplate="<b>基线</b> %{x:.1f}<extra></extra>",
    ))

    # ── Actual value dots (colored by severity) ──
    valid_idx = [i for i, a in enumerate(actual_vals) if a is not None]
    valid_x = [actual_vals[i] for i in valid_idx]
    valid_y = [y_idx[i] for i in valid_idx]
    valid_c = [colors[i] for i in valid_idx]
    valid_dev = [dev_pcts[i] for i in valid_idx]
    valid_labels_text = [y_labels[i] for i in valid_idx]

    fig.add_trace(go.Scatter(
        x=valid_x, y=valid_y,
        mode="markers+text",
        name="实测值",
        marker=dict(
            size=16, symbol="circle",
            color=valid_c,
            line=dict(width=2.5, color="#ffffff"),
        ),
        text=[f"{x:.1f}" for x in valid_x],
        textposition="middle right",
        textfont=dict(color="#f1f5f9", size=11, family="Inter, Segoe UI, sans-serif"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "实测: %{x:.1f}<br>"
            "偏差: %{customdata[1]:+.1f}%<br>"
            "<extra></extra>"
        ),
        customdata=list(zip(valid_labels_text, valid_dev)),
    ))

    # ── Missing param markers ──
    missing_idx = [i for i, a in enumerate(actual_vals) if a is None]
    if missing_idx:
        fig.add_trace(go.Scatter(
            x=[baseline_vals[i] for i in missing_idx],
            y=[y_idx[i] for i in missing_idx],
            mode="text",
            name="未检测",
            text=["?"] * len(missing_idx),
            textfont=dict(color="#ef4444", size=16, family="Inter, sans-serif"),
            hoverinfo="skip",
        ))

    # ── System label separators ──
    prev_sys = None
    for i, sys_name in enumerate(sys_names):
        if sys_name != prev_sys:
            prev_sys = sys_name
            fig.add_hline(
                y=i - 0.5,
                line=dict(color="rgba(148,163,184,0.25)", width=1, dash="dot"),
                layer="below",
            )

    # ═══════════════════════════════════
    # Layout (matches real-time trend chart)
    # ═══════════════════════════════════
    fig.update_layout(
        xaxis=dict(
            title=dict(text="参数值", font=dict(color="#94a3b8", size=13)),
            gridcolor="rgba(51,65,85,0.35)",
            zeroline=False,
            tickfont=dict(color="#94a3b8", size=11),
            side="top",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_idx,
            ticktext=y_labels,
            tickfont=dict(color="#cbd5e1", size=10.5),
            gridcolor="rgba(51,65,85,0.25)",
            zeroline=False,
            autorange="reversed",
        ),
        margin=dict(l=200, r=80, t=70, b=30),
        height=fig_height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=11),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        plot_bgcolor="rgba(15,23,42,0.4)",
        paper_bgcolor="#0f1117",
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#1e293b",
            font=dict(color="#f1f5f9", size=12, family="Inter, sans-serif"),
            bordercolor="#E5E5EA",
        ),
        font=dict(family="Inter, Segoe UI, sans-serif"),
        title=dict(
            text=f"⚡ 负载 {load_level}% · 全参数基线对比 · 可视化分析",
            font=dict(color="#f1f5f9", size=16),
            x=0.5,
            y=0.98,
        ),
    )

    return fig



# ═══════════════════════════════════════════════════════════════
# Chart-only refresh (for system switch / load / time changes)
# ═══════════════════════════════════════════════════════════════

def refresh_chart_and_cards(
    selected_system: str,
    highlight_load: int,
    time_range: str,
):
    """Refresh chart and cards without re-running chat pipeline."""
    params = SYSTEM_TABS.get(selected_system, [])
    fig = build_enhanced_chart(
        selected_system, params, session, highlight_load, time_range,
    )
    return (
        fig,
        build_status_bar_html(),
        session.to_markdown(),
    )


# ═══════════════════════════════════════════════════════════════
# Export Functions
# ═══════════════════════════════════════════════════════════════

def export_csv() -> str:
    """Export session data to CSV in the output directory."""
    if not session.points:
        return "⚠️ 暂无数据可导出"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"marine_data_{ts}.csv"

    lines = ["时间,参数,负载(%),实测值,单位,KB基准值,偏差,偏差%,是否异常"]
    for pt in session.points:
        param = pt["param"]
        kb = KB_BASELINE.get(param, {})
        unit = kb.get("unit", "")
        kb_val = kb.get("values", {}).get(pt["load"], "")
        dev = ""
        dev_pct = ""
        abnormal = ""
        if kb_val:
            dev = f"{pt['value'] - kb_val:.2f}"
            dev_pct = f"{(pt['value'] - kb_val) / kb_val * 100:.1f}"
            tol = kb.get("tolerance", 0)
            abnormal = "是" if abs(pt["value"] - kb_val) > tol else "否"
        lines.append(
            f"{pt['time']},{param},{pt['load']},{pt['value']},{unit},"
            f"{kb_val},{dev},{dev_pct},{abnormal}"
        )

    filepath.write_text("\n".join(lines), encoding="utf-8-sig")
    return f"✅ 已导出: {filepath.name}\n共 {len(session.points)} 条记录"


def export_png(selected_system: str, highlight_load: int, time_range: str) -> str:
    """Export current chart as PNG."""
    params = SYSTEM_TABS.get(selected_system, [])
    fig = build_enhanced_chart(selected_system, params, session, highlight_load, time_range)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"chart_{selected_system}_{ts}.png"
    try:
        fig.write_image(str(filepath), width=1200, height=600, scale=2)
        return f"✅ 图表已导出: {filepath.name}"
    except Exception as e:
        return f"❌ PNG导出失败: {e}\n请确保已安装 kaleido: pip install kaleido"


# ═══════════════════════════════════════════════════════════════
# History Save / Load (unchanged)
# ═══════════════════════════════════════════════════════════════

HISTORY_DIR = Path(__file__).parent / "history_data"
HISTORY_DIR.mkdir(exist_ok=True)

def _safe_history_path(filename: str) -> Path | None:
    """Resolve a history filename safely to a path inside HISTORY_DIR.

    拒绝路径穿越：文件名必须是纯文件名（不含目录分隔符、不含 `..`），
    且解析后的绝对路径必须仍落在 history_data 目录内。非法返回 None。
    """
    if not filename or not isinstance(filename, str):
        return None
    # 只允许纯文件名：去掉目录成分后必须与原串一致（拦截 ../、绝对路径等）
    if Path(filename).name != filename:
        return None
    filepath = HISTORY_DIR / filename
    # 双保险：解析后的绝对路径必须仍在 HISTORY_DIR 内
    try:
        filepath.resolve().relative_to(HISTORY_DIR.resolve())
    except ValueError:
        return None
    return filepath

def save_session(label: str = "") -> str:
    if not session.points:
        return ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{label or '未命名'}.json"
    filepath = HISTORY_DIR / filename
    # 深拷贝并序列化 datetime 字段
    pts_copy = []
    for p in session.points:
        pc = dict(p)
        if "dt" in pc and isinstance(pc["dt"], datetime):
            pc["dt"] = pc["dt"].isoformat()
        pts_copy.append(pc)
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label or "未命名",
        "type": "text",
        "points": pts_copy,
        "created": datetime.now().isoformat(),
    }
    HISTORY_DIR.mkdir(exist_ok=True)
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return filename


def _sanitize_content(content):
    """Remove non-JSON-serializable objects (e.g. gr.Plot) from message content.
    Handles Gradio 6.x format where content is list of {text, type} dicts,
    as well as older formats (str, ChatMessage objects, gr.Plot, etc.).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Gradio 6.x format: {"text": "...", "type": "text"} or {"file": ..., "type": "file"}
                if item.get("type") == "text" and "text" in item:
                    result.append(str(item["text"]))
                elif item.get("type") == "file" and "file" in item:
                    f = item["file"]
                    result.append(str(f.get("url", f.get("path", str(f)))))
                # component type (e.g. gr.Plot) — skip silently
            elif hasattr(item, 'text'):
                # Older ChatMessage format
                result.append(str(item.text))
        return result
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        return str(content)
    if hasattr(content, 'text'):
        return str(content.text)
    return str(content)


def save_image_session(hist_data: list, label: str = "", img_path: str = "", chart_points: list | None = None) -> str:
    """Save image analysis chatbot history with type marker."""
    if not hist_data:
        return ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{label or '图片分析'}.json"
    filepath = HISTORY_DIR / filename
    clean = []
    for msg in hist_data:
        clean.append({
            "role": msg.get("role", ""),
            "content": _sanitize_content(msg.get("content", "")),
        })
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label or "图片分析",
        "type": "image",
        "messages": clean,
        "created": datetime.now().isoformat(),
    }
    if img_path:
        data["image_path"] = img_path
    if chart_points:
        data["chart_points"] = chart_points
    HISTORY_DIR.mkdir(exist_ok=True)
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return filename


def load_session_data(filename: str) -> str:
    filepath = _safe_history_path(filename)
    if filepath is None or not filepath.exists():
        return f"文件不存在: {filename}"
    data = json.loads(filepath.read_text(encoding="utf-8"))
    session.clear()
    for pt in data.get("points", []):
        session.add(pt["load"], pt["param"], pt["value"], pt.get("query", ""))
    return session.to_markdown()

def list_saved_sessions(type_filter: str = "all") -> list[str]:
    files = sorted(HISTORY_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if type_filter == "all":
        return [f.name for f in files]
    result = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("type", "text") == type_filter:
                result.append(f.name)
        except Exception:
            continue
    return result

def delete_session(filename: str) -> tuple:
    filepath = _safe_history_path(filename)
    if filepath is not None and filepath.exists():
        filepath.unlink()
    remaining = list_saved_sessions()
    return gr.Dropdown(choices=remaining, value=None), "\n".join(remaining) if remaining else "*暂无历史数据*"

def refresh_history_list():
    files = list_saved_sessions()
    choices = files if files else []
    md = "\n".join(f"📊 `{f}`" for f in files) if files else "*暂无历史数据*"
    return gr.Dropdown(choices=choices, value=None), md


# ═══════════════════════════════════════════════════════════════
# DARK THEME CSS
# ═══════════════════════════════════════════════════════════════

DASHBOARD_CSS = """
/* ===== 轮机智脑 — Dark Tech Dashboard ===== */
/* 深色科技风主题 · 青绿主色 + 金色点缀 · 与官网视觉统一 */

.gradio-container {
    max-width: 100% !important;
    background: #020B14 !important;
    font-family: 'Inter', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: #F0F3F5 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,191,165,0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,191,165,0.5); }

/* ============================
   DASHBOARD LAYOUT
   ============================ */
.dash-layout {
    display: flex;
    height: 100%;
    min-height: calc(100vh - 40px);
    gap: 0;
}

/* Gradio Row wrapping dash-layout: force flex children */
.dash-layout > .gr-box,
.dash-layout > div {
    display: flex !important;
}

/* ============================
   SIDEBAR
   ============================ */
.dash-sidebar {
    width: 232px;
    flex-shrink: 0;
    background: rgba(8, 21, 37, 0.9);
    border-right: 1px solid rgba(255,255,255,0.06);
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
.dash-sidebar-brand {
    padding: 20px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 10px;
}
.dash-sidebar-brand .brand-icon {
    font-size: 24px;
    color: #00BFA5;
    filter: drop-shadow(0 0 8px rgba(0,191,165,0.4));
}
.dash-sidebar-brand .brand-text {
    font-size: 17px;
    font-weight: 700;
    color: #F0F3F5;
    letter-spacing: -0.02em;
}
.dash-sidebar-nav {
    padding: 10px 0;
    flex: 1;
}
.dash-nav-section-title {
    font-size: 10.5px;
    font-weight: 600;
    color: #556677;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 14px 20px 8px 20px;
}
.dash-nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 16px;
    margin: 2px 10px;
    border-radius: 10px;
    border-left: 3px solid transparent;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 13.5px;
    color: #8899AA;
}
.dash-nav-item:hover {
    background: rgba(0,191,165,0.06);
    color: #F0F3F5;
}
.dash-nav-item.active {
    background: rgba(0,191,165,0.10);
    color: #00BFA5;
    border-left-color: #00BFA5;
    font-weight: 600;
    box-shadow: inset 0 0 20px rgba(0,191,165,0.03);
}
.dash-nav-item .nav-icon {
    font-size: 16px;
    width: 22px;
    text-align: center;
    flex-shrink: 0;
}
.dash-nav-item .nav-label {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.dash-nav-item .nav-count {
    font-size: 10.5px;
    color: #556677;
    font-weight: 500;
    flex-shrink: 0;
}
.dash-nav-item .nav-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00BFA5;
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(0,191,165,0.5);
}
.dash-nav-item .nav-alert-badge {
    font-size: 10px;
    font-weight: 700;
    background: #F0A500;
    color: #0A0A0A;
    border-radius: 10px;
    padding: 1px 7px;
    min-width: 20px;
    text-align: center;
    flex-shrink: 0;
    line-height: 16px;
    box-shadow: 0 0 10px rgba(240,165,0,0.3);
}
.dash-sidebar-footer {
    padding: 12px 10px;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.dash-sidebar-footer .dash-nav-item {
    font-size: 12.5px;
}

/* ============================
   MAIN AREA
   ============================ */
.dash-main {
    flex: 1;
    padding: 28px 32px;
    overflow-y: auto;
    background: #020B14;
}
/* ============================
   KPI CARDS
   ============================ */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: rgba(12, 28, 48, 0.65);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px 22px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,191,165,0.3), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.kpi-card:hover {
    border-color: rgba(0,191,165,0.2);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.kpi-card:hover::before {
    opacity: 1;
}
.kpi-card .kpi-label {
    font-size: 11px;
    font-weight: 600;
    color: #556677;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.kpi-card .kpi-value {
    font-size: 32px;
    font-weight: 800;
    color: #00BFA5;
    line-height: 1.1;
    letter-spacing: -0.02em;
    text-shadow: 0 0 20px rgba(0,191,165,0.15);
}
.kpi-card:nth-child(even) .kpi-value {
    color: #F0A500;
    text-shadow: 0 0 20px rgba(240,165,0,0.15);
}
.kpi-card .kpi-sub {
    font-size: 11.5px;
    color: #8899AA;
    margin-top: 6px;
}
.kpi-card .kpi-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #00BFA5;
    display: inline-block;
    margin-right: 6px;
    animation: kpi-pulse 2.5s infinite;
    box-shadow: 0 0 8px rgba(0,191,165,0.5);
}
@keyframes kpi-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.kpi-value.warn { color: #F59E0B !important; text-shadow: 0 0 20px rgba(245,158,11,0.2) !important; }
.kpi-value.ok { color: #10B981 !important; text-shadow: 0 0 20px rgba(16,185,129,0.2) !important; }
.kpi-value.muted { color: #8899AA !important; font-weight: 400 !important; text-shadow: none !important; }

/* ============================
   CONTENT ROW (Chart + Events)
   ============================ */
.dash-content-row {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 20px;
    min-height: 380px;
}
.dash-chart-area {
    background: rgba(12, 28, 48, 0.65);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
}
.chart-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
}
.chart-title {
    font-size: 15px;
    font-weight: 600;
    color: #F0F3F5;
    letter-spacing: -0.01em;
}
.chart-tabs {
    display: flex;
    gap: 4px;
    background: rgba(255,255,255,0.03);
    padding: 3px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}
.chart-tab {
    font-size: 11.5px;
    font-weight: 500;
    color: #8899AA;
    padding: 5px 14px;
    border-radius: 7px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.chart-tab:hover {
    color: #F0F3F5;
}
.chart-tab.active {
    color: #020B14;
    background: #00BFA5;
    box-shadow: 0 2px 8px rgba(0,191,165,0.25);
    font-weight: 600;
}
.chart-placeholder {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #556677;
    font-size: 13px;
    gap: 12px;
}

.dash-events-area {
    background: rgba(12, 28, 48, 0.65);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 20px 22px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
}
.events-title {
    font-size: 13px;
    font-weight: 600;
    color: #F0F3F5;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.event-item {
    display: flex;
    gap: 10px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 14px;
    align-items: flex-start;
}
.event-item:last-of-type {
    border-bottom: none;
    margin-bottom: 0;
}
.event-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.event-dot.warn { background: #F59E0B; box-shadow: 0 0 8px rgba(245,158,11,0.5); }
.event-dot.info { background: #00BFA5; box-shadow: 0 0 8px rgba(0,191,165,0.5); }
.event-dot.ok   { background: #10B981; box-shadow: 0 0 8px rgba(16,185,129,0.5); }
.event-body { flex: 1; min-width: 0; }
.event-text {
    font-size: 13px;
    color: #F0F3F5;
    font-weight: 500;
    line-height: 1.5;
}
.event-time {
    font-size: 10.5px;
    color: #556677;
    margin-top: 4px;
}
.events-footer {
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.events-footer a {
    font-size: 12px;
    color: #00BFA5;
    text-decoration: none;
    font-weight: 500;
}
.events-footer a:hover { opacity: 0.8; }
.events-empty {
    text-align: center;
    color: #556677;
    font-size: 13px;
    padding: 24px 12px;
}

/* ============================
   HIDDEN BUTTONS
   ============================ */
.hidden-buttons { display: none !important; }

/* ============================
   DETAIL PAGE
   ============================ */
#breadcrumb-row { margin-bottom: 16px !important; align-items: center !important; }
.detail-breadcrumb {
    display: flex; align-items: center; gap: 12px;
    font-size: 17px; font-weight: 600; color: #F0F3F5;
    padding: 16px 24px;
    background: rgba(12, 28, 48, 0.65);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}
.breadcrumb-icon { font-size: 22px; }
.breadcrumb-name { letter-spacing: -0.02em; }
.breadcrumb-badge {
    font-size: 11.5px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid;
    letter-spacing: 0.02em; margin-left: 6px;
}
#back-overview-btn {
    background: rgba(12, 28, 48, 0.65) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #8899AA !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(10px);
}
#back-overview-btn:hover {
    border-color: rgba(0,191,165,0.3) !important;
    color: #00BFA5 !important;
    background: rgba(0,191,165,0.06) !important;
}

/* ============================
   MAIN / CHAT / SUMMARY PANELS
   ============================ */
.panel-title {
    font-size: 12.5px; font-weight: 600; color: #8899AA;
    margin-bottom: 12px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    letter-spacing: 0.06em; text-transform: uppercase;
}

/* ============================
   SYSTEM CARDS (Sidebar detail)
   ============================ */
.system-cards-container { display: flex; flex-direction: column; gap: 8px; }
.sys-card {
    position: relative;
    background: rgba(12, 28, 48, 0.65);
    border: 1px solid rgba(255,255,255,0.10);
    border-left-width: 3px;
    border-radius: 12px;
    padding: 12px 16px;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex; align-items: center; gap: 10px;
    backdrop-filter: blur(10px);
}
.sys-card:hover {
    border-color: rgba(0,191,165,0.2);
    transform: translateX(4px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
}
.sys-card.active {
    border-color: #00BFA5;
    background: rgba(0,191,165,0.08);
    box-shadow: 0 0 0 3px rgba(0,191,165,0.08);
}
.card-header { display: flex; align-items: center; gap: 8px; }
.card-icon { font-size: 18px; }
.data-dot { width: 7px; height: 7px; border-radius: 50%; background: rgba(255,255,255,0.15); display: inline-block; }
.data-dot.active-dot { background: #00BFA5; box-shadow: 0 0 8px rgba(0,191,165,0.5); }
.card-body { display: flex; flex-direction: column; gap: 2px; flex: 1; }
.card-label { font-size: 13px; font-weight: 600; color: #F0F3F5; }
.deviation {
    font-size: 10.5px; font-weight: 600; font-variant-numeric: tabular-nums;
    padding: 2px 7px; border-radius: 6px; display: inline-block; width: fit-content;
}
.dev-ok { color: #10B981; background: rgba(16,185,129,0.1); }
.dev-warn { color: #F59E0B; background: rgba(245,158,11,0.1); }
.dev-alert { color: #EF4444; background: rgba(239,68,68,0.1); }
.dev-none { color: #556677; }

/* ============================
   GRADIO OVERRIDES
   ============================ */
.gr-button-primary {
    background: linear-gradient(135deg, #F0A500 0%, #E8960A 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    color: #0A0A0A !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(240,165,0,0.2) !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(240,165,0,0.35) !important;
}
.gr-button {
    background: rgba(12, 28, 48, 0.65) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #F0F3F5 !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.gr-button:hover {
    background: rgba(0,191,165,0.06) !important;
    border-color: rgba(0,191,165,0.25) !important;
    color: #00BFA5 !important;
}
.gr-textbox textarea, .gr-textbox input {
    background: rgba(12, 28, 48, 0.65) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #F0F3F5 !important;
    border-radius: 10px !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #00BFA5 !important;
    box-shadow: 0 0 0 3px rgba(0,191,165,0.1) !important;
    outline: none !important;
}
.gr-slider input[type="range"] { accent-color: #00BFA5; }
.gr-dropdown {
    background: rgba(12, 28, 48, 0.65) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    color: #F0F3F5 !important;
}
.gr-chatbot {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    background: rgba(12, 28, 48, 0.4) !important;
}
.gr-chatbot .message.user {
    background: linear-gradient(135deg, #00BFA5 0%, #00A896 100%) !important;
    color: #0A0A0A !important;
    border-radius: 14px 14px 4px 14px !important;
}
.gr-chatbot .message.bot {
    background: rgba(255,255,255,0.05) !important;
    color: #F0F3F5 !important;
    border-radius: 14px 14px 14px 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
label, .label-text {
    color: #8899AA !important;
    font-weight: 500 !important;
    font-size: 12px !important;
}
.gr-markdown { color: #F0F3F5 !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: #F0F3F5 !important; }
.gr-tabs { border: none !important; }
.gr-tab {
    background: rgba(12, 28, 48, 0.4) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #8899AA !important;
    border-radius: 10px 10px 0 0 !important;
    transition: all 0.2s ease !important;
}
.gr-tab.selected {
    background: rgba(12, 28, 48, 0.7) !important;
    color: #00BFA5 !important;
    border-color: rgba(0,191,165,0.3) !important;
    border-bottom-color: rgba(12, 28, 48, 0.7) !important;
}
#system-radio label {
    display: block !important;
    padding: 10px 16px !important;
    margin: 4px 0 !important;
    background: rgba(12, 28, 48, 0.5) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-left: 3px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    color: #8899AA !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
#system-radio label:hover {
    border-color: rgba(0,191,165,0.2) !important;
    color: #F0F3F5 !important;
    background: rgba(0,191,165,0.04) !important;
}
#system-radio input:checked + label {
    border-color: #00BFA5 !important;
    border-left-color: #00BFA5 !important;
    background: rgba(0,191,165,0.08) !important;
    color: #00BFA5 !important;
    font-weight: 600 !important;
}

footer { display: none !important; }
"""


# ═══════════════════════════════════════════════════════════════
# Gradio UI — Enhanced Dashboard
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Overview page builder
# ═══════════════════════════════════════════════════════════════

_SUBSYSTEMS_META = [
    {"id": "exhaust",    "icon": "🌡️", "name": "排气系统", "count": 3, "alert": 0},
    {"id": "cooling",    "icon": "❄️",  "name": "冷却系统", "count": 4, "alert": 0},
    {"id": "lube",       "icon": "🛢️", "name": "滑油系统", "count": 2, "alert": 1},
    {"id": "scavenge",   "icon": "💨",  "name": "扫气系统", "count": 3, "alert": 0},
    {"id": "combustion", "icon": "⚡",  "name": "燃烧参数", "count": 2, "alert": 0},
    {"id": "turbo",      "icon": "🔄",  "name": "增压器",   "count": 2, "alert": 0},
    {"id": "fuel",       "icon": "📉",  "name": "油耗",     "count": 2, "alert": 0},
]


def build_overview_html() -> str:
    """Build left-right split dashboard HTML."""
    # ── sidebar ──
    nav_items = ""
    for s in _SUBSYSTEMS_META:
        alert_html = f'<span class="nav-alert-badge">[{s["alert"]}]</span>' if s["alert"] > 0 else ""
        nav_items += (
            f'<div class="dash-nav-item" onclick=\'document.querySelector("#sys-card-{s["id"]} button").click()\'>'
            f'<span class="nav-icon">{s["icon"]}</span>'
            f'<span class="nav-label">{s["name"]}</span>'
            f'<span class="nav-count">{s["count"]}</span>'
            f'{alert_html}'
            f'<span class="nav-status-dot"></span>'
            f'</div>'
        )

    sidebar = (
        '<div class="dash-sidebar-brand">'
        '  <span class="brand-icon">⚓</span>'
        '  <span class="brand-text">轮机智脑</span>'
        '</div>'
        '<div class="dash-sidebar-nav">'
        '  <div class="dash-nav-section-title">子系统</div>'
        '  <div class="dash-nav-item active" onclick=\'document.querySelector("#sys-card-overview button").click()\'>'
        '    <span class="nav-icon">📊</span>'
        '    <span class="nav-label">全局概览</span>'
        '  </div>'
        f' {nav_items}'
        '</div>'
        '<div class="dash-sidebar-footer">'
        '  <div class="dash-nav-section-title">更多</div>'
        '  <div class="dash-nav-item" onclick=\'document.querySelector("#image-entry-btn-hidden button").click()\'>'
        '    <span class="nav-icon">🖼️</span>'
        '    <span class="nav-label">图片识别</span>'
        '  </div>'
        '  <div class="dash-nav-item" onclick=\'document.querySelector("#history-top-btn-hidden button").click()\'>'
        '    <span class="nav-icon">📊</span>'
        '    <span class="nav-label">历史数据</span>'
        '  </div>'
        '</div>'
    )

    # ── chart area ──
    chart_area = (
        '<div class="dash-chart-area">'
        '  <div class="chart-header">'
        '    <span class="chart-title">全局健康趋势</span>'
        '    <div class="chart-tabs">'
        '      <span class="chart-tab active">24h</span>'
        '      <span class="chart-tab">7d</span>'
        '      <span class="chart-tab">30d</span>'
        '    </div>'
        '  </div>'
        '  <div class="chart-placeholder">'
        '    <svg width="100%" height="200" viewBox="0 0 600 200">'
        '      <polyline points="10,150 80,140 150,130 220,115 290,105 360,95 430,85 500,80 570,60" '
        '                fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" opacity="0.6"/>'
        '      <polyline points="10,160 80,148 150,142 220,130 290,120 360,112 430,105 500,100 570,90" '
        '                fill="none" stroke="#16a34a" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="5,5" opacity="0.4"/>'
        '      <line x1="10" y1="170" x2="570" y2="170" stroke="#e2e8f0" stroke-width="1"/>'
        '      <text x="10" y="185" fill="#94a3b8" font-size="10" font-family="sans-serif">0:00</text>'
        '      <text x="290" y="185" fill="#94a3b8" font-size="10" font-family="sans-serif" text-anchor="middle">12:00</text>'
        '      <text x="565" y="185" fill="#94a3b8" font-size="10" font-family="sans-serif" text-anchor="end">23:59</text>'
        '    </svg>'
        '    <span style="margin-top:8px;">选择子系统查看详细趋势图表</span>'
        '  </div>'
        '</div>'
    )

    # ── KPI ──
    kpi = (
        '<div class="kpi-card"><div class="kpi-label">监控参数</div>'
        f'<div class="kpi-value">{len(KB_BASELINE)}</div><div class="kpi-sub">18 项全覆盖</div></div>'
        '<div class="kpi-card"><div class="kpi-label">负载点</div>'
        f'<div class="kpi-value">{len(LOADS)}</div><div class="kpi-sub">25% — 110%</div></div>'
        '<div class="kpi-card"><div class="kpi-label">健康评分</div>'
        '<div class="kpi-value warn">92</div><div class="kpi-sub">滑油系统需关注</div></div>'
        '<div class="kpi-card"><div class="kpi-label">系统状态</div>'
        '<div class="kpi-value ok"><span class="kpi-dot"></span>在线</div><div class="kpi-sub">全系统运行中</div></div>'
    )

    # ── events ──
    events = """\
<div class="events-title">最新事件</div>
<div class="event-item">
    <span class="event-dot warn"></span>
    <div class="event-body">
      <div class="event-text">滑油压力偏离基线 +2.3σ</div>
      <div class="event-time">10 分钟前</div>
    </div>
  </div>
  <div class="event-item">
    <span class="event-dot info"></span>
    <div class="event-body">
      <div class="event-text">冷却系统基准校准完成</div>
      <div class="event-time">42 分钟前</div>
    </div>
  </div>
  <div class="event-item">
    <span class="event-dot ok"></span>
    <div class="event-body">
      <div class="event-text">全系统自检通过</div>
      <div class="event-time">1 小时前</div>
    </div>
  </div>
  <div class="events-footer"><a href="#">查看全部事件 →</a></div>"""

    return (
        '<div class="dash-layout">'
        f'<div class="dash-sidebar">{sidebar}</div>'
        '<div class="dash-main">'
        f'<div class="kpi-row">{kpi}</div>'
        '<div class="dash-content-row">'
        f'{chart_area}'
        f'<div class="dash-events-area">{events}</div>'
        '</div>'  # end dash-content-row
        '</div>'  # end dash-main
        '</div>'  # end dash-layout
    )


def build_detail_breadcrumb_html(system_name: str) -> str:
    meta = SYSTEM_META.get(system_name, SYSTEM_META["排气系统"])
    param_count = len(SYSTEM_TABS.get(system_name, []))
    return (
        f'<div class="detail-breadcrumb">'
        f'<span class="breadcrumb-icon">{meta["icon"]}</span>'
        f'<span class="breadcrumb-name">{system_name}</span>'
        f'<span class="breadcrumb-badge" style="border-color:{meta["color"]};color:{meta["color"]};">{param_count} 参数</span>'
        f'</div>'
    )


SYSTEM_CARD_BUTTONS = [
    ("排气系统", "sys-card-exhaust", "🌡️ 排气系统 — 3 参数"),
    ("冷却系统", "sys-card-cooling", "❄️ 冷却系统 — 4 参数"),
    ("滑油系统", "sys-card-lube", "🛢️ 滑油系统 — 2 参数"),
    ("扫气系统", "sys-card-scavenge", "💨 扫气系统 — 3 参数"),
    ("燃烧参数", "sys-card-combustion", "⚡ 燃烧参数 — 2 参数"),
    ("增压器", "sys-card-turbo", "🔄 增压器 — 2 参数"),
    ("油耗", "sys-card-fuel", "⛽ 油耗 — 2 参数"),
]


def create_ui():
    default_system = "排气系统"

    with gr.Blocks(
        title="Marine Engine AI - 轮机智脑 · 可视化",
    ) as demo:

        # ── Hidden State ──
        selected_system_state = gr.State(value=default_system)
        page_state = gr.State(value="overview")

                # ═══════════════════════════════
        # PAGE 1: 系统总览 Overview — Left-Right Dashboard
        # ═══════════════════════════════
        with gr.Group(visible=True, elem_id="overview-page") as overview_page:
            overview_html = gr.HTML(
                value=build_overview_html(),
                elem_id="overview-html",
            )

            # Hidden buttons for JS click navigation (kept invisible)
            with gr.Row(elem_classes="hidden-buttons", visible=True):
                btn_overview = gr.Button("全局概览", elem_id="sys-card-overview", size="sm")
                btn_exhaust = gr.Button("排气系统", elem_id="sys-card-exhaust", size="sm")
                btn_cooling = gr.Button("冷却系统", elem_id="sys-card-cooling", size="sm")
                btn_lube = gr.Button("滑油系统", elem_id="sys-card-lube", size="sm")
                btn_scavenge = gr.Button("扫气系统", elem_id="sys-card-scavenge", size="sm")
                btn_combustion = gr.Button("燃烧参数", elem_id="sys-card-combustion", size="sm")
                btn_turbo = gr.Button("增压器", elem_id="sys-card-turbo", size="sm")
                btn_fuel = gr.Button("油耗", elem_id="sys-card-fuel", size="sm")
                image_entry_btn = gr.Button("图片识别", elem_id="image-entry-btn-hidden", size="sm")
                history_top_btn = gr.Button("历史数据", elem_id="history-top-btn-hidden", size="sm")

        # ═══════════════════════════════
        # PAGE 2: 图片识别 Image Analysis
        # ═══════════════════════════════
        with gr.Group(visible=False) as image_page:
            with gr.Row(elem_id="breadcrumb-row"):
                back_from_image_btn = gr.Button(
                    "← 返回系统总览", elem_id="back-overview-btn", size="sm",
                )
                gr.HTML(
                    value='<div class="detail-breadcrumb">'
                          '<span class="breadcrumb-icon">🖼️</span>'
                          '<span class="breadcrumb-name">图片识别 · 轮机视觉分析</span>'
                          '<span class="breadcrumb-badge" style="border-color:#8B5CF6;color:#8B5CF6;">DSR1</span>'
                          '</div>',
                    elem_id="image-breadcrumb-html",
                )

            with gr.Row(equal_height=True):
                # ── Left: image upload + preview ──
                with gr.Column(scale=2, min_width=350):
                    gr.HTML('<div class="panel-title">📤 上传图片</div>')
                    image_input = gr.Image(
                        label="",
                        type="filepath",
                        height=420,
                        elem_id="image-upload",
                    )
                    with gr.Row():
                        image_question = gr.Textbox(
                            label="",
                            placeholder="可选：补充问题（如「这是什么型号的增压器？」）…",
                            scale=4,
                            elem_id="image-question-input",
                        )
                        image_analyze_btn = gr.Button(
                            "🔍 分析图片", variant="primary", scale=1, size="lg",
                        )

                # ── Right: analysis result ──
                with gr.Column(scale=3, min_width=400):
                    gr.HTML('<div class="panel-title">📋 分析结果</div>')
                    image_result = gr.Chatbot(
                        label="",
                        height=360,
                        elem_id="image-chatbot",
                    )
                    image_chart = gr.Plot(
                        label="可视化对比图表",
                        visible=False,
                        elem_id="image-chart",
                    )
                    with gr.Row():
                        image_clear_btn = gr.Button("🗑️ 清空结果", size="sm", scale=1)
                        image_save_label = gr.Textbox(
                            placeholder="保存名称（可选）...", scale=3,
                            show_label=False,
                        )
                        image_save_btn = gr.Button("💾 保存分析", variant="primary", size="sm", scale=1)
                    image_save_msg = gr.Markdown("")
                    image_session_state = gr.State({"img_path": "", "points": []})

        # ═══════════════════════════════
        # PAGE 3: 子系统详情 Detail
        # ═══════════════════════════════
        with gr.Group(visible=False) as monitor_page:
            # Breadcrumb bar
            with gr.Row(elem_id="breadcrumb-row"):
                back_to_overview_btn = gr.Button(
                    "← 返回系统总览", elem_id="back-overview-btn", size="sm",
                )
                detail_title_html = gr.HTML(
                    value=build_detail_breadcrumb_html(default_system),
                    elem_id="detail-title-html",
                )

            status_bar = gr.HTML(value=build_status_bar_html(), elem_id="status-bar")

            # Main content row
            with gr.Row(equal_height=False):
                # ── Column 1: System Selector + Controls ──
                with gr.Column(scale=1, min_width=220):
                    gr.HTML('<div class="panel-title">🔧 分系统监控</div>')
                    system_radio = gr.Radio(
                        choices=[(f"{SYSTEM_META[s]['icon']} {s}", s) for s in SYSTEM_TABS],
                        value=default_system,
                        label="",
                        elem_id="system-radio",
                    )

                    gr.HTML('<div class="panel-title" style="margin-top:16px;">🎛️ 控制面板</div>')

                    highlight_slider = gr.Slider(
                        minimum=25, maximum=110, value=75, step=1,
                        label="负载点高亮",
                        info="拖动选择当前关注的负载百分比",
                        elem_id="load-slider",
                    )

                    time_range_radio = gr.Radio(
                        choices=[("最近 1 小时", "1h"), ("最近 6 小时", "6h"),
                                 ("最近 24 小时", "24h"), ("最近 7 天", "7d"),
                                 ("全部数据", "all")],
                        value="all",
                        label="时间范围",
                    )

                    with gr.Row():
                        export_csv_btn = gr.Button("📥 CSV", size="sm", scale=1)
                        export_png_btn = gr.Button("📸 PNG", size="sm", scale=1)

                    export_msg = gr.Markdown("", elem_id="export-msg")

                    gr.HTML('<div class="panel-title" style="margin-top:16px;">💾 历史数据</div>')
                    history_list_md = gr.Markdown("*暂无历史数据*")
                    with gr.Row():
                        history_dropdown = gr.Dropdown(
                            choices=list_saved_sessions(), label="",
                            interactive=True, scale=3,
                        )
                        refresh_history_btn = gr.Button("🔄 刷新", size="sm", scale=1)
                    with gr.Row():
                        load_history_btn = gr.Button("📂 加载", size="sm", scale=1)
                        delete_history_btn = gr.Button("🗑️ 删除", size="sm", scale=1)

                # ── Column 2: Main Chart Area ──
                with gr.Column(scale=3.5, min_width=500):
                    gr.HTML('<div class="panel-title">📈 实时趋势图表</div>')
                    main_chart = gr.Plot(
                        value=build_enhanced_chart(
                            default_system, SYSTEM_TABS[default_system], session, 75, "all",
                        ),
                        elem_id="main-chart",
                        show_label=False,
                    )

                    data_summary = gr.Markdown(
                        session.to_markdown(),
                        elem_id="data-summary-md",
                    )

                # ── Column 3: Chat Area ──
                with gr.Column(scale=1.8, min_width=320):
                    gr.HTML('<div class="panel-title">💬 智能对话</div>')
                    chatbot = gr.Chatbot(
                        label="",
                        height=380,
                        elem_id="chatbot",
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            label="",
                            placeholder="输入传感器参数或问题…  例：负载50%，缸套水出水88度，排气263度，正常吗？",
                            lines=2, scale=5,
                            elem_id="chat-input",
                        )
                        send_btn = gr.Button("▶ 发送", variant="primary", scale=1)

                    gr.HTML('<div class="panel-title" style="margin-top:12px;">📋 快捷模板</div>')
                    with gr.Row():
                        tpl_wk = gr.Button("📋 温控", size="sm")
                        tpl_yh = gr.Button("📋 油耗", size="sm")
                        tpl_zy = gr.Button("📋 增压器", size="sm")
                        tpl_fz = gr.Button("📋 负载参数", size="sm")

                    gr.HTML('<div class="panel-title" style="margin-top:12px;">⚙️ 操作</div>')
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空对话", size="sm")
                        clear_data_btn = gr.Button("🔄 重置图表", size="sm")
                    with gr.Row():
                        save_label = gr.Textbox(
                            placeholder="测试名称（可选）...", scale=3,
                            show_label=False,
                        )
                        save_btn = gr.Button("💾 保存测试", variant="primary", size="sm", scale=1)
                        save_msg = gr.Markdown("")

        # ═══════════════════════════════
        # PAGE 3: 历史数据 History
        # ═══════════════════════════════
        with gr.Group(visible=False) as history_page:
            with gr.Row():
                back_btn = gr.Button("⬅ 返回总览", size="sm", elem_id="back-history-btn")
            gr.HTML('<div class="panel-title" style="margin-top:12px;">📂 全部历史会话</div>')
            history_type_filter = gr.Radio(
                choices=[("全部", "all"), ("文字输出", "text"), ("图片分析", "image")],
                value="all",
                label="分类筛选",
                interactive=True,
            )
            with gr.Row():
                hist_dropdown = gr.Dropdown(
                    choices=list_saved_sessions(), label="选择历史文件",
                    interactive=True, scale=4,
                )
                hist_load_btn = gr.Button("📂 加载并查看", variant="primary", scale=1)
                hist_refresh_btn = gr.Button("🔄 刷新", size="sm", scale=1)
            history_chart = gr.Plot(
                value=build_empty_chart(),
                elem_id="history-chart",
                show_label=False,
            )
            history_image = gr.Image(
                visible=False,
                elem_id="history-image",
                show_label=False,
                height=360,
            )
            history_full_md = gr.Markdown("*选择历史文件后点击加载*", elem_id="history-full-md")

        # ═══════════════════════════════════════════
        # NAVIGATION HANDLERS
        # ═══════════════════════════════════════════

        def _nav_to_system(sys_name: str):
            """Navigate from overview to detail page."""
            params = SYSTEM_TABS.get(sys_name, [])
            fig = build_enhanced_chart(sys_name, params, session, 75, "all")
            return (
                gr.update(visible=False),                            # overview_page
                gr.update(visible=False),                            # image_page
                gr.update(visible=True),                             # monitor_page
                gr.update(visible=False),                            # history_page
                build_detail_breadcrumb_html(sys_name),              # detail_title_html
                fig,                                                  # main_chart
                build_status_bar_html(),                              # status_bar
                sys_name,                                             # selected_system_state
                session.to_markdown(),                                # data_summary
                "detail",                                             # page_state
                gr.update(choices=list_saved_sessions()),             # history_dropdown
            )

        def _nav_to_overview():
            """Navigate back to overview from detail/history/image."""
            return (
                gr.update(visible=True),   # overview_page
                gr.update(visible=False),  # image_page
                gr.update(visible=False),  # monitor_page
                gr.update(visible=False),  # history_page
                "overview",                # page_state
            )

        def _nav_to_image():
            """Navigate from overview to image analysis page."""
            return (
                gr.update(visible=False),  # overview_page
                gr.update(visible=True),   # image_page
                gr.update(visible=False),  # monitor_page
                gr.update(visible=False),  # history_page
                "image",                   # page_state
                None,                       # image_input (clear)
                [],                         # image_result (clear)
                gr.update(visible=False),   # image_chart (clear)
            )

        def _nav_to_history():
            """Navigate from overview to history page."""
            files = list_saved_sessions()
            if not files:
                return (
                    gr.update(visible=False),  # overview_page
                    gr.update(visible=False),  # image_page
                    gr.update(visible=False),  # monitor_page
                    gr.update(visible=True),   # history_page
                    build_empty_chart(),        # history_chart
                    gr.update(visible=False),   # history_image
                    "*暂无历史数据*",            # history_full_md
                    gr.Dropdown(choices=[]),    # hist_dropdown
                    "history",                  # page_state
                )
            latest = files[0]
            fig, img_update, summary, dd = load_history_for_chart(latest)
            return (
                gr.update(visible=False),  # overview_page
                gr.update(visible=False),  # image_page
                gr.update(visible=False),  # monitor_page
                gr.update(visible=True),   # history_page
                fig,                        # history_chart
                img_update,                 # history_image
                summary,                    # history_full_md
                dd,                         # hist_dropdown
                "history",                  # page_state
            )

        # ── System card clicks: navigate to detail page ──
        for sys_name, btn_var_name in [
            ("排气系统", "btn_exhaust"),
            ("冷却系统", "btn_cooling"),
            ("滑油系统", "btn_lube"),
            ("扫气系统", "btn_scavenge"),
            ("燃烧参数", "btn_combustion"),
            ("增压器", "btn_turbo"),
            ("油耗", "btn_fuel"),
        ]:
            btn = locals()[btn_var_name]
            btn.click(
                fn=lambda s=sys_name: _nav_to_system(s),
                inputs=[],
                outputs=[overview_page, image_page, monitor_page, history_page,
                         detail_title_html, main_chart, status_bar,
                         selected_system_state, data_summary, page_state, history_dropdown],
            )

        # ── Back button: detail → overview ──
        back_to_overview_btn.click(
            fn=_nav_to_overview,
            inputs=[],
            outputs=[overview_page, image_page, monitor_page, history_page, page_state],
        )

        # ── Image entry button: overview → image ──
        image_entry_btn.click(
            fn=_nav_to_image,
            inputs=[],
            outputs=[
                overview_page, image_page, monitor_page, history_page,
                page_state, image_input, image_result, image_chart,
            ],
        )

        # ── Back from image → overview ──
        back_from_image_btn.click(
            fn=_nav_to_overview,
            inputs=[],
            outputs=[overview_page, image_page, monitor_page, history_page, page_state],
        )

        # ── History button: overview → history ──
        history_top_btn.click(
            fn=_nav_to_history,
            inputs=[],
            outputs=[
                overview_page, image_page, monitor_page, history_page,
                history_chart, history_image, history_full_md, hist_dropdown, page_state,
            ],
        )

        # ── Back from history → overview ──
        back_btn.click(
            fn=_nav_to_overview,
            inputs=[],
            outputs=[overview_page, image_page, monitor_page, history_page, page_state],
        )

        # ═══════════════════════════════════════════
        # EVENT HANDLERS (existing, unchanged)
        # ═══════════════════════════════════════════

        # ── System Select (via radio) ──
        def on_system_select(sys_name, hl_load, tr):
            params = SYSTEM_TABS.get(sys_name, [])
            fig = build_enhanced_chart(sys_name, params, session, hl_load, tr)
            return (
                fig,
                build_status_bar_html(),
                sys_name,
            )

        system_radio.change(
            fn=on_system_select,
            inputs=[system_radio, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar, selected_system_state],
        )

        # ── Load Slider / Time Range ──
        def on_control_change(sys_name, hl_load, tr):
            params = SYSTEM_TABS.get(sys_name, [])
            fig = build_enhanced_chart(sys_name, params, session, hl_load, tr)
            return fig, build_status_bar_html()

        highlight_slider.change(
            on_control_change,
            inputs=[selected_system_state, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar],
        )

        time_range_radio.change(
            on_control_change,
            inputs=[selected_system_state, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar],
        )

        # ── Templates ──
        tpl_wk.click(
            fn=lambda: "负载指数由【负载1】%到【负载2】%【填写温度部位及变化情况】",
            outputs=[msg],
        )
        tpl_yh.click(
            fn=lambda: "负载指数由【负载1】%到【负载2】%【填写油耗变化情况】",
            outputs=[msg],
        )
        tpl_zy.click(
            fn=lambda: "负载指数由【负载1】%到【负载2】%【填写涡轮增压器情况】",
            outputs=[msg],
        )
        tpl_fz.click(
            fn=lambda: "负载情况由【负载1】%到【负载2】%【填写部件或参数情况】",
            outputs=[msg],
        )

        # ── Send Message ──
        def on_send(msg_text, hist, sys_name, hl_load, tr):
            yield from process_chat_and_viz(msg_text, hist or [], sys_name, hl_load, tr)

        all_chat_outputs = [
            chatbot, main_chart, status_bar, data_summary,
        ]

        send_btn.click(
            on_send,
            inputs=[msg, chatbot, selected_system_state, highlight_slider, time_range_radio],
            outputs=all_chat_outputs,
        ).then(lambda: "", outputs=[msg])

        msg.submit(
            on_send,
            inputs=[msg, chatbot, selected_system_state, highlight_slider, time_range_radio],
            outputs=all_chat_outputs,
        ).then(lambda: "", outputs=[msg])

        # ── Clear Chat ──
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

        # ── Clear Data ──
        def clear_all_data(sys_name, hl_load, tr):
            session.clear()
            empty_fig = go.Figure()
            empty_fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=480,
            )
            empty_fig.add_annotation(
                text="<b>📡 图表已重置</b><br><sub>输入传感器参数后自动绘图</sub>",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=20, color="#556677"),
            )
            return (
                empty_fig,
                build_status_bar_html(),
                "*图表已重置*\n\n📡 等待传感器数据输入…",
            )

        clear_data_btn.click(
            clear_all_data,
            inputs=[selected_system_state, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar, data_summary],
        )

        # ── Save ──
        def on_save(label_text):
            if not session.points:
                return "⚠️ 暂无数据", "*暂无历史数据*"
            fname = save_session(label_text.strip() if label_text else "")
            files = list_saved_sessions()
            md = "\n".join(f"📊 `{f}`" for f in files) if files else "*暂无历史数据*"
            return f"✅ 已保存：{fname}", md

        save_btn.click(on_save, inputs=[save_label], outputs=[save_msg, history_list_md])

        # ── Load History ──
        def on_load_history(selected_file, sys_name, hl_load, tr):
            if not selected_file:
                params = SYSTEM_TABS.get(sys_name, [])
                fig = build_enhanced_chart(sys_name, params, session, hl_load, tr)
                return (
                    fig,
                    build_status_bar_html(),
                    "*请先选择一个历史文件*",
                )
            summary = load_session_data(selected_file)
            params = SYSTEM_TABS.get(sys_name, [])
            fig = build_enhanced_chart(sys_name, params, session, hl_load, tr)
            return (
                fig,
                build_status_bar_html(),
                summary,
            )

        load_history_btn.click(
            on_load_history,
            inputs=[history_dropdown, selected_system_state, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar, data_summary],
        )

        # ── Delete History ──
        def on_delete_history(selected_file):
            if not selected_file:
                return gr.Dropdown(choices=list_saved_sessions(), value=None), "*请先选择一个历史文件*"
            new_dd, new_md = delete_session(selected_file)
            return new_dd, new_md

        delete_history_btn.click(
            on_delete_history,
            inputs=[history_dropdown],
            outputs=[history_dropdown, history_list_md],
        )

        # ── Refresh History ──
        refresh_history_btn.click(
            refresh_history_list,
            inputs=[],
            outputs=[history_dropdown, history_list_md],
        )

        # ── History Page: Load file ──
        def load_history_for_chart(filename: str):
            """Load a history JSON file and render chart with all KB baselines."""
            _no_img = gr.update(visible=False)
            _dd = gr.Dropdown(choices=list_saved_sessions())
            if not filename:
                return build_empty_chart(), _no_img, "*请选择一个历史文件*", _dd
            filepath = HISTORY_DIR / filename
            if not filepath.exists():
                return build_empty_chart(), _no_img, f"文件不存在: {filename}", _dd
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
            except Exception:
                return build_empty_chart(), _no_img, "*文件读取失败*", _dd

            # Handle image analysis sessions
            if data.get("type") == "image":
                messages = data.get("messages", [])
                if not messages:
                    return build_empty_chart(), _no_img, "*该文件无对话数据*", _dd
                md_lines = []
                label = data.get("label", "图片分析")
                ts = data.get("timestamp", "")
                md_lines.append(f"## {label}")
                md_lines.append(f"*{ts}*")
                md_lines.append("")
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    role_label = "**🖼️ 用户**" if role == "user" else "**🤖 助手**"
                    if isinstance(content, list):
                        content_str = "\n".join(
                            item.get("text", str(item)) if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    else:
                        content_str = str(content)
                    md_lines.append(f"{role_label}:")
                    md_lines.append("")
                    md_lines.append(content_str)
                    md_lines.append("")
                summary_md = "\n".join(md_lines)

                # ── Build chart from saved chart_points ──
                chart_points = data.get("chart_points")
                img_path = data.get("image_path", "")
                fig = build_empty_chart()
                img_update = _no_img
                if chart_points:
                    try:
                        fig = build_image_analysis_chart(chart_points)
                    except Exception as e:
                        print(f"[History] Chart rebuild failed: {e}")
                if img_path and os.path.exists(img_path):
                    img_update = gr.update(value=img_path, visible=True)
                return fig, img_update, summary_md, _dd

            points = data.get("points", [])
            if not points:
                return build_empty_chart(), _no_img, "*该文件无数据点*", _dd

            fig = go.Figure()
            has_any_data = False
            params_seen = set()

            i = 0
            for pt in points:
                param = pt.get("param", "")
                if not param:
                    continue
                canonical = ALIAS_TO_PARAM.get(param, param)
                if canonical in params_seen:
                    continue
                params_seen.add(canonical)
                kb = KB_BASELINE.get(canonical)
                if not kb:
                    continue
                color = COLORS[i % len(COLORS)]
                unit = kb.get("unit", "")
                tolerance = kb.get("tolerance", 0)
                kb_values = kb.get("values", {})

                param_pts = [p for p in points
                           if ALIAS_TO_PARAM.get(p.get("param", ""), p.get("param", "")) == canonical]
                has_any_data = True

                if kb_values:
                    x_kb = list(kb_values.keys())
                    y_kb = list(kb_values.values())
                    fig.add_trace(go.Scatter(
                        x=x_kb, y=y_kb, mode="lines+markers",
                        name=f"{canonical} [KB基准]",
                        line=dict(dash="dot", width=1.5, color=color),
                        marker=dict(size=5, symbol="cross-thin", color=color),
                        opacity=0.40,
                        hovertemplate=f"<b>KB基准</b> {canonical}<br>值: %{{y:.1f}}{unit}<br>负载: %{{x}}%<extra></extra>",
                    ))
                    y_upper = [v + tolerance for v in y_kb]
                    y_lower = [v - tolerance for v in y_kb]
                    fig.add_trace(go.Scatter(
                        x=x_kb + x_kb[::-1], y=y_upper + y_lower[::-1],
                        fill="toself", fillcolor=_hex_to_rgba(color, 0.10),
                        line=dict(width=0), name=f"{canonical} ±{tolerance}{unit}",
                        showlegend=False, hoverinfo="skip",
                    ))

                x_act = [p["load"] for p in param_pts]
                y_act = [p["value"] for p in param_pts]
                fig.add_trace(go.Scatter(
                    x=x_act, y=y_act, mode="markers",
                    name=canonical,
                    marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                    hovertemplate=f"<b>{canonical}</b><br>值: %{{y:.1f}}{unit}<br>负载: %{{x}}%<extra></extra>",
                ))
                i += 1

            if not has_any_data:
                return build_empty_chart(), "*数据参数无匹配的KB基准*", gr.Dropdown(choices=list_saved_sessions())

            fig.update_layout(
                plot_bgcolor="rgba(12, 28, 48, 0.4)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F0F3F5"),
                height=500,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis=dict(title="负载 (%)", gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickfont=dict(color="#8899AA"), title_font=dict(color="#8899AA")),
                yaxis=dict(title="值", gridcolor="rgba(255,255,255,0.06)", zeroline=False, tickfont=dict(color="#8899AA"), title_font=dict(color="#8899AA")),
                legend=dict(font=dict(size=11, color="#8899AA")),
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="#0C1C30",
                    font=dict(color="#F0F3F5", size=12),
                    bordercolor="rgba(0,191,165,0.3)",
                ),
            )

            n_pts = len(points)
            n_params = len(params_seen)
            label = data.get("label", "未命名")
            created = data.get("created", "")[:16]
            summary = f"**{label}** — {n_pts} 个数据点，{n_params} 个参数 — {created}"
            return fig, _no_img, summary, gr.Dropdown(choices=list_saved_sessions())

        def refresh_history_page_dropdown():
            files = list_saved_sessions()
            return gr.Dropdown(choices=files)

        hist_load_btn.click(
            fn=load_history_for_chart,
            inputs=[hist_dropdown],
            outputs=[history_chart, history_image, history_full_md, hist_dropdown],
        )

        # ── History Page: Refresh list ──
        hist_refresh_btn.click(
            fn=refresh_history_page_dropdown,
            inputs=[],
            outputs=[hist_dropdown],
        )

        # ── History Page: Type filter ──
        def on_type_filter_change(type_val):
            files = list_saved_sessions(type_val)
            return gr.Dropdown(choices=files)

        history_type_filter.change(
            on_type_filter_change,
            inputs=[history_type_filter],
            outputs=[hist_dropdown],
        )

        # ── Export ──
        export_csv_btn.click(
            export_csv,
            inputs=[],
            outputs=[export_msg],
        )

        def on_export_png(sys_name, hl_load, tr):
            return export_png(sys_name, hl_load, tr)

        export_png_btn.click(
            on_export_png,
            inputs=[selected_system_state, highlight_slider, time_range_radio],
            outputs=[export_msg],
        )

        # ═══════════════════════════════════════════
        # IMAGE ANALYSIS HANDLERS
        # ═══════════════════════════════════════════

        def on_analyze_image(img_path, question, hist, session_state):
            """Analyze uploaded image using DSR1, stream result, embed chart in chatbot."""
            hist = hist or []
            session_state = session_state or {"img_path": "", "points": []}
            session_state["img_path"] = img_path or ""
            session_state["points"] = []
            if not img_path:
                hist.append({"role": "user", "content": "（未上传图片）"})
                hist.append({"role": "assistant", "content": "⚠️ 请先上传一张轮机相关图片。"})
                yield hist, gr.update(visible=False), session_state
                return

            user_msg = f"![上传图片]({img_path})"
            if question.strip():
                user_msg += f"\n\n**补充问题：** {question.strip()}"
            hist.append({"role": "user", "content": user_msg})
            hist.append({"role": "assistant", "content": ""})

            full = ""
            for chunk in analyze_image_stream(img_path, question):
                full += chunk
                hist[-1] = {"role": "assistant", "content": full}
                yield hist, gr.update(visible=False), session_state

            # ── DSR1 returned an error ──
            if full.startswith("❌") or full.startswith("⚠️"):
                yield hist, gr.update(visible=False), session_state
                return

            # ── Try structured extraction & embed chart in chatbot ──
            chart_embedded = False
            points = []
            try:
                # Primary: JSON from DSR1 response
                parsed = _extract_json_from_image_response(full)
                if parsed:
                    points_data = parsed.get("points")
                    if not points_data:
                        load_val = parsed.get("load")
                        params_val = parsed.get("params", [])
                        if params_val and load_val is not None:
                            points_data = [{"load": load_val, "params": params_val}]
                    if points_data:
                        points = _normalize_image_params(points_data)

                # Secondary: DeepSeek V3 extraction or regex sequence parsing
                if not points:
                    points = _extract_points_from_analysis_text(full)

                if points:
                    session_state["points"] = points
                    fig = build_image_analysis_chart(points)
                    if fig:
                        hist[-1] = {
                            "role": "assistant",
                            "content": [
                                full,
                                gr.Plot(value=fig, show_label=False),
                            ],
                        }
                        chart_embedded = True
                        print(f"[ImageViz] Chart embedded — {len(points)} pts, {len(set(p['name'] for p in points))} params")
            except Exception as e:
                print(f"[ImageViz] Chart generation failed: {e}")

            yield hist, gr.update(visible=False), session_state

        image_analyze_btn.click(
            on_analyze_image,
            inputs=[image_input, image_question, image_result, image_session_state],
            outputs=[image_result, image_chart, image_session_state],
        )

        image_clear_btn.click(
            lambda: ([], None, gr.update(visible=False), {"img_path": "", "points": []}),
            inputs=[],
            outputs=[image_result, image_input, image_chart, image_session_state],
        )

        def on_save_image(hist_data, label, session_state):
            if not hist_data:
                return "⚠️ 暂无分析结果"
            img_path = (session_state or {}).get("img_path", "")
            chart_points = (session_state or {}).get("points", [])
            fname = save_image_session(hist_data, label.strip() if label else "图片分析", img_path, chart_points)
            return f"✅ 已保存：{fname}"

        image_save_btn.click(
            on_save_image,
            inputs=[image_result, image_save_label, image_session_state],
            outputs=[image_save_msg],
        )

        # ── Back from history → overview ── (moved up to right after _nav_to_history) ──
        # already handled above

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("🚢 Marine Engine AI — 轮机智脑 · 可视化监控面板 (Enhanced)")
    print(f"   Chat model routing: DS V4 Pro / DS V3 / DSR1")
    print(f"   Data extraction: DSR1 (Doubao) @ {DSR1_API_BASE}")
    print(f"   KB baselines: {len(KB_BASELINE)} parameters across {len(SYSTEM_TABS)} systems")
    print("=" * 60)

    print("\n📚 Loading KB...")
    retriever = get_retriever()
    print(f"   {len(retriever.segments)} segments loaded")

    print("\n🌐 Starting at http://localhost:7861 ...")
    demo = create_ui()
    demo.queue(default_concurrency_limit=3)
    demo.launch(
        server_name="0.0.0.0", server_port=7863, share=False, inbrowser=True,
        show_error=True,
        css=DASHBOARD_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    )
