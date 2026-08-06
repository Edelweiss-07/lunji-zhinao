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
# API Config
# ═══════════════════════════════════════════════════════════════

OLLAMA_BASE = "http://localhost:11434/v1"
MODEL_NAME = "qwen2.5vl:7b"
client = OpenAI(base_url=OLLAMA_BASE, api_key="ollama")

SCHOOL_API_BASE = "https://chat.cqjtu.edu.cn/ds/api/v1"
SCHOOL_API_KEY = "sk-562cfe915f0b772b9cf663103eb962e0"
SCHOOL_MODEL = "deepseek-v3-2-251201"
school_client = OpenAI(base_url=SCHOOL_API_BASE, api_key=SCHOOL_API_KEY)

DSV4_KEY = "sk-ab957bd6618345f9e483faaa4ef66bc6"
DSV4_MODEL = "deepseek-chat"
dsv4_client = OpenAI(base_url=SCHOOL_API_BASE, api_key=DSV4_KEY)

DSR1_API_BASE = "https://chat.cqjtu.edu.cn/ds/api/v1"
DSR1_API_KEY = "sk-ba4c1b12745ac838e88520c4ddee80a0"
DSR1_MODEL = "doubao-2.0-pro"
dsr1_client = OpenAI(base_url=DSR1_API_BASE, api_key=DSR1_API_KEY)

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
    "排气系统": {"icon": "🌡️", "color": "#EF4444", "label": "排气", "btn_elem": "exhaust"},
    "冷却系统": {"icon": "❄️",  "color": "#3B82F6", "label": "冷却", "btn_elem": "cooling"},
    "滑油系统": {"icon": "🛢️", "color": "#F59E0B", "label": "滑油", "btn_elem": "lube"},
    "扫气系统": {"icon": "💨",  "color": "#10B981", "label": "扫气", "btn_elem": "scavenge"},
    "燃烧参数": {"icon": "⚡",  "color": "#F97316", "label": "燃烧", "btn_elem": "combustion"},
    "增压器":   {"icon": "🔄",  "color": "#8B5CF6", "label": "增压器", "btn_elem": "turbo"},
    "油耗":     {"icon": "⛽",  "color": "#EC4899", "label": "油耗", "btn_elem": "fuel"},
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

    def get_points_for_param(self, param: str, time_range: str = "all") -> list[dict]:
        """Alias for points_for — used by build_param_cards_html."""
        return self.points_for(param, time_range)

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
# 健康度与异常管理系统
# ═══════════════════════════════════════════════════════════════
from collections import deque

# 全局健康度历史（最多保留100个时间点，用于绘制趋势图）
MAX_HEALTH_POINTS = 100
health_history = deque(maxlen=MAX_HEALTH_POINTS)
health_history.append({"time": datetime.now().strftime("%H:%M:%S"), "score": 100})

# 全局异常列表
anomalies = []
anomaly_id_counter = 0

def get_current_health() -> int:
    """获取当前健康度：100 - 未解决异常数"""
    unresolved = sum(1 for a in anomalies if a["status"] == "unresolved")
    return max(0, 100 - unresolved)

def add_anomaly(system: str, param: str, desc: str = "") -> int:
    """新增一个异常，返回异常ID"""
    global anomaly_id_counter
    anomaly_id_counter += 1
    anomaly = {
        "id": anomaly_id_counter,
        "system": system,
        "param": param,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "unresolved",  # unresolved / resolved
        "desc": desc or f"{param} 参数偏离正常范围",
    }
    anomalies.append(anomaly)
    # 更新健康度历史
    health_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "score": get_current_health()
    })
    return anomaly_id_counter

def resolve_anomaly(anomaly_id: int) -> bool:
    """解决指定异常，返回是否成功"""
    for a in anomalies:
        if a["id"] == anomaly_id:
            a["status"] = "resolved"
            a["resolve_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # 更新健康度历史
            health_history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "score": get_current_health()
            })
            return True
    return False
def _detect_and_add_anomaly(param: str, value: float, load_pct: float):
    """Check a data point against KB baseline; auto-add anomaly if deviation > tolerance."""
    kb_entry = KB_BASELINE.get(param)
    if not kb_entry or "values" not in kb_entry:
        return
    
    kb_vals = kb_entry.get("values", {})
    # Find closest load point
    closest_load = min(kb_vals.keys(), key=lambda k: abs(k - load_pct))
    baseline_val = kb_vals[closest_load]
    tolerance = kb_entry.get("tolerance", 5)
    
    if baseline_val == 0:
        return
    
    dev_pct = abs((value - baseline_val) / baseline_val * 100)
    if dev_pct > tolerance:
        # Determine system for this param
        system_name = "未分类"
        for sys_key, params in SYSTEM_TABS.items():
            if param in params:
                system_name = sys_key
                break
        desc = f"{param}={value}，基线={baseline_val}（偏差{dev_pct:.1f}%）"
        add_anomaly(system_name, param, desc)


def get_anomalies_by_system(system_name: str = None) -> list:
    """获取指定子系统的异常，不传则返回全部"""
    if system_name is None:
        return anomalies
    return [a for a in anomalies if a["system"] == system_name]

def build_health_trend_chart():
    """生成全局健康趋势图"""
    import plotly.graph_objects as go
    
    times = [p["time"] for p in health_history]
    scores = [p["score"] for p in health_history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=scores,
        mode='lines+markers',
        name='健康度',
        line=dict(color='#165DFF', width=3),
        marker=dict(size=6, color='#165DFF'),
        fill='tozeroy',
        fillcolor='rgba(22, 93, 255, 0.1)',
    ))
    
    fig.update_layout(
        margin=dict(l=40, r=20, t=10, b=30),
        height=180,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94A3B8', size=10),
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.05)',
            tickfont=dict(color='#94A3B8', size=10),
            zeroline=False,
            title='',
        ),
        showlegend=False,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1E293B',
            font=dict(color='#FFFFFF', size=12),
        ),
    )
    return fig


def build_anomaly_grid_html() -> str:
    """生成异常卡片网格HTML（8张卡片，不足补空位）"""
    cards = []
    # 最多显示8个异常，按时间倒序
    sorted_anomalies = sorted(anomalies, key=lambda x: x["time"], reverse=True)[:8]
    
    for a in sorted_anomalies:
        status_class = a["status"]
        status_text = "待处理" if status_class == "unresolved" else "已解决"
        btn_text = "解决异常" if status_class == "unresolved" else "已解决"
        btn_class = status_class
        
        card = f'''
        <div class="anomaly-card {status_class}">
            <div class="anomaly-card-header">
                <div class="anomaly-system">{a["system"]}</div>
                <span class="anomaly-status {status_class}">{status_text}</span>
            </div>
            <div class="anomaly-param">📌 {a["param"]}</div>
            <div class="anomaly-desc">{a["desc"]}</div>
            <div class="anomaly-time">⏰ {a["time"]}</div>
            <button class="resolve-btn {btn_class}" 
                    onclick="resolveAnomaly({a['id']})"
                    {"disabled" if status_class == "resolved" else ""}>
                {btn_text}
            </button>
        </div>
        '''
        cards.append(card)
    
    # 补足8张卡片的空位
    empty_count = 8 - len(cards)
    for _ in range(empty_count):
        cards.append('<div class="empty-card">暂无异常</div>')
    
    return f'<div class="anomaly-grid">{"".join(cards)}</div>'


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

def call_llm_stream(system_prompt, user_message):
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME, temperature=0.7, max_tokens=4096, stream=True,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}],
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"❌ LLM调用失败: {e}"

def call_school_llm_stream(system_prompt, user_message):
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
    try:
        stream = dsv4_client.chat.completions.create(
            model=DSV4_MODEL, temperature=0.7, max_tokens=4096, stream=True,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}],
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"[V4 Pro 不可用，降级]\n"
        yield from call_school_llm_stream(system_prompt, user_message)

def call_dsr1_chat_stream(system_prompt, user_message):
    try:
        stream = dsr1_client.chat.completions.create(
            model=DSR1_MODEL, temperature=0.7, max_tokens=4096, stream=True,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_message}],
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"[DSR1 不可用，降级本地qwen]\n"
        yield from call_llm_stream(system_prompt, user_message)

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
            font=dict(color=SYSTEM_META.get(system_name, {}).get('color', '#1E293B'), size=16),
            x=0.02,
            xanchor="left",
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


def build_param_cards_html(system_name: str, load_pct: float = 75.0) -> str:
    """生成指定系统的参数卡片HTML"""
    params = SYSTEM_TABS.get(system_name, [])
    cards = ""
    
    for param_name in params:
        baseline = KB_BASELINE.get(param_name, {})
        baseline_val = baseline.get("baseline", {}).get(load_pct, None)
        unit = baseline.get("unit", "")
        tolerance = baseline.get("tolerance", 0)
        
        # 获取当前session中的最新值
        current_val = None
        param_points = session.get_points_for_param(param_name)
        if param_points:
            current_val = param_points[-1]["value"]
        
        # 确定状态
        status = "ok"
        display_val = f"{current_val:.1f}" if current_val is not None else "--"
        
        if current_val is not None and baseline_val is not None:
            deviation = abs(current_val - baseline_val)
            if deviation > tolerance * 2:
                status = "alert"
            elif deviation > tolerance:
                status = "warn"
        
        baseline_display = f"基线 {baseline_val:.1f}" if baseline_val is not None else "基线 --"
        
        cards += (
            f'<div class="param-card {status}">'
            f'  <div class="param-name">{param_name}</div>'
            f'  <div class="param-value">{display_val}<span class="param-unit">{unit}</span></div>'
            f'  <div class="param-baseline">{baseline_display}</div>'
            f'</div>'
        )
    
    sys_color = SYSTEM_META.get(system_name, {}).get("color", "#3B82F6")
    return f'<div class="param-cards-row" style="--sys-color: {sys_color};">{cards}</div>'


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
                    # Auto-detect anomaly: check against KB_BASELINE
                    _detect_and_add_anomaly(pt["name"], pt["value"], pt["load"])
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
# Image Understanding Pipeline (qwen2.5vl multimodal)
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
            kb_results = retriever.search(search_query, kb_names=IMAGE_KB_NAMES, top_k=12)
            if kb_results:
                lines = ["## 知识库参考资料\n"]
                seen = set()
                for seg in kb_results:
                    key = seg.content[:60]
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"### （来源：{seg.kb_name}）\n{seg.content[:2000]}\n")
                kb_context = "\n".join(lines)
                print(f"[KB] 图片分析多KB联合检索命中 {len(kb_results)} 条")
    except Exception as e:
        print(f"[KB] 图片分析知识库检索失败: {e}")

    # KB 数据注入 user message（与聊天管线一致），system prompt 保持不变
    if kb_context:
        user_content = kb_context + "\n\n---\n\n请根据以上知识库参考数据，分析以下图片：\n" + user_content
        # DEBUG: 写日志验证 KB 是否注入（相对路径，保证可移植）
        with open(Path(__file__).parent / "viz_debug.log", "w", encoding="utf-8") as _f:
            _f.write(f"kb_context length: {len(kb_context)}\n")
            _f.write(f"kb_context preview: {kb_context[:500]}\n")
            _f.write(f"user_content length: {len(user_content)}\n")

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
                    abs_dev = abs(vy - kb_values[lx])
                    rel_dev = abs_dev / kb_values[lx] if kb_values[lx] else 0
                    if rel_dev > tolerance * 1.5 / kb_values.get(lx, 1):
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
    filepath = HISTORY_DIR / filename
    if not filepath.exists():
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
    filepath = HISTORY_DIR / filename
    if filepath.exists():
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
/* ========================================
   轮机智脑 — 3D舰船HUD风格 · 白绿配色
   Naval Engine HUD · White + Green Theme
   ======================================== */

.gradio-container {
    max-width: 100% !important;
    background: #0A1628 !important;
    font-family: 'Inter', 'Noto Sans SC', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: #FFFFFF !important;
    overflow-x: hidden;
}

/* 背景网格效果 - 舰船指挥室风格 */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(rgba(0, 255, 136, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 255, 136, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(0, 255, 136, 0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 136, 0.45); }

/* ============================
   TOP NAVBAR — 全局顶部导航
   ============================ */
.hud-topbar {
    position: relative;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    height: 64px;
    background: linear-gradient(180deg, rgba(15, 30, 54, 0.95) 0%, rgba(10, 22, 40, 0.9) 100%);
    border-bottom: 1px solid rgba(0, 255, 136, 0.15);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3), inset 0 -1px 0 rgba(0, 255, 136, 0.1);
}

.hud-topbar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
}
.hud-topbar-brand .brand-icon {
    font-size: 28px;
    filter: drop-shadow(0 0 12px rgba(0, 255, 136, 0.5));
    animation: brand-glow 3s ease-in-out infinite;
}
@keyframes brand-glow {
    0%, 100% { filter: drop-shadow(0 0 8px rgba(0, 255, 136, 0.4)); }
    50% { filter: drop-shadow(0 0 16px rgba(0, 255, 136, 0.6)); }
}
.hud-topbar-brand .brand-text {
    font-size: 18px;
    font-weight: 700;
    color: #00FF88;
    letter-spacing: 0.05em;
    text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
}
.hud-topbar-brand .brand-sub {
    font-size: 10px;
    color: rgba(0,255,136,0.7);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-left: 4px;
}

.hud-topbar-nav {
    display: flex;
    gap: 4px;
    align-items: center;
}
.hud-nav-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 13.5px;
    font-weight: 500;
    color: #00FF88;
    cursor: pointer;
    transition: all 0.25s ease;
    border: 1px solid transparent;
    position: relative;
}
.hud-nav-item:hover {
    color: #00FF88;
    background: rgba(0, 255, 136, 0.06);
}
.hud-nav-item.active {
    color: #00FF88;
    background: rgba(0, 255, 136, 0.08);
    border-color: rgba(0, 255, 136, 0.2);
    font-weight: 600;
    text-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
}
.hud-nav-item.active::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 20%;
    right: 20%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00FF88, transparent);
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
}
.hud-nav-item .nav-icon {
    font-size: 16px;
    color: inherit;
}
.hud-nav-item span {
    color: inherit;
}

.hud-topbar-status {
    display: flex;
    align-items: center;
    gap: 16px;
}
.hud-status-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11.5px;
    color: rgba(0,255,136,0.75);
}
.hud-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00FF88;
    box-shadow: 0 0 8px rgba(0, 255, 136, 0.6);
    animation: status-pulse 2s infinite;
}
@keyframes status-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(0, 255, 136, 0.6); }
    50% { opacity: 0.6; box-shadow: 0 0 4px rgba(0, 255, 136, 0.3); }
}

/* ============================
   PAGE CONTENT AREA
   ============================ */
.hud-page-content {
    position: relative;
    z-index: 1;
    padding: 24px 32px;
    min-height: calc(100vh - 64px);
}

/* ============================
   HUD CARD — 通用卡片样式
   ============================ */
.hud-card {
    background: linear-gradient(135deg, rgba(15, 30, 54, 0.8) 0%, rgba(10, 22, 40, 0.9) 100%);
    border: 1px solid rgba(0, 255, 136, 0.12);
    border-radius: 12px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.hud-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.4), transparent);
}
.hud-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 12px; height: 12px;
    border-top: 2px solid #00FF88;
    border-left: 2px solid #00FF88;
    border-radius: 2px 0 0 0;
    opacity: 0.6;
}
.hud-card:hover {
    border-color: rgba(0, 255, 136, 0.25);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(0, 255, 136, 0.1);
    transform: translateY(-2px);
}
.hud-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px 12px 20px;
    border-bottom: 1px solid rgba(0, 255, 136, 0.08);
}
.hud-card-title {
    font-size: 13px;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hud-card-title .title-icon {
    font-size: 16px;
}
.hud-card-body {
    padding: 20px;
}

/* ============================
   KPI CARDS — 数据指标卡
   ============================ */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(15, 30, 54, 0.8) 0%, rgba(10, 22, 40, 0.9) 100%);
    border: 1px solid rgba(0, 255, 136, 0.12);
    border-radius: 12px;
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
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.4), transparent);
}
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 10px; height: 10px;
    border-top: 2px solid #00FF88;
    border-left: 2px solid #00FF88;
    opacity: 0.5;
}
.kpi-card:hover {
    border-color: rgba(0, 255, 136, 0.25);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}
.kpi-card .kpi-label {
    font-size: 10.5px;
    font-weight: 600;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
}
.kpi-card .kpi-value {
    font-size: 34px;
    font-weight: 800;
    color: #00FF88;
    line-height: 1.1;
    letter-spacing: -0.02em;
    text-shadow: 0 0 24px rgba(0, 255, 136, 0.3);
    font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
}
.kpi-card:nth-child(even) .kpi-value {
    color: #FFFFFF;
    text-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
}
.kpi-card .kpi-sub {
    font-size: 11.5px;
    color: rgba(255,255,255,0.5);
    margin-top: 8px;
}
.kpi-card .kpi-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #00FF88;
    display: inline-block;
    margin-right: 6px;
    animation: kpi-pulse 2.5s infinite;
    box-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
}
@keyframes kpi-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.kpi-value.warn { color: #FFB800 !important; text-shadow: 0 0 20px rgba(255, 184, 0, 0.25) !important; }
.kpi-value.ok { color: #00FF88 !important; text-shadow: 0 0 20px rgba(0, 255, 136, 0.3) !important; }
.kpi-value.muted { color: rgba(255,255,255,0.4) !important; font-weight: 400 !important; text-shadow: none !important; }

/* ============================
   ENGINE DIAGRAM — 3D柴油机示意图
   ============================ */
.engine-diagram-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    min-height: 320px;
    position: relative;
}
.engine-svg-wrapper {
    position: relative;
    width: 100%;
    max-width: 360px;
    perspective: 1000px;
}
.engine-svg-wrapper svg {
    width: 100%;
    height: auto;
    filter: drop-shadow(0 0 20px rgba(0, 255, 136, 0.15));
    transform: rotateY(-5deg) rotateX(3deg);
    transition: transform 0.5s ease;
}
.engine-svg-wrapper:hover svg {
    transform: rotateY(0deg) rotateX(0deg);
}
.engine-part {
    cursor: pointer;
    transition: all 0.3s ease;
}
.engine-part:hover {
    filter: brightness(1.3) drop-shadow(0 0 8px rgba(0, 255, 136, 0.5));
}
.engine-part-label {
    font-size: 9px;
    fill: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    pointer-events: none;
}
.diagram-hint {
    margin-top: 16px;
    font-size: 11px;
    color: rgba(255,255,255,0.35);
    text-align: center;
}
.diagram-hint span {
    color: #00FF88;
}

/* ============================
   SYSTEM GRID — 系统状态网格
   ============================ */
.system-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}
.system-mini-card {
    background: rgba(0, 255, 136, 0.03);
    border: 1px solid rgba(0, 255, 136, 0.1);
    border-radius: 10px;
    padding: 14px 16px;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.system-mini-card:hover {
    background: rgba(0, 255, 136, 0.08);
    border-color: rgba(0, 255, 136, 0.25);
    transform: translateY(-2px);
}
.system-mini-card .sys-icon {
    font-size: 20px;
    margin-bottom: 8px;
}
.system-mini-card .sys-name {
    font-size: 12.5px;
    font-weight: 600;
    color: #FFFFFF;
    margin-bottom: 4px;
}
.system-mini-card .sys-count {
    font-size: 10.5px;
    color: rgba(255,255,255,0.4);
}
.system-mini-card .sys-status-bar {
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: #00FF88;
    opacity: 0.6;
}

/* ============================
   CONTENT ROW (Chart + Events)
   ============================ */
.dash-content-row {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 20px;
    min-height: 380px;
}
.dash-chart-area {
    background: linear-gradient(135deg, rgba(15, 30, 54, 0.8) 0%, rgba(10, 22, 40, 0.9) 100%);
    border: 1px solid rgba(0, 255, 136, 0.12);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
}
.dash-chart-area::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.4), transparent);
}
.chart-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}
.chart-title {
    font-size: 14px;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 0.03em;
    display: flex;
    align-items: center;
    gap: 8px;
}
.chart-tabs {
    display: flex;
    gap: 2px;
    background: rgba(0, 255, 136, 0.04);
    padding: 3px;
    border-radius: 8px;
    border: 1px solid rgba(0, 255, 136, 0.1);
}
.chart-tab {
    font-size: 11px;
    font-weight: 500;
    color: rgba(255,255,255,0.5);
    padding: 4px 12px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s ease;
}
.chart-tab:hover {
    color: #FFFFFF;
}
.chart-tab.active {
    color: #0A1628;
    background: #00FF88;
    box-shadow: 0 2px 8px rgba(0, 255, 136, 0.3);
    font-weight: 600;
}
.chart-placeholder {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: rgba(255,255,255,0.35);
    font-size: 12.5px;
    gap: 12px;
}

.dash-events-area {
    background: linear-gradient(135deg, rgba(15, 30, 54, 0.8) 0%, rgba(10, 22, 40, 0.9) 100%);
    border: 1px solid rgba(0, 255, 136, 0.12);
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
}
.dash-events-area::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.4), transparent);
}
.events-title {
    font-size: 13px;
    font-weight: 600;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(0, 255, 136, 0.08);
    display: flex;
    align-items: center;
    gap: 8px;
}
.event-item {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    align-items: flex-start;
}
.event-item:last-of-type {
    border-bottom: none;
}
.event-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.event-dot.warn { background: #FFB800; box-shadow: 0 0 8px rgba(255, 184, 0, 0.5); }
.event-dot.info { background: #00FF88; box-shadow: 0 0 8px rgba(0, 255, 136, 0.5); }
.event-dot.ok { background: #00CC6A; box-shadow: 0 0 8px rgba(0, 204, 106, 0.5); }
.event-body { flex: 1; }
.event-text {
    font-size: 12.5px;
    color: #FFFFFF;
    margin-bottom: 2px;
}
.event-time {
    font-size: 10.5px;
    color: rgba(255,255,255,0.35);
}
.events-footer {
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid rgba(0, 255, 136, 0.08);
    text-align: right;
}
.events-footer a {
    font-size: 11px;
    color: #00FF88;
    text-decoration: none;
    opacity: 0.8;
}
.events-footer a:hover { opacity: 1; }
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
.hidden-buttons { position: absolute; left: -9999px; top: -9999px; }

/* ── JS Navigation Trigger Textboxes (hidden but in DOM) ── */
#sys-nav-trigger, #page-nav-trigger {
    position: absolute; width: 1px; height: 1px; overflow: hidden;
    opacity: 0; pointer-events: none; z-index: -1;
}

/* ============================
   DETAIL PAGE
   ============================ */
/* ============================
   MONITOR PAGE — 监控详情页
   ============================ */
#breadcrumb-row { margin-bottom: 16px !important; align-items: center !important; }
.detail-breadcrumb {
    display: flex; align-items: center; gap: 12px;
    font-size: 16px; font-weight: 600; color: #FFFFFF;
    padding: 14px 20px;
    background: linear-gradient(135deg, rgba(15, 30, 54, 0.8) 0%, rgba(10, 22, 40, 0.9) 100%);
    border: 1px solid rgba(0, 255, 136, 0.12);
    border-radius: 10px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    position: relative;
}
.detail-breadcrumb::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 255, 136, 0.4), transparent);
}
.breadcrumb-icon { font-size: 20px; }
.breadcrumb-name { letter-spacing: 0.02em; }
.breadcrumb-badge {
    font-size: 10.5px; font-weight: 600;
    padding: 3px 10px; border-radius: 12px;
    background: rgba(0, 255, 136, 0.08);
    border: 1px solid;
    letter-spacing: 0.05em; margin-left: 6px;
    text-transform: uppercase;
}
#back-overview-btn {
    background: rgba(15, 30, 54, 0.8) !important;
    border: 1px solid rgba(0, 255, 136, 0.12) !important;
    color: rgba(255,255,255,0.6) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    padding: 9px 18px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
    backdrop-filter: blur(10px);
}
#back-overview-btn:hover {
    border-color: rgba(0, 255, 136, 0.3) !important;
    color: #00FF88 !important;
    background: rgba(0, 255, 136, 0.06) !important;
}

/* ============================
   PANEL TITLES
   ============================ */
.panel-title {
    font-size: 11.5px; font-weight: 600; color: #1E293B;
    margin-bottom: 12px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(0, 255, 136, 0.15);
    letter-spacing: 0.1em; text-transform: uppercase;
}

/* ============================
   PARAMETER CARDS — 参数卡片
   ============================ */
.param-cards-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
    margin-top: 16px;
}
.param-card {
    background: rgba(0, 255, 136, 0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid var(--sys-color, #00FF88);
    border-radius: 8px;
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
}
.param-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 6px; height: 6px;
    border-top: 1.5px solid #00FF88;
    border-left: 1.5px solid #00FF88;
    opacity: 0.5;
}
.param-card .param-name {
    font-size: 10.5px;
    color: rgba(255,255,255,0.45);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.param-card .param-value {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    font-family: 'SF Mono', 'Consolas', monospace;
    line-height: 1;
}
.param-card .param-unit {
    font-size: 11px;
    color: rgba(255,255,255,0.4);
    margin-left: 3px;
    font-weight: 400;
}
.param-card .param-baseline {
    font-size: 10px;
    color: rgba(255,255,255,0.3);
    margin-top: 4px;
}
.param-card.ok .param-value { color: #00FF88; text-shadow: 0 0 10px rgba(0, 255, 136, 0.3); }
.param-card.warn .param-value { color: #FFB800; text-shadow: 0 0 10px rgba(255, 184, 0, 0.3); }
.param-card.alert .param-value { color: #FF4757; text-shadow: 0 0 10px rgba(255, 71, 87, 0.3); }

/* ============================
   GRADIO OVERRIDES — HUD风格
   ============================ */
.gr-button-primary {
    background: linear-gradient(135deg, #00FF88 0%, #00CC6A 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    color: #0A1628 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 16px rgba(0, 255, 136, 0.25) !important;
    letter-spacing: 0.02em !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(0, 255, 136, 0.4) !important;
}
.gr-button {
    background: rgba(15, 30, 54, 0.8) !important;
    border: 1px solid rgba(0, 255, 136, 0.12) !important;
    color: #FFFFFF !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.gr-button:hover {
    background: rgba(0, 255, 136, 0.06) !important;
    border-color: rgba(0, 255, 136, 0.3) !important;
    color: #00FF88 !important;
}
.gr-textbox textarea, .gr-textbox input {
    background: rgba(15, 30, 54, 0.8) !important;
    border: 1px solid rgba(0, 255, 136, 0.12) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
.gr-textbox textarea:focus, .gr-textbox input:focus {
    border-color: #00FF88 !important;
    box-shadow: 0 0 0 3px rgba(0, 255, 136, 0.1) !important;
    outline: none !important;
}
.gr-slider input[type="range"] { accent-color: #00FF88; }
.gr-dropdown {
    background: rgba(15, 30, 54, 0.8) !important;
    border: 1px solid rgba(0, 255, 136, 0.12) !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}
/* 聊天框固定高度，对话不撑开布局 */
.chat-col {
    flex-shrink: 0 !important;
}
.chat-card {
    display: flex !important;
    flex-direction: column !important;
}
.chat-card > div:first-of-type {
    flex-shrink: 0 !important;
}
#chatbot {
    height: 420px !important;
    max-height: 420px !important;
    min-height: 420px !important;
    overflow: hidden !important;
    flex-shrink: 0 !important;
}
#chatbot > div {
    height: 420px !important;
    max-height: 420px !important;
    overflow-y: auto !important;
}
/* 之前的 .gr-chatbot 规则覆盖 */
.gr-chatbot {
    border-radius: 10px !important;
    border: 1px solid rgba(0, 255, 136, 0.12) !important;
    background: rgba(15, 30, 54, 0.5) !important;
    height: 420px !important;
    max-height: 420px !important;
    min-height: 420px !important;
    overflow: hidden !important;
    flex-shrink: 0 !important;
}
.gr-chatbot .message.user {
    background: linear-gradient(135deg, #00FF88 0%, #00CC6A 100%) !important;
    color: #0A1628 !important;
    border-radius: 10px 10px 4px 10px !important;
    font-weight: 500 !important;
}
.gr-chatbot .message.bot {
    background: rgba(0, 255, 136, 0.03) !important;
    color: #FFFFFF !important;
    border-radius: 10px 10px 10px 4px !important;
    border: 1px solid rgba(0, 255, 136, 0.08) !important;
}
label, .label-text {
    color: rgba(255,255,255,0.5) !important;
    font-weight: 500 !important;
    font-size: 11px !important;
    letter-spacing: 0.03em;
}
.gr-markdown { color: #FFFFFF !important; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: #FFFFFF !important; }
.gr-tabs { border: none !important; }
.gr-tab {
    background: rgba(15, 30, 54, 0.5) !important;
    border: 1px solid rgba(0, 255, 136, 0.08) !important;
    color: rgba(255,255,255,0.5) !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
.gr-tab.selected {
    background: rgba(15, 30, 54, 0.8) !important;
    color: #00FF88 !important;
    border-color: rgba(0, 255, 136, 0.2) !important;
    border-bottom-color: rgba(15, 30, 54, 0.8) !important;
}
#system-radio label {
    display: block !important;
    padding: 10px 14px !important;
    margin: 4px 0 !important;
    background: rgba(15, 30, 54, 0.6) !important;
    border: 1px solid rgba(0, 255, 136, 0.08) !important;
    border-left: 3px solid rgba(0, 255, 136, 0.15) !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    color: rgba(255,255,255,0.6) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
}
#system-radio label:hover {
    border-color: rgba(0, 255, 136, 0.25) !important;
    color: #FFFFFF !important;
    background: rgba(0, 255, 136, 0.04) !important;
}
#system-radio input:checked + label {
    border-color: #00FF88 !important;
    border-left-color: #00FF88 !important;
    background: rgba(0, 255, 136, 0.08) !important;
    color: #00FF88 !important;
    font-weight: 600 !important;
    box-shadow: 0 0 0 2px rgba(0, 255, 136, 0.05) !important;
}

/* 隐藏Gradio默认footer */
footer { display: none !important; }

/* ============================
   HIDDEN UTILITY
   ============================ */
.hidden-buttons { position: absolute; left: -9999px; top: -9999px; }

/* ── JS Navigation Trigger Textboxes (hidden but in DOM) ── */
#sys-nav-trigger, #page-nav-trigger {
    position: absolute; width: 1px; height: 1px; overflow: hidden;
    opacity: 0; pointer-events: none; z-index: -1;
}

/* ============================
   STATUS BAR
   ============================ */
.status-bar {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 10px 16px;
    background: rgba(0, 255, 136, 0.02);
    border: 1px solid rgba(0, 255, 136, 0.08);
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 11.5px;
}
.status-bar-item {
    display: flex;
    align-items: center;
    gap: 6px;
    color: rgba(255,255,255,0.5);
}
.status-bar-item strong {
    color: #FFFFFF;
    font-weight: 600;
    font-family: 'SF Mono', 'Consolas', monospace;
}

/* Hide JS bridge trigger Textboxes */
#repair-trigger, #resolve-trigger, #refresh-trigger {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
    overflow: hidden !important;
}

/* 时间范围 Radio — 强亮度覆盖 */
.time-range-group,
.time-range-group * {
    color: #FFFFFF !important;
}
.time-range-group label,
.time-range-group span,
.time-range-group legend,
.time-range-group button,
.time-range-group input + label {
    color: #FFFFFF !important;
    opacity: 1 !important;
}
.time-range-group input[type="radio"]:checked + label {
    color: #00FF88 !important;
    font-weight: 700;
}

/* ═════ 快捷模板 & 操作按钮 ═════ */
.chat-section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(0, 255, 136, 0.45);
    margin: 14px 0 8px 2px;
    font-weight: 600;
    border-left: 2px solid rgba(0, 255, 136, 0.3);
    padding-left: 8px;
}
/* 模板按钮 — 玻璃拟态霓虹 */
.tpl-btns button,
.action-btns button {
    background: rgba(0, 255, 136, 0.22) !important;
    border: 1px solid rgba(0, 255, 136, 0.45) !important;
    color: #0A1628 !important;
    border-radius: 6px !important;
    padding: 5px 10px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}
.tpl-btns button:hover,
.action-btns button:hover {
    background: rgba(0, 255, 136, 0.18) !important;
    border-color: #00FF88 !important;
    color: #FFFFFF !important;
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.35);
    transform: translateY(-1px);
}
.tpl-btns button:active,
.action-btns button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 0 6px rgba(0, 255, 136, 0.1) !important;
}
/* 保存测试按钮 — 渐变更炫酷 */
.save-row button {
    background: linear-gradient(135deg, #00FF88 0%, #00CC6A 100%) !important;
    border: none !important;
    color: #0A1628 !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    box-shadow: 0 0 16px rgba(0, 255, 136, 0.25) !important;
    transition: all 0.25s ease !important;
}
.save-row button:hover {
    box-shadow: 0 0 28px rgba(0, 255, 136, 0.45) !important;
    transform: translateY(-1px);
}
.save-row input {
    background: rgba(0, 255, 136, 0.03) !important;
    border: 1px solid rgba(0, 255, 136, 0.12) !important;
    color: #7EC8A0 !important;
    border-radius: 6px !important;
    font-size: 12px !important;
}

/* ═════ 分类筛选 Radio — 炫酷玻璃态 ═════ */
.history-filter {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
}
.history-filter legend,
.history-filter .gr-radio-label {
    display: none !important;
}
.history-filter label {
    padding: 7px 18px !important;
    border-radius: 8px !important;
    background: rgba(0, 255, 136, 0.06) !important;
    border: 1px solid rgba(0, 255, 136, 0.2) !important;
    color: #1E293B !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.history-filter label:hover {
    background: rgba(0, 255, 136, 0.14) !important;
    border-color: rgba(0, 255, 136, 0.45) !important;
    color: #0A1628 !important;
    box-shadow: 0 0 14px rgba(0, 255, 136, 0.15) !important;
}
.history-filter input:checked + label,
.history-filter input:checked ~ span {
    color: #0A1628 !important;
}
.history-filter label:has(input:checked),
.history-filter input[type="radio"]:checked + label {
    background: rgba(0, 255, 136, 0.22) !important;
    border-color: #00FF88 !important;
    color: #0A1628 !important;
    box-shadow: 0 0 18px rgba(0, 255, 136, 0.25) !important;
    font-weight: 700 !important;
}
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


def build_engine_diagram_svg() -> str:
    """构建3D柴油机剖面SVG示意图 — MAN B&W 12K98ME-C7 二冲程十字头低速机"""
    return '''
    <svg viewBox="0 0 500 500" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <!-- 金属质感渐变 -->
        <linearGradient id="metalDark" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#1e3350"/>
          <stop offset="50%" stop-color="#253d5e"/>
          <stop offset="100%" stop-color="#162540"/>
        </linearGradient>
        <linearGradient id="metalLight" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#253d5e"/>
          <stop offset="50%" stop-color="#304a6e"/>
          <stop offset="100%" stop-color="#1e3350"/>
        </linearGradient>
        <linearGradient id="pistonGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#2a4060"/>
          <stop offset="100%" stop-color="#1a2d47"/>
        </linearGradient>
        <linearGradient id="coolantGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="rgba(59,130,246,0.15)"/>
          <stop offset="100%" stop-color="rgba(59,130,246,0.05)"/>
        </linearGradient>
        <linearGradient id="oilGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="rgba(245,158,11,0.12)"/>
          <stop offset="100%" stop-color="rgba(245,158,11,0.03)"/>
        </linearGradient>
        <linearGradient id="glowGrad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#00FF88" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="#00FF88" stop-opacity="0"/>
        </linearGradient>
        <linearGradient id="turboGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#1e3350"/>
          <stop offset="100%" stop-color="#121f33"/>
        </linearGradient>
        <linearGradient id="exhaustGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="#2a2035"/>
          <stop offset="50%" stop-color="#352840"/>
          <stop offset="100%" stop-color="#1e1828"/>
        </linearGradient>
        <!-- 辉光滤镜 -->
        <filter id="glow">
          <feGaussianBlur stdDeviation="2" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <filter id="softGlow">
          <feGaussianBlur stdDeviation="3" result="b"/>
          <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
        <!-- 扫描线动画 -->
        <linearGradient id="scanLine" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="rgba(0,255,136,0.06)"/>
          <stop offset="50%" stop-color="rgba(0,255,136,0.0)"/>
          <stop offset="100%" stop-color="rgba(0,255,136,0.06)"/>
        </linearGradient>
      </defs>

      <!-- ───── 背景网格 ───── -->
      <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,255,136,0.03)" stroke-width="0.5"/>
      </pattern>
      <rect x="0" y="0" width="500" height="500" fill="url(#grid)" rx="8"/>

      <!-- ========== ⛽ 燃烧系统 · 气缸 (中心剖面) ========== -->
      <g class="engine-part" onclick="navigateSystem('combustion')" style="cursor:pointer">

        <!-- 气缸盖 -->
        <rect x="168" y="62" width="84" height="28" rx="3" fill="url(#metalLight)" stroke="rgba(0,255,136,0.35)" stroke-width="1.2"/>
        <rect x="172" y="66" width="76" height="20" rx="2" fill="#1a2a42" stroke="rgba(0,255,136,0.15)" stroke-width="0.5"/>

        <!-- 缸套 (气缸壁) — 左右两侧 -->
        <rect x="168" y="90" width="14" height="210" rx="1" fill="url(#metalDark)" stroke="rgba(0,255,136,0.25)" stroke-width="1"/>
        <rect x="238" y="90" width="14" height="210" rx="1" fill="url(#metalDark)" stroke="rgba(0,255,136,0.25)" stroke-width="1"/>
        <rect x="168" y="90" width="84" height="210" rx="0" fill="none" stroke="rgba(0,255,136,0.1)" stroke-width="0.5"/>

        <!-- 活塞 -->
        <rect x="182" y="170" width="56" height="22" rx="2" fill="url(#pistonGrad)" stroke="rgba(0,255,136,0.4)" stroke-width="1.2"/>
        <!-- 活塞环槽 -->
        <line x1="180" y1="173" x2="240" y2="173" stroke="rgba(0,255,136,0.25)" stroke-width="0.8"/>
        <line x1="180" y1="176" x2="240" y2="176" stroke="rgba(0,255,136,0.2)" stroke-width="0.8"/>
        <line x1="180" y1="179" x2="240" y2="179" stroke="rgba(0,255,136,0.2)" stroke-width="0.8"/>
        <!-- 活塞裙 -->
        <rect x="186" y="192" width="48" height="16" rx="1" fill="#1a2d47" stroke="rgba(0,255,136,0.3)" stroke-width="0.8"/>

        <!-- 活塞杆 -->
        <rect x="205" y="208" width="10" height="45" fill="#1a2d47" stroke="rgba(0,255,136,0.3)" stroke-width="0.8"/>

        <!-- 十字头 -->
        <rect x="192" y="253" width="36" height="14" rx="2" fill="#253d5e" stroke="rgba(0,255,136,0.35)" stroke-width="1"/>
        <rect x="184" y="256" width="52" height="8" rx="2" fill="#1e3350" stroke="rgba(0,255,136,0.2)" stroke-width="0.5"/>
        <!-- 十字头导板 -->
        <line x1="172" y1="259" x2="182" y2="259" stroke="rgba(0,255,136,0.3)" stroke-width="1.5"/>
        <line x1="238" y1="259" x2="248" y2="259" stroke="rgba(0,255,136,0.3)" stroke-width="1.5"/>

        <!-- 连杆 -->
        <line x1="210" y1="267" x2="200" y2="335" stroke="#2a4060" stroke-width="12" stroke-linecap="round"/>
        <line x1="210" y1="267" x2="200" y2="335" stroke="rgba(0,255,136,0.2)" stroke-width="12" stroke-linecap="round" stroke-dasharray="1,3"/>

        <!-- 曲轴 -->
        <circle cx="195" cy="348" r="18" fill="#1a2d47" stroke="rgba(0,255,136,0.4)" stroke-width="1.5"/>
        <circle cx="195" cy="348" r="6" fill="#0d1a2d" stroke="rgba(0,255,136,0.3)" stroke-width="1"/>
        <circle cx="195" cy="348" r="2" fill="rgba(0,255,136,0.5)"/>

        <!-- 飞轮 (右侧) -->
        <ellipse cx="295" cy="348" rx="30" ry="30" fill="none" stroke="rgba(0,255,136,0.2)" stroke-width="6" stroke-dasharray="8,4"/>
        <ellipse cx="295" cy="348" rx="24" ry="24" fill="none" stroke="rgba(0,255,136,0.15)" stroke-width="1.5"/>

        <!-- 缸内燃烧火焰示意 -->
        <ellipse cx="210" cy="165" rx="20" ry="5" fill="rgba(255,120,30,0.04)" filter="url(#softGlow)"/>
        <ellipse cx="210" cy="163" rx="10" ry="3" fill="rgba(255,180,30,0.06)"/>

        <!-- 标签 -->
        <text x="210" y="405" text-anchor="middle" class="engine-part-label">🔥 燃烧系统 · 气缸 — 十字头式二冲程</text>
      </g>

      <!-- ========== ⛽ 高压油泵 & 喷油器 ========== -->
      <g class="engine-part" onclick="navigateSystem('fuel')" style="cursor:pointer">
        <!-- 喷油器 -->
        <rect x="198" y="72" width="6" height="16" rx="1" fill="#253d5e" stroke="rgba(236,72,153,0.5)" stroke-width="1"/>
        <rect x="216" y="72" width="6" height="16" rx="1" fill="#253d5e" stroke="rgba(236,72,153,0.5)" stroke-width="1"/>
        <!-- 高压油管 -->
        <path d="M201 72 Q201 52 180 45 L180 35" fill="none" stroke="rgba(236,72,153,0.5)" stroke-width="1.5"/>
        <path d="M219 72 Q219 52 240 45 L240 35" fill="none" stroke="rgba(236,72,153,0.5)" stroke-width="1.5"/>
        <!-- 高压油泵 -->
        <rect x="170" y="30" width="20" height="14" rx="2" fill="#1e3350" stroke="rgba(236,72,153,0.6)" stroke-width="1.2"/>
        <rect x="230" y="30" width="20" height="14" rx="2" fill="#1e3350" stroke="rgba(236,72,153,0.6)" stroke-width="1.2"/>
        <text x="210" y="430" text-anchor="middle" class="engine-part-label" style="fill:rgba(236,72,153,0.75)">⛽ 高压共轨 · 油耗监测</text>
      </g>

      <!-- ========== 🌡️ 排气系统 ========== -->
      <g class="engine-part" onclick="navigateSystem('exhaust')" style="cursor:pointer">
        <!-- 排气阀液压执行器 -->
        <rect x="198" y="26" width="24" height="18" rx="2" fill="url(#exhaustGrad)" stroke="rgba(255,184,0,0.5)" stroke-width="1"/>
        <rect x="202" y="30" width="16" height="10" rx="1" fill="#1e1828" stroke="rgba(255,184,0,0.3)" stroke-width="0.5"/>
        <!-- 排气阀杆 -->
        <rect x="207" y="44" width="6" height="24" fill="#253d5e" stroke="rgba(255,184,0,0.4)" stroke-width="0.8"/>
        <!-- 排气阀盘 -->
        <rect x="196" y="66" width="28" height="4" rx="1" fill="#2a2035" stroke="rgba(255,184,0,0.5)" stroke-width="1"/>
        <!-- 排气道 -->
        <path d="M196 68 Q170 68 170 54 L90 54" fill="none" stroke="rgba(255,184,0,0.5)" stroke-width="5" stroke-linecap="round"/>
        <!-- 排气总管 -->
        <rect x="85" y="48" width="90" height="14" rx="3" fill="url(#exhaustGrad)" stroke="rgba(255,184,0,0.6)" stroke-width="1"/>
        <!-- 到增压器排气进口 -->
        <path d="M175 55 L175 35 L370 40" fill="none" stroke="rgba(255,184,0,0.4)" stroke-width="3" stroke-dasharray="5,3"/>
        <text x="130" y="40" text-anchor="middle" class="engine-part-label" style="fill:rgba(255,184,0,0.7);font-size:9px">🌡️ 排气系统 · 液压排气阀</text>
      </g>

      <!-- ========== 🔄 涡轮增压器 ========== -->
      <g class="engine-part" onclick="navigateSystem('turbo')" style="cursor:pointer">
        <!-- 涡轮壳 (右) -->
        <path d="M395 50 Q425 45 435 65 Q445 95 430 115 Q410 130 390 125" 
              fill="url(#turboGrad)" stroke="rgba(139,92,246,0.6)" stroke-width="1.5"/>
        <!-- 涡轮叶片 -->
        <circle cx="415" cy="85" r="10" fill="none" stroke="rgba(139,92,246,0.4)" stroke-width="1"/>
        <line x1="415" y1="75" x2="415" y2="95" stroke="rgba(139,92,246,0.3)" stroke-width="0.5"/>
        <line x1="405" y1="85" x2="425" y2="85" stroke="rgba(139,92,246,0.3)" stroke-width="0.5"/>
        <line x1="408" y1="78" x2="422" y2="92" stroke="rgba(139,92,246,0.3)" stroke-width="0.5"/>
        <line x1="422" y1="78" x2="408" y2="92" stroke="rgba(139,92,246,0.3)" stroke-width="0.5"/>
        <!-- 压气机壳 (左) -->
        <path d="M350 80 Q340 60 350 40 Q365 30 385 35 Q395 45 395 65 Q395 85 390 95"
              fill="url(#turboGrad)" stroke="rgba(139,92,246,0.5)" stroke-width="1.2"/>
        <!-- 压气机叶轮 -->
        <circle cx="378" cy="62" r="8" fill="none" stroke="rgba(139,92,246,0.35)" stroke-width="1"/>
        <line x1="378" y1="54" x2="378" y2="70" stroke="rgba(139,92,246,0.3)" stroke-width="0.5"/>
        <line x1="370" y1="62" x2="386" y2="62" stroke="rgba(139,92,246,0.3)" stroke-width="0.5"/>
        <!-- 转子轴 -->
        <line x1="386" y1="75" x2="410" y2="85" stroke="rgba(139,92,246,0.5)" stroke-width="2"/>
        <!-- 进气口 -->
        <path d="M350 35 L335 25 L325 25" fill="none" stroke="rgba(139,92,246,0.4)" stroke-width="3" stroke-linecap="round"/>
        <!-- 扫气出口 → 扫气箱 -->
        <path d="M355 90 Q340 110 330 120" fill="none" stroke="rgba(139,92,246,0.3)" stroke-width="2.5" stroke-dasharray="4,2"/>
        <text x="395" y="148" text-anchor="middle" class="engine-part-label" style="fill:rgba(139,92,246,0.75);font-size:9px">🔄 废气涡轮增压器</text>
      </g>

      <!-- ========== 💨 扫气系统 ========== -->
      <g class="engine-part" onclick="navigateSystem('scavenge')" style="cursor:pointer">
        <!-- 扫气箱 -->
        <rect x="140" y="125" width="25" height="95" rx="3" fill="#152540" stroke="rgba(16,185,129,0.4)" stroke-width="1"/>
        <!-- 扫气口 (缸套上的孔) -->
        <rect x="166" y="155" width="4" height="12" fill="rgba(16,185,129,0.25)"/>
        <rect x="166" y="175" width="4" height="12" fill="rgba(16,185,129,0.25)"/>
        <rect x="166" y="195" width="4" height="12" fill="rgba(16,185,129,0.25)"/>
        <!-- 扫气口右侧 -->
        <rect x="250" y="155" width="4" height="12" fill="rgba(16,185,129,0.25)"/>
        <rect x="250" y="175" width="4" height="12" fill="rgba(16,185,129,0.25)"/>
        <rect x="250" y="195" width="4" height="12" fill="rgba(16,185,129,0.25)"/>
        <!-- 气流箭头示意 -->
        <text x="120" y="115" text-anchor="middle" class="engine-part-label" style="fill:rgba(16,185,129,0.7);font-size:8px">💨 扫气箱 · 直流扫气</text>
      </g>

      <!-- ========== ❄️ 冷却水套 ========== -->
      <g class="engine-part" onclick="navigateSystem('cooling')" style="cursor:pointer">
        <!-- 左侧水套 -->
        <rect x="152" y="90" width="16" height="120" rx="1" fill="url(#coolantGrad)" stroke="rgba(59,130,246,0.35)" stroke-width="1" stroke-dasharray="3,2"/>
        <!-- 右侧水套 -->
        <rect x="252" y="90" width="16" height="120" rx="1" fill="url(#coolantGrad)" stroke="rgba(59,130,246,0.35)" stroke-width="1" stroke-dasharray="3,2"/>
        <!-- 冷却水管 -->
        <path d="M130 130 L152 130" stroke="rgba(59,130,246,0.5)" stroke-width="2" stroke-linecap="round"/>
        <path d="M130 220 L152 190" stroke="rgba(59,130,246,0.35)" stroke-width="1.5" stroke-linecap="round"/>

        <text x="100" y="98" text-anchor="middle" class="engine-part-label" style="fill:rgba(59,130,246,0.7);font-size:9px">❄️ 缸套冷却水</text>
      </g>

      <!-- ========== 🛢️ 滑油系统 ========== -->
      <g class="engine-part" onclick="navigateSystem('lube')" style="cursor:pointer">
        <!-- 油底壳 -->
        <path d="M160 366 Q170 395 210 400 Q250 395 260 366 Z" fill="url(#oilGrad)" stroke="rgba(245,158,11,0.35)" stroke-width="1.2"/>
        <!-- 油位线 -->
        <line x1="170" y1="380" x2="250" y2="380" stroke="rgba(245,158,11,0.3)" stroke-width="0.8" stroke-dasharray="3,3"/>
        <!-- 主轴承 -->
        <rect x="175" y="360" width="40" height="8" rx="2" fill="#1e3350" stroke="rgba(245,158,11,0.3)" stroke-width="0.8"/>
        <!-- 滑油泵示意 -->
        <rect x="270" y="378" width="16" height="22" rx="2" fill="#1a2d47" stroke="rgba(245,158,11,0.4)" stroke-width="1"/>
        <circle cx="278" cy="387" r="4" fill="none" stroke="rgba(245,158,11,0.4)" stroke-width="0.8"/>

        <text x="210" y="455" text-anchor="middle" class="engine-part-label" style="fill:rgba(245,158,11,0.7);font-size:9px">🛢️ 滑油系统 · 主轴承 · 油底壳</text>
      </g>

      <!-- ========== 背景气缸 (透视深度感) ========== -->
      <g opacity="0.12">
        <rect x="290" y="75" width="50" height="200" rx="2" fill="none" stroke="#00FF88" stroke-width="0.8"/>
        <rect x="300" y="80" width="30" height="15" rx="1" fill="#1a2d47"/>
        <line x1="290" y1="120" x2="340" y2="120" stroke="#00FF88" stroke-width="0.5"/>
        <line x1="290" y1="150" x2="340" y2="150" stroke="#00FF88" stroke-width="0.5"/>
        <line x1="290" y1="200" x2="340" y2="200" stroke="#00FF88" stroke-width="0.5"/>
      </g>
      <g opacity="0.08">
        <rect x="345" y="70" width="40" height="210" rx="2" fill="none" stroke="#00FF88" stroke-width="0.6"/>
        <rect x="352" y="75" width="26" height="12" rx="1" fill="#1a2d47"/>
        <line x1="345" y1="130" x2="385" y2="130" stroke="#00FF88" stroke-width="0.4"/>
        <line x1="345" y1="180" x2="385" y2="180" stroke="#00FF88" stroke-width="0.4"/>
      </g>

      <!-- ── 底部发光线 ── -->
      <rect x="100" y="472" width="300" height="2" rx="1" fill="url(#glowGrad)" filter="url(#glow)"/>

      <!-- ── 型号标签 ── -->
      <text x="250" y="492" text-anchor="middle" style="fill:rgba(0,255,136,0.5);font-size:11px;font-weight:600;letter-spacing:0.18em;">
        MAN B&amp;W 12K98ME-C7 — 缸径 980mm · 冲程 2660mm · 额定功率 69720kW
      </text>

      <!-- ── 扫描线动画覆盖 ── -->
      <rect x="0" y="0" width="500" height="500" fill="url(#scanLine)" rx="8" pointer-events="none">
        <animate attributeName="y" from="-500" to="500" dur="4s" repeatCount="indefinite"/>
      </rect>
    </svg>
    '''


def build_overview_html() -> str:
    """Build HUD-style dashboard overview page."""
    
    health = get_current_health()
    unresolved_count = sum(1 for a in anomalies if a["status"] == "unresolved")
    health_sub_text = "全系统状态良好" if health == 100 else f"检测到 {unresolved_count} 项异常"

    # ── KPI Cards ──
    kpi_cards = (
        '<div class="kpi-card"><div class="kpi-label">监控参数</div>'
        f'<div class="kpi-value">{len(KB_BASELINE)}</div><div class="kpi-sub">18 项全覆盖监测</div></div>'
        '<div class="kpi-card"><div class="kpi-label">负载区间</div>'
        f'<div class="kpi-value">{len(LOADS)}</div><div class="kpi-sub">25% — 110% MCR</div></div>'
        '<div class="kpi-card"><div class="kpi-label">综合健康度</div>'
        f'<div class="kpi-value ok">{health}</div><div class="kpi-sub">{health_sub_text}</div></div>'
        '<div class="kpi-card"><div class="kpi-label">系统状态</div>'
        '<div class="kpi-value ok"><span class="kpi-dot"></span>在线</div><div class="kpi-sub">全系统正常运行</div></div>'
    )
    
    # ── 3D柴油机示意图区域 ──
    engine_diagram = (
        '<div class="hud-card">'
        '  <div class="hud-card-header">'
        '    <div class="hud-card-title"><span class="title-icon">⚙️</span>柴油机系统概览</div>'
        '  </div>'
        '  <div class="hud-card-body" style="padding:12px 20px 20px 20px;">'
        '    <div class="engine-diagram-container">'
        '      <div class="engine-svg-wrapper">'
        + build_engine_diagram_svg() +
        '      </div>'
        '      <div class="diagram-hint">点击 <span>系统部位</span> 跳转到详细监控</div>'
        '    </div>'
        '  </div>'
        '</div>'
    )
    
    # ── 全局趋势 + 事件 ──
    trend_events = (
        '<div class="dash-chart-area">'
        '  <div class="chart-header">'
        '    <span class="chart-title">📈 全局健康趋势</span>'
        '    <div class="chart-tabs">'
        '      <span class="chart-tab active">24h</span>'
        '      <span class="chart-tab">7d</span>'
        '      <span class="chart-tab">30d</span>'
        '    </div>'
        '  </div>'
        '  <div class="chart-placeholder">'
        '    <svg width="100%" height="180" viewBox="0 0 600 180">'
        '      <polyline points="10,130 80,122 150,115 220,100 290,92 360,85 430,78 500,72 570,55" '
        '                fill="none" stroke="#00FF88" stroke-width="2" stroke-linecap="round" opacity="0.7"/>'
        '      <polyline points="10,145 80,138 150,132 220,120 290,112 360,105 430,98 500,92 570,82" '
        '                fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="5,5" opacity="0.5"/>'
        '      <line x1="10" y1="155" x2="570" y2="155" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>'
        '      <text x="10" y="170" fill="rgba(255,255,255,0.35)" font-size="10" font-family="sans-serif">00:00</text>'
        '      <text x="290" y="170" fill="rgba(255,255,255,0.35)" font-size="10" font-family="sans-serif" text-anchor="middle">12:00</text>'
        '      <text x="565" y="170" fill="rgba(255,255,255,0.35)" font-size="10" font-family="sans-serif" text-anchor="end">24:00</text>'
        '    </svg>'
        '    <span style="margin-top:8px;font-size:11px;">选择子系统查看详细趋势图表</span>'
        '  </div>'
        '</div>'
        '<div class="dash-events-area" style="margin-top:16px;">'
        '  <div class="events-title">🚨 最新事件</div>'
        '  <div class="event-item">'
        '    <span class="event-dot ok"></span>'
        '    <div class="event-body">'
        '      <div class="event-text">各子系统就绪</div>'
        '      <div class="event-time">刚刚</div>'
        '    </div>'
        '  </div>'
        '  <div class="event-item">'
        '    <span class="event-dot info"></span>'
        '    <div class="event-body">'
        '      <div class="event-text">冷却系统基准值校准完成</div>'
        '      <div class="event-time">3 天前</div>'
        '    </div>'
        '  </div>'
        '  <div class="event-item">'
        '    <span class="event-dot ok"></span>'
        '    <div class="event-body">'
        '      <div class="event-text">全系统自检通过</div>'
        '      <div class="event-time">7 天前</div>'
        '    </div>'
        '  </div>'
        '</div>'
    )
    
    # ── 系统状态网格 ──
    system_grid = '<div class="system-grid">'
    for sys_name, params in SYSTEM_TABS.items():
        meta = SYSTEM_META[sys_name]
        btn_id = f"sys-card-{meta['btn_elem']}"
        system_grid += (
            f'<div class="system-mini-card" onclick=\'navigateSystem("{meta["btn_elem"]}")\'>'
            f'  <div class="sys-icon">{meta["icon"]}</div>'
            f'  <div class="sys-name">{sys_name}</div>'
            f'  <div class="sys-count">{len(params)} 个监测参数</div>'
            f'  <div class="sys-status-bar"></div>'
            f'</div>'
        )
    system_grid += '</div>'
    
    system_section = (
        '<div class="hud-card" style="margin-top:20px;">'
        '  <div class="hud-card-header">'
        '    <div class="hud-card-title"><span class="title-icon">🗂️</span>子系统监控入口</div>'
        '  </div>'
        '  <div class="hud-card-body">'
        + system_grid +
        '  </div>'
        '  <div style="margin-top:16px;text-align:center;">'
        '    <button class="repair-entry-btn" onclick="navigateToRepair()">'
        '      🔧 系统维修中心'
        '    </button>'
        '  </div>'
        '</div>'
    )
    
    # ── 组装完整页面 ──
    return (
        '<div class="hud-page-content">'
        f'  <div class="kpi-row">{kpi_cards}</div>'
        '  <div class="dash-content-row">'
        + engine_diagram +
        '    <div>'
        + trend_events +
        '    </div>'
        '  </div>'
        + system_section +
        '</div>'
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

# ═══════════════════════════════════════════════════════════════
# Navigation Bridge JS (module-level, injected via launch(js=...))
# ═══════════════════════════════════════════════════════════════
NAV_BRIDGE_JS = """
function _setAndTrigger(elemId, value) {
    var el = document.getElementById(elemId);
    if (!el) return;
    var input = el.querySelector('textarea') || el.querySelector('input');
    if (!input) return;
    var proto = input.tagName === 'TEXTAREA' ? HTMLTextAreaElement : HTMLInputElement;
    var setter = Object.getOwnPropertyDescriptor(proto.prototype, 'value').set;
    setter.call(input, value);
    input.dispatchEvent(new Event('input', {bubbles: true}));
}
window.navigateSystem = function(id) { _setAndTrigger('sys-nav-trigger', id); };
window.navigatePage   = function(pg, el) { document.querySelectorAll('.hud-nav-item').forEach(e => e.classList.remove('active')); el.classList.add('active'); _setAndTrigger('page-nav-trigger', pg); };
window.navigateToRepair = function() { _setAndTrigger('repair-trigger', 'go'); };
window.resolveAnomaly  = function(id) { _setAndTrigger('resolve-trigger', String(id)); };
window.refreshOverview = function() { _setAndTrigger('refresh-trigger', Date.now().toString()); };

// 事件委托：不依赖 DOM 变化，点击事件冒泡到 document 统一处理
document.addEventListener('click', function(e) {
    var repairBtn = e.target.closest('.repair-entry-btn');
    if (repairBtn) {
        e.preventDefault();
        e.stopPropagation();
        _setAndTrigger('repair-trigger', 'go_' + Date.now());
        return;
    }
    var resolveBtn = e.target.closest('.resolve-btn.unresolved');
    if (resolveBtn) {
        e.preventDefault();
        e.stopPropagation();
        var id = resolveBtn.getAttribute('data-id');
        if (id) _setAndTrigger('resolve-trigger', id);
    }
});
// Auto-refresh every 10 seconds
setInterval(function() { _setAndTrigger('refresh-trigger', Date.now().toString()); }, 30000);
"""


def create_ui():
    default_system = "排气系统"

    with gr.Blocks(
        title="Marine Engine AI - 轮机智脑 · 可视化",
        css="""
        #history-btn-row { display:flex !important; gap:8px !important; margin-bottom:6px; align-items:stretch !important; }
        #history-btn-row>:first-child { flex:4 !important; }
        #hist-load-btn,#hist-refresh-btn { flex:1 !important; display:flex !important; align-items:stretch !important; }
        @keyframes btnGlow { 0%,100% { box-shadow:0 0 8px rgba(0,255,136,0.3),0 0 20px rgba(0,255,136,0.08),0 2px 10px rgba(0,0,0,0.15),inset 0 1px 0 rgba(255,255,255,0.08); } 50% { box-shadow:0 0 14px rgba(0,255,136,0.45),0 0 30px rgba(0,255,136,0.14),0 2px 10px rgba(0,0,0,0.15),inset 0 1px 0 rgba(255,255,255,0.1); } }
        #hist-load-btn button,#hist-refresh-btn button { width:100% !important; min-height:42px !important; padding:8px 14px !important; font-size:12.5px !important; font-weight:700 !important; letter-spacing:0.03em !important; color:#00FF88 !important; background:linear-gradient(180deg,rgba(0,20,10,0.95) 0%,rgba(0,40,20,0.9) 100%) !important; border:1.5px solid rgba(0,255,136,0.5) !important; border-radius:10px !important; cursor:pointer !important; transition:all .25s ease !important; text-shadow:0 0 8px rgba(0,255,136,0.5),0 0 2px rgba(0,255,136,0.3) !important; white-space:nowrap !important; animation:btnGlow 3s ease-in-out infinite !important; position:relative !important; overflow:hidden !important; }
        #hist-load-btn button::after,#hist-refresh-btn button::after { content:'' !important; position:absolute !important; inset:1px !important; border-radius:8px !important; background:linear-gradient(180deg,rgba(0,255,136,0.04) 0%,transparent 80%) !important; pointer-events:none !important; }
        #hist-load-btn button:hover,#hist-refresh-btn button:hover { color:#00FF88 !important; background:linear-gradient(180deg,rgba(0,30,15,0.95) 0%,rgba(0,60,30,0.9) 100%) !important; border-color:#00FF88 !important; box-shadow:0 0 20px rgba(0,255,136,0.5),0 0 40px rgba(0,255,136,0.2),0 0 60px rgba(0,255,136,0.08),0 2px 10px rgba(0,0,0,0.2),inset 0 1px 0 rgba(255,255,255,0.12) !important; text-shadow:0 0 12px rgba(0,255,136,0.8),0 0 4px rgba(0,255,136,0.5),0 0 20px rgba(0,255,136,0.3) !important; transform:translateY(-1.5px) !important; animation:none !important; }
        #hist-load-btn button:hover::after,#hist-refresh-btn button:hover::after { background:linear-gradient(180deg,rgba(0,255,136,0.08) 0%,transparent 80%) !important; }
        #hist-load-btn button:active,#hist-refresh-btn button:active { color:#FFF !important; background:rgba(0,255,136,0.25) !important; border-color:#FFF !important; box-shadow:0 0 30px rgba(0,255,136,0.6),0 0 50px rgba(0,255,136,0.25),inset 0 2px 4px rgba(0,0,0,0.3) !important; text-shadow:0 0 16px rgba(0,255,136,1),0 0 30px rgba(0,255,136,0.6) !important; transform:translateY(0) !important; animation:none !important; }
        /* 图片识别页所有文字强制深色 */
        .gradio-container .panel-title,
        .gradio-container #image-page .panel-title,
        div.panel-title { color:#1E293B !important; }
        .gradio-container #image-chart label,
        .gradio-container #image-chatbot label { color:#1E293B !important; font-weight:600; }
        .gradio-container #image-question-input input::placeholder { color:#64748B !important; }
        .gradio-container #image-chart .gr-plot label { color:#1E293B !important; }
        #image-chart label, #image-chatbot label { color:#1E293B !important; font-weight:600; }
        #image-question-input input::placeholder { color:#64748B !important; }
        /* 分析按钮全宽居中 */
        #image-analyze-btn button { width:100% !important; height:44px !important; font-size:15px !important; font-weight:600 !important; margin-top:8px !important; }
        /* 清空结果按钮样式 */
        #image-page .gr-button-secondary { border-color:#CBD5E1 !important; color:#475569 !important; }
        /* 分析结果占位文字 */
        #image-chatbot .placeholder { text-align:center; color:#94A3B8; padding:48px 20px; font-size:14px; line-height:1.8; }
        """,
    ) as demo:

        # ── Hidden State ──
        selected_system_state = gr.State(value=default_system)
        page_state = gr.State(value="overview")

        # ── JS Navigation Triggers (hidden Textboxes, styled via CSS) ──
        sys_nav_trigger = gr.Textbox(
            value="", visible=True, elem_id="sys-nav-trigger",
            label="", show_label=False, container=False,
        )
        page_nav_trigger = gr.Textbox(
            value="", visible=True, elem_id="page-nav-trigger",
            label="", show_label=False, container=False,
        )


        # ═══════════════════════════════
        # GLOBAL TOP NAVBAR — 全局顶部导航
        # ═══════════════════════════════
        top_nav_html = gr.HTML(
            value="""
            <div class="hud-topbar">
                <div class="hud-topbar-brand">
                    <span class="brand-icon">⚓</span>
                    <span class="brand-text">轮机智脑</span>
                    <span class="brand-sub">Marine Engine AI</span>
                </div>
                <div class="hud-topbar-nav">
                    <div class="hud-nav-item active" onclick="navigatePage('overview', this)">
                        <span class="nav-icon">📊</span>
                        <span>总览</span>
                    </div>
                    <div class="hud-nav-item" onclick="navigatePage('monitor', this)">
                        <span class="nav-icon">📈</span>
                        <span>监控</span>
                    </div>
                    <div class="hud-nav-item" onclick="navigatePage('image', this)">
                        <span class="nav-icon">🖼️</span>
                        <span>视觉识别</span>
                    </div>
                    <div class="hud-nav-item" onclick="navigatePage('history', this)">
                        <span class="nav-icon">📂</span>
                        <span>历史数据</span>
                    </div>
                </div>
                <div class="hud-topbar-status">
                    <div class="hud-status-item">
                        <span class="hud-status-dot"></span>
                        <span>系统在线</span>
                    </div>
                    <div class="hud-status-item">
                        <span>12K98ME-C7</span>
                    </div>
                </div>
            </div>
            """,
            elem_id="top-navbar",
        )

        # ═══════════════════════════════
        # GLOBAL HIDDEN BUTTONS — 全局导航隐藏按钮
        # ═══════════════════════════════
        with gr.Row(elem_classes="hidden-buttons", visible=True):
            # 顶部导航按钮
            nav_btn_overview = gr.Button("总览", elem_id="nav-btn-overview", size="sm")
            nav_btn_monitor = gr.Button("监控", elem_id="nav-btn-monitor", size="sm")
            nav_btn_image = gr.Button("视觉识别", elem_id="nav-btn-image", size="sm")
            nav_btn_history = gr.Button("历史数据", elem_id="nav-btn-history", size="sm")
            # 系统卡片跳转按钮
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
        # PAGE 1: 系统总览 Overview
        # ═══════════════════════════════
        with gr.Group(visible=True, elem_id="overview-page") as overview_page:
            overview_html = gr.HTML(
                value=build_overview_html(),
                elem_id="overview-html",
            )
            health_trend_plot = gr.Plot(
                value=build_health_trend_chart(),
                elem_id="health-trend-plot",
                label="",
                show_label=False,
                container=False,
            )

        # ═══════════════════════════════
        # Global JS Bridge Triggers (hidden, always accessible)
        # ═══════════════════════════════
        repair_trigger = gr.Textbox(
            value="", visible=True, elem_id="repair-trigger",
            label="", show_label=False, container=False,
        )
        resolve_trigger = gr.Textbox(
            value="", visible=True, elem_id="resolve-trigger",
            label="", show_label=False, container=False,
        )
        refresh_trigger = gr.Textbox(
            value="", visible=True, elem_id="refresh-trigger",
            label="", show_label=False, container=False,
        )

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
                    gr.HTML('<div class="panel-title" style="color:#1E293B !important;">📤 上传图片</div>')
                    image_input = gr.Image(
                        label="",
                        type="filepath",
                        height=420,
                        elem_id="image-upload",
                    )
                    image_question = gr.Textbox(
                        label="",
                        show_label=False,
                        placeholder="可选：补充问题（如「这是什么型号的增压器？」）…",
                        elem_id="image-question-input",
                    )
                    image_analyze_btn = gr.Button(
                        "🔍 分析图片", variant="primary", size="lg",
                        elem_id="image-analyze-btn",
                    )

                # ── Right: analysis result ──
                with gr.Column(scale=3, min_width=400):
                    gr.HTML('<div class="panel-title" style="color:#1E293B !important;">📋 分析结果</div>')
                    image_result = gr.Chatbot(
                        label="",
                        height=420,
                        elem_id="image-chatbot",
                        placeholder="<div style='text-align:center;color:#94A3B8;padding:40px;'>上传图片并点击「分析图片」后<br>识别结果将在此显示</div>",
                    )
                    image_chart = gr.Plot(
                        label="可视化对比图表",
                        visible=False,
                        elem_id="image-chart",
                    )
                    with gr.Row():
                        image_clear_btn = gr.Button("🗑️ 清空结果", size="sm", scale=1, variant="secondary")
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
            # 子系统详情页 - 蓝金轻奢风格 固定台布局
            gr.HTML("""
            <style>
            /* ========== 全局布局 固定高度 ========== */
            .monitor-page {
                background: #F5F7FA !important;
                color: #1F2937 !important;
                height: calc(100vh - 64px);
                overflow: hidden;
            }
            .monitor-page > .gr-row {
                height: 100%;
                gap: 0 !important;
            }
            
            /* ========== 左侧固定导航栏 ========== */
            .monitor-sidebar {
                background: #FFFFFF !important;
                border-right: 1px solid #EBEEF5;
                padding: 0 !important;
                height: 100%;
                overflow-y: auto;
                box-shadow: 2px 0 8px rgba(0,0,0,0.02);
                display: flex;
                flex-direction: column;
            }
            .sidebar-header {
                padding: 20px 20px 16px;
                font-size: 13px;
                font-weight: 600;
                color: #1E293B;
                border-bottom: 1px solid #F1F5F9;
                letter-spacing: 0.05em;
            }
            .sidebar-nav {
                flex: 1;
                padding: 12px 0;
            }
            #system-radio {
                display: flex;
                flex-direction: column;
                gap: 2px;
                padding: 0 8px;
            }
            #system-radio label {
                padding: 10px 12px;
                border-radius: 6px;
                transition: all 0.2s;
                font-size: 14px;
                color: #606266;
                width: 100%;
                margin: 0;
            }
            #system-radio label:hover {
                background: #F0F5FF;
            }
            #system-radio input:checked + span {
                color: #165DFF !important;
                font-weight: 500;
            }
            #system-radio label:has(input:checked) {
                background: #F0F5FF;
                color: #165DFF;
                border-left: 3px solid #C9A25C;
                padding-left: 9px;
            }
            
            /* 历史数据区 */
            .sidebar-history {
                border-top: 1px solid #E2E8F0;
                padding: 16px;
            }
            .sidebar-history .section-title {
                font-size: 12px;
                font-weight: 600;
                color: #475569;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .sidebar-history .gr-dropdown {
                margin-bottom: 10px !important;
            }
            .sidebar-history .gr-markdown {
                color: #334155 !important;
                font-size: 12px;
            }
            .sidebar-history .gr-markdown code {
                color: #1E293B !important;
                background: #F1F5F9 !important;
            }
            .sidebar-history .gr-dropdown label,
            .sidebar-history .gr-dropdown .wrap,
            .sidebar-history input[type="text"] {
                color: #1E293B !important;
            }
            
            /* ========== 右侧主内容区 ========== */
            .monitor-main-area {
                padding: 20px 24px !important;
                gap: 16px;
                height: 100%;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
            }
            
            /* ========== 顶部工具栏 ========== */
            .top-toolbar {
                background: #FFFFFF;
                border: 1px solid #EBEEF5;
                border-radius: 10px;
                padding: 14px 20px;
                align-items: center;
                box-shadow: 0 2px 12px rgba(0,0,0,0.04);
                display: flex;
            }
            .toolbar-left {
                display: flex;
                align-items: center;
                gap: 16px;
                flex: 1;
            }
            #back-overview-btn {
                border: 1px solid #DCDFE6;
                background: #FFFFFF;
                color: #606266;
                border-radius: 6px;
                transition: all 0.2s;
                font-size: 13px;
                height: 32px;
            }
            #back-overview-btn:hover {
                border-color: #165DFF;
                color: #165DFF;
            }
            .detail-breadcrumb {
                font-size: 18px;
                font-weight: 600;
                color: #1E293B;
            }
            .breadcrumb-badge {
                font-size: 12px;
                padding: 2px 10px;
                border-radius: 10px;
                background: #F0F5FF;
                color: #165DFF;
                margin-left: 10px;
                font-weight: 400;
            }
            .toolbar-right {
                display: flex;
                align-items: center;
                gap: 20px;
            }
            .toolbar-right .gr-slider {
                width: 180px !important;
                margin-bottom: 0 !important;
            }
            .time-range-group {
                margin-bottom: 0 !important;
            }
            .time-range-group label {
                padding: 6px 14px;
                border: 1px solid #DCDFE6;
                border-radius: 4px;
                font-size: 13px;
                color: #606266;
                transition: all 0.2s;
            }
            .time-range-group label:has(input:checked) {
                background: #165DFF;
                border-color: #165DFF;
                color: #FFFFFF !important;
            }
            .time-range-group input:checked + span {
                color: #FFFFFF !important;
            }
            .export-btns {
                display: flex;
                gap: 8px;
            }
            .export-btns button {
                border: 1px solid #DCDFE6;
                background: #FFFFFF;
                color: #606266;
                border-radius: 4px;
                font-size: 13px;
                padding: 6px 12px;
                transition: all 0.2s;
                height: 32px;
            }
            .export-btns button:hover {
                border-color: #165DFF;
                color: #165DFF;
            }
            
            /* ========== 状态条 ========== */
            #status-bar {
                background: #F0F9EB;
                border: 1px solid #E1F3D8;
                border-radius: 8px;
                padding: 10px 16px;
                color: #67C23A;
                font-size: 13px;
            }
            
            /* ========== 主体内容双列 ========== */
            .main-content {
                flex: 1;
                display: flex;
                gap: 16px;
                min-height: 0;
            }
            
            /* ========== 通用卡片样式 - 相册边框 ========== */
            .content-card {
                background: #FFFFFF;
                border: 1px solid #EBEEF5;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.04);
                transition: box-shadow 0.3s ease;
                display: flex;
                flex-direction: column;
            }
            .content-card:hover {
                box-shadow: 0 4px 16px rgba(22,93,255,0.08), 0 0 0 1px rgba(201,162,92,0.15);
            }
            .card-title {
                font-size: 15px;
                font-weight: 600;
                color: #1E293B;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 10px;
                padding-left: 10px;
                border-left: 3px solid #C9A25C;
                flex-shrink: 0;
            }
            
            /* ========== 左列：智能对话 ========== */
            .chat-col {
                width: 380px;
                flex-shrink: 0;
                display: flex;
                flex-direction: column;
            }
            .chat-card {
                flex: 1;
                min-height: 0;
            }
            #chatbot {
                background: #FAFBFC !important;
                border: 1px solid #EBEEF5;
                border-radius: 8px !important;
                padding: 12px !important;
                flex: 1;
                overflow-y: auto;
                min-height: 320px;
            }
            #chatbot .chat-message {
                gap: 10px !important;
                margin-bottom: 12px !important;
            }
            #chatbot .chat-message.user {
                flex-direction: row-reverse !important;
            }
            #chatbot .chat-message.user .message {
                background: #165DFF !important;
                color: #FFFFFF !important;
                border-radius: 12px 12px 4px 12px !important;
                padding: 10px 14px !important;
                border: none !important;
                max-width: 80% !important;
                box-shadow: 0 2px 8px rgba(22,93,255,0.15) !important;
            }
            #chatbot .chat-message.bot .message {
                background: #FFFFFF !important;
                color: #303133 !important;
                border-radius: 12px 12px 12px 4px !important;
                padding: 10px 14px !important;
                border: 1px solid #EBEEF5 !important;
                max-width: 85% !important;
                box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
            }
            
            /* 聊天输入 - 嵌入按钮 */
            .chat-input-row {
                position: relative !important;
                margin-top: 12px !important;
                flex-shrink: 0;
            }
            .chat-input-row .gr-textbox {
                margin-bottom: 0 !important;
                width: 100% !important;
            }
            .chat-input-row .gr-textbox textarea {
                min-height: 56px !important;
                padding: 10px 44px 10px 12px !important;
                resize: none !important;
                border: 1px solid #DCDFE6;
                border-radius: 6px;
            }
            .chat-input-row .gr-textbox textarea:focus {
                border-color: #165DFF;
            }
            .chat-input-row > div:last-child {
                position: absolute !important;
                right: 6px !important;
                bottom: 6px !important;
                z-index: 10 !important;
                width: auto !important;
                min-width: auto !important;
                flex: none !important;
            }
            .chat-input-row button {
                width: 28px !important;
                height: 28px !important;
                min-width: 28px !important;
                min-height: 28px !important;
                padding: 0 !important;
                margin: 0 !important;
                border-radius: 4px !important;
                font-size: 12px !important;
                line-height: 28px !important;
            }
            
            /* 快捷按钮 */
            .chat-section-title {
                font-size: 12px;
                font-weight: 600;
                color: #94A3B8;
                margin: 16px 0 8px;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                flex-shrink: 0;
            }
            .tpl-btns {
                gap: 8px !important;
                flex-shrink: 0;
            }
            .tpl-btns button {
                height: 32px !important;
                flex: 1 !important;
                margin: 0 !important;
                border: 1px solid #DCDFE6;
                background: #FFFFFF;
                color: #606266;
                border-radius: 4px;
                font-size: 12px;
                transition: all 0.2s;
            }
            .tpl-btns button:hover {
                border-color: #165DFF;
                color: #165DFF;
            }
            .action-btns {
                gap: 8px !important;
                flex-shrink: 0;
            }
            .action-btns button {
                height: 32px !important;
                flex: 1 !important;
                margin: 0 !important;
                border: 1px solid #DCDFE6;
                background: #FFFFFF;
                color: #606266;
                border-radius: 4px;
                font-size: 12px;
                transition: all 0.2s;
            }
            .action-btns button:hover {
                border-color: #165DFF;
                color: #165DFF;
            }
            .save-row {
                align-items: stretch !important;
                gap: 8px !important;
                margin-top: 8px !important;
                flex-shrink: 0;
            }
            .save-row .gr-textbox {
                margin-bottom: 0 !important;
            }
            .save-row .gr-textbox input {
                height: 32px !important;
                padding: 0 12px !important;
                border: 1px solid #DCDFE6;
                border-radius: 4px;
            }
            .save-row button {
                height: auto !important;
                min-height: 32px !important;
                align-self: stretch !important;
            }
            
            /* ========== 右列：数据区 ========== */
            .data-col {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 16px;
                min-width: 0;
            }
            
            /* 趋势图卡片 */
            .chart-card {
                flex: 1;
                min-height: 0;
            }
            #main-chart {
                flex: 1;
                min-height: 280px;
            }
            
            /* 参数卡片 */
            #param-cards-html .param-cards-row {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 12px;
                flex-shrink: 0;
            }
            #param-cards-html .param-card {
                background: #FFFFFF;
                border: 1px solid #EBEEF5;
                border-left: 3px solid var(--sys-color, #165DFF);
                border-radius: 8px;
                padding: 14px 16px;
                transition: all 0.25s ease;
                box-shadow: 0 1px 4px rgba(0,0,0,0.03);
            }
            #param-cards-html .param-card:hover {
                border-color: #165DFF;
                background: #FCFDFF;
                box-shadow: 0 3px 12px rgba(22,93,255,0.1), 0 0 0 1px rgba(201,162,92,0.15);
                transform: translateY(-1px);
            }
            #param-cards-html .param-card .param-name {
                font-size: 12px;
                color: #1E293B;
                font-weight: 500;
                margin-bottom: 6px;
            }
            #param-cards-html .param-card .param-value {
                font-size: 20px;
                font-weight: 600;
                color: #1E293B;
                margin-bottom: 4px;
            }
            #param-cards-html .param-card .param-unit {
                font-size: 11px;
                color: #64748B;
                font-weight: 400;
                margin-left: 2px;
            }
            #param-cards-html .param-card .param-baseline {
                font-size: 11px;
                color: #64748B;
            }
            #param-cards-html .param-card.ok .param-value { color: #22C55E; }
            #param-cards-html .param-card.warn .param-value { color: #E6A23C; }
            #param-cards-html .param-card.alert .param-value { color: #F56C6C; }
            
            /* 数据摘要卡片 */
            .summary-card {
                max-height: 200px;
                overflow-y: auto;
            }
            #data-summary-md {
                color: #475569;
                font-size: 13px;
                line-height: 1.8;
            }
            #data-summary-md h1, #data-summary-md h2, #data-summary-md h3 {
                color: #1E293B;
                font-weight: 600;
                margin: 12px 0 8px;
                padding-bottom: 6px;
                border-bottom: 1px solid #F1F5F9;
                font-size: 14px;
            }
            #data-summary-md table {
                width: 100%;
                border-collapse: collapse;
                margin: 10px 0;
                font-size: 12px;
            }
            #data-summary-md th {
                background: #F0F5FF;
                color: #165DFF;
                font-weight: 600;
                padding: 8px 12px;
                text-align: left;
                border-bottom: 2px solid #C9A25C;
            }
            #data-summary-md td {
                padding: 8px 12px;
                border-bottom: 1px solid #F1F5F9;
                color: #475569;
            }
            #data-summary-md tr:hover td {
                background: #FAFBFC;
            }
            #data-summary-md strong {
                color: #1E293B;
                font-weight: 600;
            }
            #data-summary-md ul, #data-summary-md ol {
                padding-left: 20px;
                margin: 6px 0;
            }
            #data-summary-md li {
                margin: 3px 0;
            }
            </style>
            """)

            # 整体左右分栏：左侧固定导航 + 右侧主内容
            with gr.Row(equal_height=False, elem_classes="monitor-page"):
                # ── 左侧边栏 ──
                with gr.Column(scale=1, min_width=200, elem_classes="monitor-sidebar"):
                    gr.HTML('<div class="sidebar-header">📊 子系统监控</div>')
                    
                    with gr.Column(elem_classes="sidebar-nav"):
                        system_radio = gr.Radio(
                            choices=[(f"{SYSTEM_META[s]['icon']} {s}", s) for s in SYSTEM_TABS],
                            value=default_system,
                            label="",
                            elem_id="system-radio",
                        )
                    
                    # 历史数据区
                    with gr.Column(elem_classes="sidebar-history"):
                        gr.HTML('<div class="section-title">历史数据</div>')
                        history_dropdown = gr.Dropdown(
                            choices=list_saved_sessions(), label="",
                            interactive=True,
                        )
                        with gr.Row():
                            refresh_history_btn = gr.Button("🔄 刷新", size="sm", scale=1)
                            load_history_btn = gr.Button("📂 加载", size="sm", scale=1)
                        with gr.Row():
                            delete_history_btn = gr.Button("🗑️ 删除", size="sm", scale=1)
                        history_list_md = gr.Markdown("*暂无历史数据*")

                # ── 右侧主区域 ──
                with gr.Column(scale=6, elem_classes="monitor-main-area"):
                    # 顶部工具栏
                    with gr.Row(elem_classes="top-toolbar"):
                        with gr.Column(scale=3, elem_classes="toolbar-left"):
                            back_to_overview_btn = gr.Button(
                                "← 返回总览", elem_id="back-overview-btn", size="sm",
                            )
                            detail_title_html = gr.HTML(
                                value=build_detail_breadcrumb_html(default_system),
                                elem_id="detail-title-html",
                            )
                        with gr.Column(scale=2, min_width=180):
                            highlight_slider = gr.Slider(
                                minimum=25, maximum=110, value=75, step=1,
                                label="负载高亮 (%)",
                                elem_id="load-slider",
                                show_label=True,
                            )
                        with gr.Column(scale=2, min_width=220, elem_classes="time-range-group"):
                            time_range_radio = gr.Radio(
                                choices=[("1小时", "1h"), ("6小时", "6h"), ("24小时", "24h"), ("7天", "7d"), ("全部", "all")],
                                value="all",
                                label="时间范围",
                                
                            )
                        with gr.Column(scale=1, min_width=120, elem_classes="export-btns"):
                            export_csv_btn = gr.Button("📄 CSV", size="sm", scale=1)
                            export_png_btn = gr.Button("🖼️ PNG", size="sm", scale=1)
                            export_msg = gr.Markdown("", elem_id="export-msg")

                    status_bar = gr.HTML(value=build_status_bar_html(), elem_id="status-bar")

                    # 主体双列内容
                    with gr.Row(elem_classes="main-content", equal_height=False):
                        # ── 左列：智能对话 ──
                        with gr.Column(elem_classes="chat-col"):
                            with gr.Group(elem_classes="content-card chat-card"):
                                gr.HTML('<div class="card-title">💬 智能对话助手</div>')
                                chatbot = gr.Chatbot(
                                    label="",
                                    height=420,
                                    elem_id="chatbot",
                                    show_label=False,
                                )
                                with gr.Row(elem_classes="chat-input-row"):
                                    msg = gr.Textbox(
                                        label="",
                                        placeholder="输入传感器参数或问题...",
                                        lines=2, scale=5,
                                        elem_id="chat-input",
                                    )
                                    send_btn = gr.Button("➤", variant="primary", scale=1)

                                gr.HTML('<div class="chat-section-title">快捷模板</div>')
                                with gr.Row(elem_classes="tpl-btns"):
                                    tpl_wk = gr.Button("温控分析", size="sm")
                                    tpl_yh = gr.Button("油耗分析", size="sm")
                                    tpl_zy = gr.Button("增压器诊断", size="sm")
                                    tpl_fz = gr.Button("负载参数", size="sm")

                                gr.HTML('<div class="chat-section-title">操作</div>')
                                with gr.Row(elem_classes="action-btns"):
                                    clear_btn = gr.Button("清空对话", size="sm")
                                    clear_data_btn = gr.Button("重置图表", size="sm")
                                with gr.Row(elem_classes="save-row"):
                                    save_label = gr.Textbox(
                                        placeholder="测试名称（可选）", scale=3,
                                        show_label=False,
                                    )
                                    save_btn = gr.Button("保存测试", variant="primary", size="sm", scale=1)
                                save_msg = gr.Markdown("")

                        # ── 右列：数据区 ──
                        with gr.Column(elem_classes="data-col"):
                            # 趋势图
                            with gr.Group(elem_classes="content-card chart-card"):
                                gr.HTML('<div class="card-title">📈 实时趋势图表</div>')
                                main_chart = gr.Plot(
                                    value=build_enhanced_chart(
                                        default_system, SYSTEM_TABS[default_system], session, 75, "all",
                                    ),
                                    elem_id="main-chart",
                                    show_label=False,
                                )

                            # 参数卡片
                            param_cards_html = gr.HTML(
                                value='<div class="param-cards-row"><div class="param-card ok"><div class="param-name">加载中...</div></div></div>',
                                elem_id="param-cards-html",
                            )

                            # 数据摘要
                            with gr.Group(elem_classes="content-card summary-card"):
                                gr.HTML('<div class="card-title">📊 数据摘要</div>')
                                data_summary = gr.Markdown(
                                    session.to_markdown(),
                                    elem_id="data-summary-md",
                                )


        # PAGE 4: 系统维修中心 Repair
        # ═══════════════════════════════
        with gr.Group(visible=False) as repair_page:
            # 维修页专属样式
            gr.HTML("""
            <style>
            .repair-page {
                background: #F5F7FA !important;
                min-height: calc(100vh - 64px);
                padding: 24px;
            }
            .repair-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 24px;
            }
            .repair-title {
                font-size: 22px;
                font-weight: 600;
                color: #1E293B;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .repair-stats {
                display: flex;
                gap: 20px;
            }
            .stat-item {
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: 700;
            }
            .stat-value.red { color: #EF4444; }
            .stat-value.green { color: #22C55E; }
            .stat-label {
                font-size: 12px;
                color: #94A3B8;
                margin-top: 2px;
            }
            .anomaly-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
            }
            .anomaly-card {
                background: #FFFFFF;
                border: 1px solid #EBEEF5;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 12px rgba(0,0,0,0.04);
                transition: all 0.3s ease;
                display: flex;
                flex-direction: column;
            }
            .anomaly-card.unresolved {
                border-left: 4px solid #EF4444;
            }
            .anomaly-card.resolved {
                border-left: 4px solid #22C55E;
                opacity: 0.85;
            }
            .anomaly-card-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 12px;
            }
            .anomaly-system {
                font-size: 15px;
                font-weight: 600;
                color: #1E293B;
            }
            .anomaly-status {
                font-size: 11px;
                padding: 3px 8px;
                border-radius: 10px;
                font-weight: 500;
            }
            .anomaly-status.unresolved {
                background: #FEF2F2;
                color: #EF4444;
            }
            .anomaly-status.resolved {
                background: #F0FDF4;
                color: #22C55E;
            }
            .anomaly-param {
                font-size: 13px;
                color: #475569;
                margin-bottom: 8px;
            }
            .anomaly-desc {
                font-size: 12px;
                color: #94A3B8;
                margin-bottom: 8px;
                flex: 1;
            }
            .anomaly-time {
                font-size: 11px;
                color: #CBD5E1;
                margin-bottom: 12px;
            }
            .resolve-btn {
                width: 100%;
                height: 32px;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }
            .resolve-btn.unresolved {
                background: #165DFF;
                color: #FFFFFF;
            }
            .resolve-btn.unresolved:hover {
                background: #0D47A1;
            }
            .resolve-btn.resolved {
                background: #F1F5F9;
                color: #94A3B8;
                cursor: default;
            }
            .empty-card {
                background: #FFFFFF;
                border: 1px dashed #CBD5E1;
                border-radius: 10px;
                padding: 40px 20px;
                text-align: center;
                color: #94A3B8;
                font-size: 13px;
            }
            .back-btn-row {
                margin-bottom: 16px;
            }
            </style>
            """)

            with gr.Column(elem_classes="repair-page"):
                with gr.Row(elem_classes="back-btn-row"):
                    back_to_overview_from_repair = gr.Button("← 返回总览", size="sm")

                # 顶部统计
                with gr.Row(elem_classes="repair-header"):
                    gr.HTML('<div class="repair-title">🔧 系统维修中心</div>')
                    repair_stats_html = gr.HTML(
                        '<div class="repair-stats">'
                        '<div class="stat-item"><div class="stat-value red" id="unresolved-count">0</div><div class="stat-label">待处理异常</div></div>'
                        '<div class="stat-item"><div class="stat-value green" id="resolved-count">0</div><div class="stat-label">已解决</div></div>'
                        '<div class="stat-item"><div class="stat-value" style="color:#165DFF;" id="total-health">100</div><div class="stat-label">当前健康度</div></div>'
                        '</div>'
                    )

                # 异常卡片网格（8张）
                anomaly_grid_html = gr.HTML('<div class="anomaly-grid" id="anomaly-grid"></div>')

        
        # ═══════════════════════════════
        # PAGE 3: 历史数据 History
        # ═══════════════════════════════
        with gr.Group(visible=False) as history_page:
            with gr.Row():
                back_btn = gr.Button("⬅ 返回总览", size="sm", elem_id="back-history-btn")
            gr.HTML('<div class="panel-title" style="margin-top:12px; color:#1E293B; font-size:15px; text-transform:none; letter-spacing:0.02em; border-bottom:1px solid rgba(0,200,100,0.15);">📂 全部历史会话</div>')
            history_type_filter = gr.Radio(
                choices=[("全部", "all"), ("文字输出", "text"), ("图片分析", "image")],
                value="all",
                label="分类筛选",
                interactive=True,
                elem_classes="history-filter",
            )
            with gr.Row(elem_id="history-btn-row"):
                hist_dropdown = gr.Dropdown(
                    choices=list_saved_sessions(), label="选择历史文件",
                    interactive=True, scale=4,
                )
                hist_load_btn = gr.Button("📂 加载并查看", scale=1, elem_id="hist-load-btn")
                hist_refresh_btn = gr.Button("🔄 刷新", scale=1, elem_id="hist-refresh-btn")
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
            param_cards = build_param_cards_html(sys_name, 75.0)
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
                param_cards,                                          # param_cards_html
            )

        def _nav_to_overview():
            """Navigate back to overview from detail/history/image/repair."""
            return (
                gr.update(visible=True),   # overview_page
                gr.update(visible=False),  # image_page
                gr.update(visible=False),  # monitor_page
                gr.update(visible=False),  # history_page
                gr.update(visible=False),  # repair_page
                "overview",                # page_state
            )
        
        # Helper: build repair stats HTML string
        def _build_repair_stats_html():
            unresolved = sum(1 for a in anomalies if a["status"] == "unresolved")
            resolved = sum(1 for a in anomalies if a["status"] == "resolved")
            health = get_current_health()
            return (
                '<div class="repair-stats">'
                f'<div class="stat-item"><div class="stat-value red">{unresolved}</div><div class="stat-label">待处理异常</div></div>'
                f'<div class="stat-item"><div class="stat-value green">{resolved}</div><div class="stat-label">已解决</div></div>'
                f'<div class="stat-item"><div class="stat-value" style="color:#165DFF;">{health}</div><div class="stat-label">当前健康度</div></div>'
                '</div>'
            )

        def _nav_to_repair():
            """Navigate from overview to repair center page."""
            return (
                gr.update(visible=False),       # overview_page
                gr.update(visible=False),       # image_page
                gr.update(visible=False),       # monitor_page
                gr.update(visible=False),       # history_page
                gr.update(visible=True),        # repair_page
                "repair",                       # page_state
                _build_repair_stats_html(),     # repair_stats_html
                build_anomaly_grid_html(),      # anomaly_grid_html
            )

        # ── JS Bridge Handlers ──
        def _on_repair_trigger(_x: str):
            """JS navigateToRepair() → open repair page."""
            return _nav_to_repair()

        def _on_resolve_trigger(aid_str: str):
            """JS resolveAnomaly(id) → mark resolved + refresh repair UI."""
            if not aid_str or not aid_str.isdigit():
                return [gr.update(), gr.update()]
            aid = int(aid_str)
            resolve_anomaly(aid)
            return (
                _build_repair_stats_html(),
                build_anomaly_grid_html(),
            )

        def _on_refresh_trigger(_x: str):
            """JS timer (10s) → refresh health trend chart + overview KPI."""
            return build_health_trend_chart(), build_overview_html()

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

        # ── JS Navigation Trigger: system cards → detail page ──
        SYS_ID_MAP = {
            "exhaust": "排气系统", "cooling": "冷却系统",
            "lube": "滑油系统", "scavenge": "扫气系统",
            "combustion": "燃烧参数", "turbo": "增压器", "fuel": "油耗",
        }
        def _on_sys_nav_trigger(sys_id: str):
            sys_name = SYS_ID_MAP.get(sys_id)
            if sys_name:
                return _nav_to_system(sys_name)
            # fallback — return no-ops for all 12 outputs
            return tuple([gr.update()] * 12)

        sys_nav_trigger.change(
            fn=_on_sys_nav_trigger,
            inputs=[sys_nav_trigger],
            outputs=[overview_page, image_page, monitor_page, history_page,
                     detail_title_html, main_chart, status_bar,
                     selected_system_state, data_summary, page_state, history_dropdown,
                     param_cards_html],
        )

        # ── JS Navigation Trigger: page tabs ──
        # Unified output: overlay/image/monitor/history visibility,
        # history components, page_state, image components (12 total)
        _none_input = None
        _empty_result = []
        _hidden_chart = gr.update(visible=False)
        _noop = gr.update()

        def _on_page_nav_trigger(page: str):
            if page == "overview":
                # _nav_to_overview → (ov, img, mon, hist, state) → 5 items
                # Pad to 12: indices 5-8=noop, 9=state, 10-12=image clears
                vals = list(_nav_to_overview())  # 5 items
                return (vals[0], vals[1], vals[2], vals[3],   # 1-4: pages
                        _noop, _noop, _noop, _noop,            # 5-8: history
                        vals[4],                                # 9: page_state
                        _none_input, _empty_result, _hidden_chart)  # 10-12: image
            elif page == "image":
                # _nav_to_image → (ov, img, mon, hist, state, input, result, chart) → 8 items
                vals = list(_nav_to_image())  # 8 items
                return (vals[0], vals[1], vals[2], vals[3],   # 1-4: pages
                        _noop, _noop, _noop, _noop,            # 5-8: history
                        vals[4],                                # 9: page_state
                        vals[5], vals[6], vals[7])              # 10-12: image
            elif page == "monitor":
                # Camera monitor page: show monitor, hide others
                return (gr.update(visible=False),   # overview_page
                        gr.update(visible=False),   # image_page
                        gr.update(visible=True),    # monitor_page
                        gr.update(visible=False),   # history_page
                        _noop, _noop, _noop, _noop, # 5-8: history
                        "monitor",                   # 9: page_state
                        _none_input, _empty_result, _hidden_chart)  # 10-12: image
            elif page == "history":
                # _nav_to_history → (ov, img, mon, hist, chart, himg, md, dd, state) → 9
                vals = list(_nav_to_history())  # 9 items
                return (vals[0], vals[1], vals[2], vals[3],   # 1-4: pages
                        vals[4], vals[5], vals[6], vals[7],   # 5-8: history
                        vals[8],                                # 9: page_state
                        _none_input, _empty_result, _hidden_chart)  # 10-12: image
            # fallback
            return tuple([_noop] * 12)

        page_nav_trigger.change(
            fn=_on_page_nav_trigger,
            inputs=[page_nav_trigger],
            outputs=[
                overview_page, image_page, monitor_page, history_page,
                history_chart, history_image, history_full_md, hist_dropdown,
                page_state,
                image_input, image_result, image_chart,
            ],
        )

        # ── Back button: detail → overview ──
        back_to_overview_btn.click(
            fn=_nav_to_overview,
            inputs=[],
            outputs=[overview_page, image_page, monitor_page, history_page, repair_page, page_state],
        )
        
        # 维修页返回总览
        back_to_overview_from_repair.click(
            fn=_nav_to_overview,
            inputs=[],
            outputs=[overview_page, image_page, monitor_page, history_page, repair_page, page_state],
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

        # ═══════════════════════════════
        # TOP NAV BUTTONS — 顶部导航按钮事件
        # ═══════════════════════════════
        nav_btn_overview.click(
            fn=_nav_to_overview,
            inputs=[],
            outputs=[overview_page, image_page, monitor_page, history_page, page_state],
        )

        nav_btn_monitor.click(
            fn=lambda: (
                gr.update(visible=False),   # overview
                gr.update(visible=False),   # image
                gr.update(visible=True),    # monitor
                gr.update(visible=False),   # history
                _noop, _noop, _noop, _noop, # history components
                "monitor",                   # page_state
                _none_input, _empty_result, _hidden_chart,  # image components
            ),
            inputs=[],
            outputs=[
                overview_page, image_page, monitor_page, history_page,
                history_chart, history_image, history_full_md, hist_dropdown,
                page_state,
                image_input, image_result, image_chart,
            ],
        )

        nav_btn_image.click(
            fn=_nav_to_image,
            inputs=[],
            outputs=[
                overview_page, image_page, monitor_page, history_page,
                page_state, image_input, image_result, image_chart,
            ],
        )

        nav_btn_history.click(
            fn=_nav_to_history,
            inputs=[],
            outputs=[
                overview_page, image_page, monitor_page, history_page,
                history_chart, history_image, history_full_md, hist_dropdown, page_state,
            ],
        )


        # ═══════════════════════════════
        # JS BRIDGE TRIGGERS — 与前端 JS 交互
        # ═══════════════════════════════
        repair_trigger.change(
            fn=_on_repair_trigger,
            inputs=[repair_trigger],
            outputs=[overview_page, image_page, monitor_page, history_page,
                     repair_page, page_state, repair_stats_html, anomaly_grid_html],
        )

        resolve_trigger.change(
            fn=_on_resolve_trigger,
            inputs=[resolve_trigger],
            outputs=[repair_stats_html, anomaly_grid_html],
        )

        refresh_trigger.change(
            fn=_on_refresh_trigger,
            inputs=[refresh_trigger],
            outputs=[health_trend_plot, overview_html],
        )

        # ═══════════════════════════════════════════
        # EVENT HANDLERS (existing, unchanged)
        # ═══════════════════════════════════════════

        # ── System Select (via radio) ──
        def on_system_select(sys_name, hl_load, tr):
            params = SYSTEM_TABS.get(sys_name, [])
            fig = build_enhanced_chart(sys_name, params, session, hl_load, tr)
            param_cards = build_param_cards_html(sys_name, hl_load)
            return (
                fig,
                build_status_bar_html(),
                sys_name,
                param_cards,
            )

        system_radio.change(
            fn=on_system_select,
            inputs=[system_radio, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar, selected_system_state, param_cards_html],
        )

        # ── Load Slider / Time Range ──
        def on_control_change(sys_name, hl_load, tr):
            params = SYSTEM_TABS.get(sys_name, [])
            fig = build_enhanced_chart(sys_name, params, session, hl_load, tr)
            param_cards = build_param_cards_html(sys_name, hl_load)
            return fig, build_status_bar_html(), param_cards

        highlight_slider.change(
            on_control_change,
            inputs=[selected_system_state, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar, param_cards_html],
        )

        time_range_radio.change(
            on_control_change,
            inputs=[selected_system_state, highlight_slider, time_range_radio],
            outputs=[main_chart, status_bar, param_cards_html],
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
    print(f"   Chat model routing: DS V4 Pro / DS V3 / Local qwen")
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
        server_name="0.0.0.0", server_port=7861, share=False, inbrowser=False,
        # 不设 root_path，统一由反代在 /demo/ 路径下提供访问，剥前缀转发到 7861/
        show_error=True,
        css=DASHBOARD_CSS,
        js=NAV_BRIDGE_JS,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    )
