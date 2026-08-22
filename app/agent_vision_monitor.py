#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮机智脑 · 智能体视觉监控
每 5 分钟用 playwright 打开 marine-sensor-panel.html
截图（视觉证据）+ 读 DOM 获取所有传感器参数值
用 KB_BASELINE 对比当前负载下的基准值
如果有异常 → 调 LLM 关联推理 → 诊断结论
通过 HTTP 7864 端口推送结果到 monitor-portal.html

启动：python agent_vision_monitor.py
停止：Ctrl+C
"""

import asyncio
import json
import time
import logging
import os
import re
import threading
import base64
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from openai import OpenAI

# ===== 配置 =====
# 所有运行时产物都放在脚本同级的 data/ 下，保证可移植（拷到任何机器都能跑）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PANEL_URL = "http://127.0.0.1:7862/static/cooling-system.html"  # 截用户实际看到的页面（http 协议下 fetch 7864 才不被 CORS 拦）
SCREENSHOT_DIR = os.path.join(DATA_DIR, "screenshots")
RESULT_FILE = os.path.join(DATA_DIR, "latest_diagnosis.json")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
HISTORY_INDEX = os.path.join(DATA_DIR, "history", "index.json")
HISTORY_MAX = 200  # 最多保留 200 条历史
INTERVAL_SEC = 300  # 5 分钟
MAX_SCREENSHOTS = 50  # 截图目录最多保留最近 50 张，旧的自动删除
HTTP_PORT = 7864

# 确保运行时目录存在（可移植，不依赖绝对路径）
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# LLM 配置（学校 API · 文本推理备用）
# 密钥通过环境变量注入（HF Spaces 设为 Space Secret），本地保留默认值兜底
SCHOOL_API_BASE = os.environ.get("SCHOOL_API_BASE", "https://chat.cqjtu.edu.cn/ds/api/v1")
SCHOOL_API_KEY_LLM = os.environ.get("SCHOOL_API_KEY_LLM", "sk-562cfe915f0b772b9cf663103eb962e0")
SCHOOL_API_KEY_DSR1 = os.environ.get("SCHOOL_API_KEY_DSR1", "sk-ba4c1b12745ac838e88520c4ddee80a0")
LLM_CLIENT = OpenAI(
    base_url=SCHOOL_API_BASE,
    api_key=SCHOOL_API_KEY_LLM
)
LLM_MODEL = "deepseek-v3-2-251201"

# DSR1 配置（doubao-2.0-pro 多模态视觉模型 · 学校中转）
DSR1_CLIENT = OpenAI(
    base_url=SCHOOL_API_BASE,
    api_key=SCHOOL_API_KEY_DSR1
)
DSR1_MODEL = "doubao-2.0-pro"

# RAG 知识库检索（复用主程序 kb_loader）
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kb_loader import get_retriever
    _kb_retriever = None
    def get_kb_retriever():
        global _kb_retriever
        if _kb_retriever is None:
            _kb_retriever = get_retriever()
        return _kb_retriever
except Exception as _e:
    print(f"[KB] kb_loader 加载失败: {_e}")
    def get_kb_retriever():
        return None

# RAG 检索目录（只用 MAN 12K98ME-C7 专属数据，不混入通用船舶知识库）
RAG_KB_NAMES = ["温度监测", "负载指数", "涡轮增压器", "油耗监测"]

# ===== 视觉读数 Prompt（step1：DSR1 看截图读参数） =====
VISION_READ_PROMPT = """你是"轮机智脑"的视觉识别模块，读取 MAN B&W 12K98ME-C7 船舶主机传感器监控面板截图中的参数值。

面板深色背景、卡片式布局，每张卡片含：参数中文名、大号数值（等宽字体）、单位、报警限值、状态。**顶部有一个大号负载数字（实时显示的连续值，带 1 位小数，如 87.3%），是当前主机的实际负载**。页面可能没有传统的 6 档按钮。

观察截图，识别：1)顶部显示的负载数字（连续值，保留小数） 2)当前系统（如有显示） 3)所有可见传感器卡片的参数名、数值、单位、状态

严格输出 JSON（不要其他文字）：
{"load": 87.3, "system": "冷却系统", "sensors": [{"name": "淡水进水温度", "value": 31.98, "unit": "℃", "status": "正常"}]}

规则：
- load = 页面顶部大号显示的负载数字（连续值，保留 1 位小数），如 87.3；不要四舍五入到整数
- name 用标准名：排气温度、涡轮前排气温度、涡轮后排气温度、淡水进水温度、缸套水出水温度、冷却淡水出水温度、活塞冷却油出口温度、涡轮滑油进口温度、涡轮滑油出口温度、扫气温度、扫气接收温度、扫气压力、最大爆发压力、压缩压力、增压器转速、增压器空气出口温度、燃油消耗率(实测)、燃油消耗率(修正)
- 面板名映射：气缸排气温度→排气温度，排气阀淡水出口温度→缸套水出水温度，增压器滑油进口温度→涡轮滑油进口温度，增压器滑油出口温度→涡轮滑油出口温度，空冷器进气温度→扫气温度，空冷器出气温度→扫气接收温度，鼓风机入口温度→增压器空气出口温度，燃油消耗率→燃油消耗率(实测)，修正油耗→燃油消耗率(修正)
- value 纯数值；看不清设 null
- status：正常/高报/低报/高高/低低/未知
- 只输出可见传感器，不编造
"""

# ===== 工况评估 Prompt（step2：基于读数做文本评估，不传图，轻量快速） =====
ASSESS_PROMPT = """你是资深轮机长，对 MAN B&W 12K98ME-C7 船舶主机传感器读数做工况评估。

你会收到当前负载和传感器读数（JSON）。像轮机长扫仪表盘那样，评估整体工况、发现异常和参数关联。

输出 JSON（不要其他文字）：
{"overall_status": "正常", "assessment": "评估描述1-2句", "concerns": ["关注点1"]}

规则：
- overall_status：正常/关注/异常
- assessment：整体工况判断（轮机长口吻，1-2句）
- concerns：主动发现的异常或关联问题数组；无异常则空数组 []
- 关注参数间关联（如：进水温度高但出水正常→问题在入口侧）
"""

# ===== KB_BASELINE（18 参数，严格遵循 7861 知识库） =====
LOADS = [25, 50, 75, 90, 100, 110]
KB_BASELINE = {
    "排气温度": {"unit":"℃","system":"排气","tolerance":15,"values":{25:224.8,50:262.7,75:270.5,90:288.8,100:310.1,110:345.9}},
    "涡轮前排气温度": {"unit":"℃","system":"排气","tolerance":25,"values":{25:380.5,50:351.8,75:364.0,90:389.8,100:427.5,110:462.0}},
    "涡轮后排气温度": {"unit":"℃","system":"排气","tolerance":15,"values":{25:251.8,50:264.0,75:228.5,90:227.0,100:239.3,110:258.0}},
    "淡水进水温度": {"unit":"℃","system":"冷却","tolerance":5,"values":{25:17,50:19,75:22,90:26,100:32,110:34}},
    "缸套水出水温度": {"unit":"℃","system":"冷却","tolerance":3,"values":{25:86.3,50:88.4,75:89.9,90:92.8,100:95.2,110:99.1}},
    "冷却淡水出水温度": {"unit":"℃","system":"冷却","tolerance":2,"values":{25:83.7,50:79.6,75:78.8,90:78.4,100:78.8,110:78.5}},
    "活塞冷却油出口温度": {"unit":"℃","system":"冷却","tolerance":2,"values":{25:48.3,50:54.1,75:57.6,90:58.8,100:59.1,110:59.8}},
    "涡轮滑油进口温度": {"unit":"℃","system":"滑油","tolerance":2,"values":{25:40,50:42,75:42,90:42,100:42,110:42}},
    "涡轮滑油出口温度": {"unit":"℃","system":"滑油","tolerance":5,"values":{25:46,50:52,75:58,90:62,100:66,110:68}},
    "扫气温度": {"unit":"℃","system":"扫气","tolerance":10,"values":{25:45,50:97,75:145,90:175,100:192,110:210}},
    "扫气接收温度": {"unit":"℃","system":"扫气","tolerance":5,"values":{25:18,50:20,75:27,90:35,100:40,110:44}},
    "扫气压力": {"unit":"bar","system":"扫气","tolerance":0.2,"values":{25:0.31,50:0.95,75:1.85,90:2.47,100:2.86,110:3.12}},
    "最大爆发压力": {"unit":"bar","system":"燃烧","tolerance":2,"values":{25:74.2,50:112.5,75:143.6,90:150.4,100:150.0,110:150.0}},
    "压缩压力": {"unit":"bar","system":"燃烧","tolerance":5,"values":{25:49.7,50:75.7,75:107.8,90:120.3,100:132.1,110:141.7}},
    "增压器转速": {"unit":"rpm","system":"增压器","tolerance":500,"values":{25:3778,50:6585,75:8433,90:9383,100:9945,110:10457}},
    "增压器空气出口温度": {"unit":"℃","system":"增压器","tolerance":5,"values":{25:25.25,50:26.63,75:27.75,90:28.63,100:28.63,110:27.75}},
    "燃油消耗率(实测)": {"unit":"g/kWh","system":"油耗","tolerance":5,"values":{25:184.42,50:173.93,75:172.43,90:175.57,100:179.58,110:183.27}},
    "燃油消耗率(修正)": {"unit":"g/kWh","system":"油耗","tolerance":3,"values":{25:181.11,50:170.25,75:168.64,90:171.58,100:175.39,110:179.00}},
}

# marine-sensor-panel.html 传感器名 → KB_BASELINE 参数名 映射
# 面板里的名称可能和 KB 不完全一致，需要映射
NAME_MAP = {
    "气缸排气温度": "排气温度",
    "涡轮前排气温度": "涡轮前排气温度",
    "涡轮后排气温度": "涡轮后排气温度",
    "排气接收器压力": None,  # KB 里没有
    "淡水进水温度": "淡水进水温度",
    "排气阀淡水出口温度": "缸套水出水温度",
    "冷却淡水出口温度": "冷却淡水出水温度",
    "活塞冷却油出口温度": "活塞冷却油出口温度",
    "空冷器淡水进水温度": "淡水进水温度",  # 近似映射
    "空冷器淡水出水温度": None,  # KB 里没有
    "增压器滑油进口温度": "涡轮滑油进口温度",
    "增压器滑油出口温度": "涡轮滑油出口温度",
    "滑油入口压力": None,  # KB 里没有
    "滑油进口温度": "涡轮滑油进口温度",
    "滑油出口温度": "涡轮滑油出口温度",
    "扫气压力": "扫气压力",
    "空冷器压降": None,
    "空冷器进气温度": "扫气温度",
    "空冷器出气温度": "扫气接收温度",
    "鼓风机入口温度": "增压器空气出口温度",
    "最大爆发压力": "最大爆发压力",
    "压缩压力": "压缩压力",
    "增压器转速": "增压器转速",
    "增压器空气出口温度": "增压器空气出口温度",
    "燃油消耗率": "燃油消耗率(实测)",
    "修正油耗": "燃油消耗率(修正)",
}

# ===== 日志 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agent_vision")

# ===== 全局最新诊断结果（供 HTTP 服务读取） =====
# 初始化为"已就绪·等待首次扫描"状态：让前端首屏不会显示一片空白横线，
# 而是显示"已启动，5 分钟后第一次扫描开始"
LATEST_RESULT = {
    "id": None,
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "load": None,
    "screenshot": None,
    "screenshot_base64": None,
    "sensors": {},
    "assessment": {
        "system": "冷却系统",
        "overall_status": "等待",
        "assessment": "智能体服务已就绪，等待首次视觉扫描（约 5 分钟内完成）...",
        "concerns": []
    },
    "anomalies": [],
    "diagnosis": None,
    "vision_method": None,
    "kb_sources": [],
    "status": "waiting",
    "next_check": None,
}


# ===== 历史归档 =====
def archive_history(record_id, result):
    """把诊断结果归档到 history/ 目录（截图已存 screenshots/，这里存诊断json + 更新索引）"""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    record = {k: v for k, v in result.items() if k != "screenshot_base64"}
    record_file = os.path.join(HISTORY_DIR, f"{record_id}.json")
    try:
        with open(record_file, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"归档失败: {e}")
        return

    index = []
    try:
        if os.path.exists(HISTORY_INDEX):
            with open(HISTORY_INDEX, "r", encoding="utf-8") as f:
                index = json.load(f)
    except Exception:
        index = []

    index.insert(0, {
        "id": record_id,
        "time": record.get("time"),
        "system": record.get("assessment", {}).get("system", "未知"),
        "load": record.get("load"),
        "status": record.get("status"),
        "overall_status": record.get("assessment", {}).get("overall_status", "未知"),
        "anomaly_count": len(record.get("anomalies", [])),
        "concern_count": len(record.get("assessment", {}).get("concerns", [])),
        "screenshot": record.get("screenshot"),
    })

    if len(index) > HISTORY_MAX:
        old = index[HISTORY_MAX:]
        index = index[:HISTORY_MAX]
        for o in old:
            try:
                os.remove(os.path.join(HISTORY_DIR, f"{o['id']}.json"))
            except Exception:
                pass

    try:
        with open(HISTORY_INDEX, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"索引更新失败: {e}")


# ===== 1. 视觉读取：playwright 截图 + DSR1 视觉识别 =====
def vision_assess(screenshot_path):
    """DSR1 视觉分析两步法（B深度）：step1看图读数 + step2文本评估工况，两个轻请求避免504"""
    with open(screenshot_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # step1: 看图读数（轻量，传图，已验证稳定）
    try:
        resp = DSR1_CLIENT.chat.completions.create(
            model=DSR1_MODEL,
            temperature=0.1,
            max_tokens=700,
            timeout=25,  # 单次调用最长 25 秒，避免学校 API 卡死整个 capture_panel
            messages=[
                {"role": "system", "content": VISION_READ_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "读取这张传感器面板截图的负载档位和所有参数值。"},
                ]},
            ],
        )
        result = resp.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', result)
        if not json_match:
            log.warning("DSR1 读数失败：无JSON")
            return None, None, None, None
        data = json.loads(json_match.group())
        load = int(data.get("load", 100))
        system = data.get("system", "未知")
        sensors_raw = data.get("sensors", [])
        sensors = {}
        for s in sensors_raw:
            name = s.get("name", "")
            if not name:
                continue
            kb_name = NAME_MAP.get(name, name)
            if not kb_name or kb_name not in KB_BASELINE:
                continue
            bl = KB_BASELINE.get(kb_name, {})
            sensors[kb_name] = {
                "value": s.get("value"),
                "unit": s.get("unit") or bl.get("unit", ""),
                "system": bl.get("system", system),
                "status": s.get("status", "未知"),
            }
        log.info(f"✓ step1 读数：{system} 负载{load}%，{len(sensors)}参数")
    except Exception as e:
        log.warning(f"DSR1 读数失败: {e}")
        return None, None, None, None

    # step2: 文本评估工况（轻量，不传图，基于读数，快）
    try:
        sensors_brief = [{"name": k, "value": v["value"], "unit": v["unit"], "status": v["status"]} for k, v in sensors.items()]
        assess_input = json.dumps({"load": load, "system": system, "sensors": sensors_brief}, ensure_ascii=False)
        resp2 = DSR1_CLIENT.chat.completions.create(
            model=DSR1_MODEL,
            temperature=0.3,
            max_tokens=400,
            timeout=25,
            messages=[
                {"role": "system", "content": ASSESS_PROMPT},
                {"role": "user", "content": assess_input},
            ],
        )
        result2 = resp2.choices[0].message.content.strip()
        json_match2 = re.search(r'\{[\s\S]*\}', result2)
        if json_match2:
            adata = json.loads(json_match2.group())
            assessment = {
                "system": system,
                "overall_status": adata.get("overall_status", "未知"),
                "assessment": adata.get("assessment", ""),
                "concerns": adata.get("concerns", []),
            }
        else:
            assessment = {"system": system, "overall_status": "未知", "assessment": result2[:100], "concerns": []}
        log.info(f"✓ step2 评估：{assessment['overall_status']}，{len(assessment['concerns'])}关注点")
        return load, sensors, assessment, "vision"
    except Exception as e:
        log.warning(f"DSR1 评估失败（读数已成功）: {e}")
        assessment = {"system": system, "overall_status": "未知", "assessment": "评估失败，读数已获取", "concerns": []}
        return load, sensors, assessment, "vision"


def prune_screenshots():
    """保留 SCREENSHOT_DIR 中最近 MAX_SCREENSHOTS 张截图，删除更早的，避免无限膨胀。"""
    try:
        if not os.path.isdir(SCREENSHOT_DIR):
            return
        files = [os.path.join(SCREENSHOT_DIR, f) for f in os.listdir(SCREENSHOT_DIR)
                 if f.lower().endswith('.png')]
        if len(files) <= MAX_SCREENSHOTS:
            return
        # 按修改时间排序，最早的排前面
        files.sort(key=lambda p: os.path.getmtime(p))
        for old in files[:-MAX_SCREENSHOTS]:
            try:
                os.remove(old)
            except OSError:
                pass
        log.info(f"截图目录已清理：保留最近 {MAX_SCREENSHOTS} 张，删除 {len(files) - MAX_SCREENSHOTS} 张旧截图")
    except Exception as e:
        log.warning(f"清理旧截图失败: {e}")


async def capture_panel():
    """用 playwright 打开面板，截图 → DSR1 视觉识别读参数（读DOM作fallback）"""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        await page.goto(PANEL_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)  # 等 JS 渲染

        # 切换到冷却系统 Tab（原型聚焦冷却系统）
        try:
            await page.locator(".sys-tab", has_text="冷却").click()
            await page.wait_for_timeout(800)  # 等切换渲染
        except Exception as e:
            log.warning(f"切换冷却 Tab 失败: {e}")

        # 截图（视觉证据 + 供 DSR1 识别）
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        screenshot_path = os.path.join(SCREENSHOT_DIR, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        await page.screenshot(path=screenshot_path, full_page=True)
        prune_screenshots()  # 拍完即清理，目录恒定保留最近 MAX_SCREENSHOTS 张

        # 读 DOM 作为 fallback / 交叉验证
        dom_load = await page.evaluate(
            "document.querySelector('.load-btn.active')?.dataset.load || '100'"
        )
        dom_sensors = await page.evaluate("""
            () => {
                const result = {};
                if (typeof SYSTEMS === 'undefined' || typeof sensorState === 'undefined') return result;
                Object.keys(SYSTEMS).forEach(sysKey => {
                    SYSTEMS[sysKey].sensors.forEach(s => {
                        const st = sensorState[s.id] || {};
                        result[s.name] = {
                            value: st.current != null ? st.current : null,
                            unit: s.unit,
                            system: SYSTEMS[sysKey].name,
                            status: st.status || 'unknown'
                        };
                    });
                });
                return result;
            }
        """)

        await browser.close()

        # 截图转 base64（供前端直接显示）
        with open(screenshot_path, "rb") as f:
            screenshot_b64 = base64.b64encode(f.read()).decode("utf-8")

    # DSR1 工况评估（B深度真视觉）
    load, sensors, assessment, method = vision_assess(screenshot_path)

    if sensors:
        # 只存 basename，保证可移植（HF/其他机器用 SCREENSHOT_DIR 拼接）
        return int(load), sensors, assessment, os.path.basename(screenshot_path), screenshot_b64, "vision"

    # 降级读 DOM
    log.warning("⚠ DSR1 工况评估失败，降级读 DOM")
    dom_load_int = int(dom_load) if str(dom_load).isdigit() else 100
    sensors_fallback = {}
    for name, data in dom_sensors.items():
        kb_name = NAME_MAP.get(name, name)
        if kb_name and kb_name in KB_BASELINE:
            sensors_fallback[kb_name] = data
    assessment_fallback = {"system": "冷却系统", "overall_status": "未知", "assessment": "视觉评估失败，已降级读DOM", "concerns": []}
    return dom_load_int, sensors_fallback, assessment_fallback, os.path.basename(screenshot_path), screenshot_b64, "dom"


# ===== 2. KB 匹配：用 KB_BASELINE 对比 =====
def check_against_kb(load, sensors):
    """对比所有参数与 KB 基准，返回异常列表"""
    anomalies = []
    for panel_name, data in sensors.items():
        if data["value"] is None:
            continue
        # 映射到 KB 参数名
        kb_name = NAME_MAP.get(panel_name)
        if not kb_name or kb_name not in KB_BASELINE:
            continue

        bl = KB_BASELINE[kb_name]
        baseline_val = bl["values"].get(load)
        if baseline_val is None:
            continue

        try:
            val = float(data["value"])
        except (ValueError, TypeError):
            continue

        delta = val - baseline_val
        if abs(delta) > bl["tolerance"]:
            anomalies.append({
                "param": kb_name,
                "panel_name": panel_name,
                "value": val,
                "baseline": baseline_val,
                "delta": round(delta, 2),
                "tolerance": bl["tolerance"],
                "unit": bl["unit"],
                "system": bl["system"],
                "deviation_pct": round(abs(delta / baseline_val * 100), 1) if baseline_val else 0,
            })
    return anomalies


# ===== 2.5 RAG 检索：异常时检索 kb_data 故障案例 =====
def retrieve_kb_evidence(anomalies, load):
    """检测到异常时，检索知识库故障案例/维修知识作为诊断依据"""
    retriever = get_kb_retriever()
    if not retriever or not anomalies:
        return "", []

    # 用异常参数名 + 系统名 + 故障关键词构造查询
    param_names = [a["param"] for a in anomalies]
    systems = list(set(a["system"] for a in anomalies))
    query = " ".join(param_names + systems + ["故障 原因 处理 温度 异常"])

    try:
        results = retriever.search(query, kb_names=RAG_KB_NAMES, top_k=12)
        if not results:
            return "", []

        lines = []
        sources = []
        seen = set()
        for seg in results:
            key = seg.content[:60]
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"### （来源：{seg.kb_name}）\n{seg.content[:1500]}\n")
            sources.append(seg.kb_name)

        kb_context = "\n".join(lines)
        log.info(f"[RAG] 检索命中 {len(results)} 条知识库片段（来源：{set(sources)}）")
        return kb_context, sources
    except Exception as e:
        log.warning(f"[RAG] 知识库检索失败: {e}")
        return "", []


# ===== 3. LLM 关联推理 =====
def llm_diagnose(load, anomalies, all_sensors, kb_context="", kb_sources=None):
    """调 DSR1 做多参数关联推理（带 RAG 知识库依据，三段式输出）"""
    if not anomalies:
        return None

    anomaly_text = "\n".join([
        f"- {a['param']}({a['system']}): 当前{a['value']}{a['unit']}，基准{a['baseline']}{a['unit']}，"
        f"偏差{a['delta']:+.1f}{a['unit']}（容差±{a['tolerance']}{a['unit']}）"
        for a in anomalies
    ])

    # 提供全部参数供 LLM 排除
    normal_text = "\n".join([
        f"- {name}: {data['value']}{data['unit']}（{data['system']}）"
        for name, data in all_sensors.items()
        if data["value"] is not None
    ])

    kb_section = f"【知识库故障案例参考】\n{kb_context}\n" if kb_context else "【知识库故障案例参考】\n（本次未检索到相关知识库内容）\n"
    sources_json = json.dumps(kb_sources or [], ensure_ascii=False)

    prompt = f"""你是船舶轮机智能诊断专家"轮机智脑"，负责 MAN B&W 12K98ME-C7 主机的故障诊断。你通过视觉识别读取了传感器面板的参数值，并与知识库基准对比发现了异常。

当前主机负载：{load}%

【检测到的异常参数】
{anomaly_text}

【全部传感器读数（含正常参数，用于排除推理）】
{normal_text}

{kb_section}

请作为专业轮机长进行关联推理诊断，体现智能体的专业判断能力：
1. 分析异常参数之间是否存在因果或关联关系
2. 用正常参数排除不可能的故障原因
3. 结合知识库故障案例，给出最可能的诊断结论
4. 明确引用知识库依据

严格按以下 JSON 格式输出（不要 markdown 代码块）：
{{
  "reasoning": "推理链：详细描述你怎么关联异常参数、怎么用正常参数排除、怎么结合知识库得出结论",
  "conclusions": [
    {{"diagnosis": "诊断结论1", "confidence": 75, "reasoning": "依据", "evidence": "引用的知识库内容"}},
    {{"diagnosis": "诊断结论2", "confidence": 25, "reasoning": "依据", "evidence": "引用的知识库内容"}}
  ],
  "evidence_sources": {sources_json},
  "recommendation": "处置建议（含预计工时/备件）"
}}"""

    try:
        resp = DSR1_CLIENT.chat.completions.create(
            model=DSR1_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
            timeout=25,
        )
        result = resp.choices[0].message.content.strip()
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            return json.loads(json_match.group())
        return {"raw": result, "reasoning": result, "conclusions": [], "evidence_sources": kb_sources or [], "recommendation": ""}
    except Exception as e:
        log.error(f"DSR1 诊断推理失败: {e}")
        return {"error": str(e), "reasoning": "DSR1 诊断推理失败", "conclusions": [], "evidence_sources": kb_sources or [], "recommendation": ""}


# ===== 4. HTTP 服务（供 monitor-portal.html 轮询） =====
class DiagnosisHandler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/latest":
            result = {k: v for k, v in LATEST_RESULT.items() if k != "screenshot_base64"}
            self._json(200, result)
        elif path == "/screenshot":
            b64 = LATEST_RESULT.get("screenshot_base64")
            if b64:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self._cors()
                self.end_headers()
                self.wfile.write(base64.b64decode(b64))
            else:
                self._json(404, {"error": "no screenshot"})
        elif path == "/history":
            index = []
            try:
                if os.path.exists(HISTORY_INDEX):
                    with open(HISTORY_INDEX, "r", encoding="utf-8") as f:
                        index = json.load(f)
            except Exception:
                pass
            self._json(200, {"count": len(index), "records": index[:50]})
        elif path.startswith("/history/"):
            rid = path.replace("/history/", "")
            record_file = os.path.join(HISTORY_DIR, f"{rid}.json")
            if os.path.exists(record_file):
                with open(record_file, "r", encoding="utf-8") as f:
                    self._json(200, json.load(f))
            else:
                self._json(404, {"error": "record not found"})
        elif path.startswith("/screenshot/"):
            rid = path.replace("/screenshot/", "")
            record_file = os.path.join(HISTORY_DIR, f"{rid}.json")
            shot_name = None
            if os.path.exists(record_file):
                with open(record_file, "r", encoding="utf-8") as f:
                    shot_name = json.load(f).get("screenshot")
            shot_path = os.path.join(SCREENSHOT_DIR, shot_name) if shot_name else None
            if shot_path and os.path.exists(shot_path):
                with open(shot_path, "rb") as f:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self._cors()
                    self.end_headers()
                    self.wfile.write(f.read())
            else:
                self._json(404, {"error": "screenshot not found"})
        elif path.startswith("/screenshots/"):
            # 按文件名直接取最新截图（前端 cooling-diagnosis.html 用此路由）
            fname = path.replace("/screenshots/", "").split("?")[0]
            shot_path = os.path.join(SCREENSHOT_DIR, fname)
            if os.path.exists(shot_path):
                with open(shot_path, "rb") as f:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self._cors()
                    self.end_headers()
                    self.wfile.write(f.read())
            else:
                self._json(404, {"error": "screenshot not found"})
        elif path == "/health":
            # 计算是否在诊断中：
            # 1) LATEST_RESULT.id 是空 → 从未成功跑过 capture_panel → "已暂停"
            # 2) LATEST_RESULT.id 有值 且距今 < 6 分钟 → 监控循环活跃 → "诊断中"
            # 3) LATEST_RESULT.id 有值 且距今 > 6 分钟 → 卡住了 → "已暂停"（异常）
            last_t = LATEST_RESULT.get("time")
            record_id = LATEST_RESULT.get("id")
            diagnosing = False
            if record_id and last_t:
                try:
                    last_dt = datetime.strptime(last_t, "%Y-%m-%d %H:%M:%S")
                    age_sec = (datetime.now() - last_dt).total_seconds()
                    diagnosing = 0 <= age_sec < (INTERVAL_SEC + 60)
                except Exception:
                    pass
            self._json(200, {
                "status": "ok",
                "port": HTTP_PORT,
                "interval_sec": INTERVAL_SEC,
                "diagnosing": diagnosing,
                "last_check": LATEST_RESULT.get("time"),
                "next_check": LATEST_RESULT.get("next_check"),
                "result_status": LATEST_RESULT.get("status"),
            })
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):
        pass


def start_http_server():
    server = HTTPServer(("127.0.0.1", HTTP_PORT), DiagnosisHandler)
    server.serve_forever()


# ===== 5. 推送到 alert_bridge（Server酱） =====
def push_to_alert_bridge(anomalies, diagnosis):
    """有异常时推送到 alert_bridge.py（7863）→ Server酱"""
    try:
        import urllib.request
        # 取最严重的异常（按偏差超出容差的倍数排序，与知识库绝对容差同单位）
        worst = max(anomalies, key=lambda a: abs(a["delta"]) / a["tolerance"] if a["tolerance"] else 0)
        level = "hh" if (worst["tolerance"] and abs(worst["delta"]) > worst["tolerance"] * 1.5) else "h"
        alert_data = json.dumps({
            "sensorName": worst["param"],
            "sensorSymbol": "",
            "level": level,
            "value": f"{worst['value']}{worst['unit']}(偏差{worst['delta']:+.1f})",
            "unit": "",
            "system": worst["system"],
            "time": datetime.now().strftime("%H:%M:%S"),
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:7863/alert",
            data=alert_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        log.info("已推送到 alert_bridge → Server酱")
    except Exception as e:
        log.warning(f"推送 alert_bridge 失败（服务可能未启动）: {e}")


# ===== 6. 主循环 =====
async def monitor_loop():
    global LATEST_RESULT
    log.info("=" * 60)
    log.info("轮机智脑 · 智能体视觉监控")
    log.info(f"面板: {PANEL_URL}")
    log.info(f"间隔: {INTERVAL_SEC}s（{INTERVAL_SEC // 60} 分钟）")
    log.info(f"LLM: {LLM_MODEL}")
    log.info(f"HTTP: http://127.0.0.1:{HTTP_PORT}/latest")
    log.info("=" * 60)

    while True:
        try:
            log.info("--- 开始视觉检测 ---")
            # 1. 截图 + DSR1 工况评估（读DOM作fallback）
            try:
                load, sensors, assessment, screenshot, screenshot_b64, method = await capture_panel()
                log.info(f"✓ {assessment.get('system','?')} 负载{load}%：{assessment.get('overall_status','?')}，{len(sensors)}参数，{len(assessment.get('concerns',[]))}关注点")
                log.info(f"✓ 评估：{assessment.get('assessment','')[:80]}")
                log.info(f"✓ 截图: {screenshot}")
            except Exception as cap_err:
                # 把 capture_panel 错误直接暴露到前端，避免用户干等不知道出了啥问题
                log.error(f"capture_panel 崩溃: {cap_err}")
                import traceback as _tb
                _tb.print_exc()
                load, sensors, assessment = None, {}, {"system": "冷却系统", "overall_status": "等待", "assessment": f"capture_panel 异常: {cap_err}", "concerns": []}
                screenshot, screenshot_b64, method = None, None, "error"

            # 2. KB 对比（客观校验视觉评估）
            anomalies = check_against_kb(load, sensors)
            if anomalies:
                log.warning(f"⚠ KB校验检测到 {len(anomalies)} 个异常参数:")
                for a in anomalies:
                    log.warning(f"  - {a['param']}: {a['value']}{a['unit']} (基准{a['baseline']}, 偏差{a['delta']:+.1f})")

            # 3. RAG 检索 + DSR1 诊断（仅有异常时）
            diagnosis = None
            kb_sources = []
            if anomalies:
                log.info("检索知识库故障案例...")
                kb_context, kb_sources = retrieve_kb_evidence(anomalies, load)
                log.info("调用 DSR1 关联推理...")
                diagnosis = llm_diagnose(load, anomalies, sensors, kb_context, kb_sources)
                if diagnosis and diagnosis.get("conclusions"):
                    for c in diagnosis["conclusions"]:
                        log.info(f"  📋 {c.get('diagnosis','?')} (置信度 {c.get('confidence',0)}%)")
                if diagnosis and diagnosis.get("recommendation"):
                    log.info(f"  ✅ 建议: {diagnosis['recommendation']}")

            # 4. 更新全局结果
            next_time = datetime.fromtimestamp(time.time() + INTERVAL_SEC).strftime("%H:%M:%S")
            record_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            LATEST_RESULT = {
                "id": record_id,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "load": load,
                "screenshot": screenshot,
                "screenshot_base64": screenshot_b64,
                "sensors": sensors,
                "assessment": assessment,
                "anomalies": anomalies,
                "diagnosis": diagnosis,
                "vision_method": method,
                "kb_sources": kb_sources,
                "status": "abnormal" if anomalies else ("concern" if assessment.get("concerns") else "normal"),
                "next_check": next_time,
                "interval_sec": INTERVAL_SEC,
            }

            # 5. 保存到文件 + 历史归档
            with open(RESULT_FILE, "w", encoding="utf-8") as f:
                save = {k: v for k, v in LATEST_RESULT.items() if k != "screenshot_base64"}
                json.dump(save, f, ensure_ascii=False, indent=2)
            archive_history(record_id, LATEST_RESULT)

            # 6. 推送（有异常时）
            if anomalies and diagnosis:
                push_to_alert_bridge(anomalies, diagnosis)

        except Exception as e:
            log.error(f"检测异常: {e}")
            import traceback
            traceback.print_exc()

        log.info(f"下次检测: {LATEST_RESULT.get('next_check', '?')}（{INTERVAL_SEC}s 后）")
        await asyncio.sleep(INTERVAL_SEC)


def main():
    # 启动 HTTP 服务（独立线程）
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    log.info(f"HTTP 服务已启动: http://127.0.0.1:{HTTP_PORT}")

    # 启动监控主循环
    asyncio.run(monitor_loop())


if __name__ == "__main__":
    main()
