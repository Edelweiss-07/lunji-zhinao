"""
Knowledge base loader and retriever.
Loads all extracted Coze KB segments and provides search.

检索算法（v2）—— 从原始的 bigram Jaccard 相似度升级为多信号加权：

  1. BM25（字符 bigram 作为词元 + 全局 IDF + 文档长度归一化）
     词频与长文档不再天然占优，长段落不会被高估。
  2. 4-gram 短语命中（乘性提升）
     精确捕获「涡轮增压器转速」「冷却水出口温度」这类 5-7 字技术短语。
  3. unigram 覆盖度（加性补强）
     短查询 / 生僻参数名只有 1-2 个字时，bigram 失效，unigram 兜底。
  4. 数值负载命中（加性补强）
     轮机查询大量围绕负载档位（75%、110%）与转速阈值（13350rpm），
     数值在段落中的精确出现是极强的区分信号。

所有 n-gram 在加载时预计算并缓存在 Segment 上，检索过程零重复切分。
不引入任何第三方依赖，接口（search / retrieve_for_intent / get_retriever）
保持不变，visualizer_core / visualizer_main / api_server 无需改动。
"""
import json
import re
import math
from collections import Counter
from pathlib import Path
from typing import Optional

KB_DIR = Path(__file__).parent / "kb_data"

# Intent → KB directory mapping (matching workflow "垂杨" routing)
INTENT_KB_MAP = {
    "class01_船舶分类": "船舶分类",
    "class05_应急响应": "应急知识库",
    "class06_培训需要": "培养知识库",
    "class07_维护保养": "维护知识库",
    "class08_故障维修": "查询知识库",
    "class09_温度监控": "温度监测",
    "class10_油耗监控": "油耗监测",
    "class11_增压器监测": "涡轮增压器",
    "class12_负载参数": "负载指数",
}

# Additional KBs used by specific intents
EXTRA_KB_MAP = {
    "class05_应急响应": ["应急知识库", "培养知识库"],
    "class06_培训需要": ["培养知识库", "应急知识库"],
}

# ── BM25 参数 ─────────────────────────────────────────────────────
_K1 = 1.5            # 词频饱和因子
_B = 0.75            # 文档长度归一化强度
# ── 信号权重 ──────────────────────────────────────────────────────
_W_QUAD = 2.0        # 4-gram 短语命中：乘性提升系数 (1 + w*qj)
_W_UNI = 1.0         # unigram 覆盖：加性分
_W_NUM = 1.5         # 数值命中：每个数值加此分
_DUP_THRESHOLD = 0.85  # 4-gram Jaccard ≥ 该值视为与已选段落重复，跳过


class Segment:
    """知识库分段，加载时预计算全部检索特征。"""
    __slots__ = ("content", "doc_id", "kb_name", "position",
                 "length", "bigrams", "quads", "unigrams")

    def __init__(self, content: str, doc_id: str, kb_name: str, position: int):
        self.content = content
        self.doc_id = doc_id
        self.kb_name = kb_name
        self.position = position
        self.length = 0
        self.bigrams: Counter = Counter()
        self.quads: set = set()
        self.unigrams: set = set()
        self._index(content)

    def _index(self, content: str):
        text = re.sub(r'\s+', '', content)
        self.length = len(text)
        if self.length == 0:
            return
        self.unigrams = set(text)
        if self.length >= 2:
            self.bigrams = Counter(text[i:i + 2] for i in range(self.length - 1))
        if self.length >= 4:
            self.quads = {text[i:i + 4] for i in range(self.length - 3)}

    def quad_jaccard(self, q_quads: set) -> float:
        if not self.quads or not q_quads:
            return 0.0
        inter = len(self.quads & q_quads)
        union = len(self.quads | q_quads)
        return inter / union if union else 0.0

    def unigram_jaccard(self, q_uni: set) -> float:
        if not self.unigrams or not q_uni:
            return 0.0
        inter = len(self.unigrams & q_uni)
        union = len(self.unigrams | q_uni)
        return inter / union if union else 0.0


class KnowledgeRetriever:
    def __init__(self):
        self.segments: list[Segment] = []
        self._avg_len = 0.0
        self._idf: dict[str, float] = {}
        self._load_all()

    def _load_all(self):
        """Load all segments from all KB directories, then compute IDF / avg length."""
        if not KB_DIR.exists():
            print(f"[KB] WARNING: {KB_DIR} not found!")
            return

        for kb_path in sorted(KB_DIR.iterdir()):
            if not kb_path.is_dir():
                continue
            seg_file = kb_path / "segments.jsonl"
            if not seg_file.exists():
                continue

            kb_name = kb_path.name
            count = 0
            for line in seg_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = data.get("content", "")
                if not content:
                    continue

                self.segments.append(Segment(
                    content=content,
                    doc_id=data.get("document_id", ""),
                    kb_name=kb_name,
                    position=data.get("position", 0),
                ))
                count += 1
            print(f"[KB] Loaded {kb_name}: {count} segments")

        # ── 全局 IDF 与平均长度 ──
        n = len(self.segments)
        if n == 0:
            print("[KB] No segments loaded!")
            return
        df: Counter = Counter()
        total_len = 0
        for seg in self.segments:
            total_len += seg.length
            df.update(seg.bigrams.keys())
        self._avg_len = total_len / n
        for term, doc_freq in df.items():
            self._idf[term] = math.log(1.0 + (n - doc_freq + 0.5) / (doc_freq + 0.5))

        print(f"[KB] Total: {n} segments across all KBs (avg_len={self._avg_len:.0f})")

    # ── 查询解析 ──────────────────────────────────────────────────
    @staticmethod
    def _query_features(query: str):
        text = re.sub(r'\s+', '', query)
        n = len(text)
        bigrams = Counter(text[i:i + 2] for i in range(max(0, n - 1)))
        quads = {text[i:i + 4] for i in range(max(0, n - 3))}
        unigrams = set(text)
        return bigrams, quads, unigrams

    @staticmethod
    def _build_number_regex(query: str):
        """提取查询中的数值（75%、13350rpm → 75、13350），构造边界匹配正则。

        数值以非数字为边界整体匹配，避免 25 误命中 250。长数值优先，
        保证 13350 先于 13 被匹配。
        """
        nums = [m.group(0) for m in re.finditer(r'\d+(?:\.\d+)?', query)]
        if not nums:
            return None
        ordered = sorted(set(nums), key=len, reverse=True)
        pattern = r'(?<!\d)(?:' + '|'.join(re.escape(x) for x in ordered) + r')(?!\d)'
        return re.compile(pattern)

    # ── 打分 ──────────────────────────────────────────────────────
    def _bm25(self, seg: Segment, q_bigrams: Counter) -> float:
        if not q_bigrams or seg.length == 0:
            return 0.0
        norm = _K1 * (1.0 - _B + _B * seg.length / self._avg_len)
        score = 0.0
        for term, qtf in q_bigrams.items():
            idf = self._idf.get(term, 0.0)
            if idf <= 0:
                continue
            tf = seg.bigrams.get(term, 0)
            if tf == 0:
                continue
            score += qtf * idf * (tf * (_K1 + 1.0)) / (tf + norm)
        return score

    def _count_numeric_hits(self, seg: Segment, num_regex) -> int:
        if num_regex is None:
            return 0
        return len(num_regex.findall(seg.content))

    # ── 检索 ──────────────────────────────────────────────────────
    def search(self, query: str, kb_names: Optional[list[str]] = None, top_k: int = 5) -> list[Segment]:
        """
        Search for relevant segments.
        多信号融合打分：BM25(bigram) * 短语提升 + unigram 覆盖 + 数值命中。
        Filtered by kb_names if provided.
        """
        candidates = self.segments
        if kb_names:
            candidates = [s for s in self.segments if s.kb_name in kb_names]
        if not candidates:
            return []

        q_bigrams, q_quads, q_unigrams = self._query_features(query)
        num_regex = self._build_number_regex(query)

        scored = []
        for seg in candidates:
            if seg.length == 0:
                continue
            bm25 = self._bm25(seg, q_bigrams)
            quad_jac = seg.quad_jaccard(q_quads)
            # 相关性门槛：无任何双字/短语命中即视为不相关。
            # 否则无关查询仅靠 1 个单字重合（如「今天」里的「天」）就会拖进整段垃圾。
            if bm25 <= 0 and quad_jac <= 0:
                continue
            uni_jac = seg.unigram_jaccard(q_unigrams)
            num_hits = self._count_numeric_hits(seg, num_regex)

            score = bm25 * (1.0 + _W_QUAD * quad_jac) + _W_UNI * uni_jac + _W_NUM * num_hits
            scored.append((score, seg))

        scored.sort(key=lambda x: x[0], reverse=True)

        # 去重：跳过与已入选段落 4-gram 过于相似的（同一文档相邻分块）
        results: list[Segment] = []
        chosen: list[Segment] = []
        for _, seg in scored:
            if self._is_duplicate(seg, chosen):
                continue
            results.append(seg)
            chosen.append(seg)
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _is_duplicate(seg: Segment, chosen: list[Segment]) -> bool:
        if not seg.quads:
            return False
        for c in chosen:
            if not c.quads:
                continue
            inter = len(seg.quads & c.quads)
            union = len(seg.quads | c.quads)
            if union and inter / union >= _DUP_THRESHOLD:
                return True
        return False

    def get_kb_names_for_intent(self, intent: str) -> list[str]:
        """Get KB directory names for a given intent ID."""
        names = []
        primary = INTENT_KB_MAP.get(intent, "")
        if primary:
            names.append(primary)
        extra = EXTRA_KB_MAP.get(intent, [])
        names.extend(extra)
        return names

    def retrieve_for_query(self, query: str, kb_names: Optional[list[str]] = None,
                           top_k: int = 5, max_chars: int = 4000,
                           with_source: bool = True) -> str:
        """Search, dedup and format results as prompt context with a char budget.

        max_chars 控制注入 prompt 的总字符预算——旧实现 top_k=12 × 2000 字最多
        塞 24k 字符上下文，严重稀释模型注意力；这里限制在几 KB 内，只保留最相关段落。
        """
        results = self.search(query, kb_names=kb_names, top_k=top_k)
        if not results:
            return ""

        lines = []
        used = 0
        for seg in results:
            content = seg.content[:1200]
            chunk = f"### （来源：{seg.kb_name}）\n{content}\n" if with_source else f"{content}\n"
            if used + len(chunk) > max_chars and lines:
                break
            lines.append(chunk)
            used += len(chunk)
            if used >= max_chars:
                break
        return "".join(lines)

    def retrieve_for_intent(self, query: str, intent: str, top_k: int = 5,
                            max_chars: int = 4000) -> str:
        """Search and format results for a specific intent."""
        kb_names = self.get_kb_names_for_intent(intent)
        if not kb_names:
            return ""
        return self.retrieve_for_query(query, kb_names=kb_names, top_k=top_k,
                                       max_chars=max_chars, with_source=False)


# Singleton
_retriever: Optional[KnowledgeRetriever] = None

def get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever
