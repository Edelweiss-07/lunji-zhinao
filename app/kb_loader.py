"""
Knowledge base loader and retriever.
Loads all extracted Coze KB segments and provides search.

检索优化（v2）：
- 加载时预计算每段字符 2-gram 集合，查询时不再重复构建
- 查询与正文统一去除标点/空白后再切 gram，避免标点干扰匹配
- IDF 加权的查询覆盖率打分：score = Σ idf(命中gram) / Σ idf(查询gram)
  （旧 Jaccard 的分母包含段落全部 gram，导致长段落被系统性压低排名）
- 同分按文档位置靠前优先；结果按内容前缀去重
"""
import json
import math
import re
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

_STRIP_RE = re.compile(r"[^\w]+", re.UNICODE)


def _char_ngrams(text: str, n: int = 2) -> set:
    """字符 n-gram（去标点空白后切分，中英文/数字通用）。"""
    text = _STRIP_RE.sub("", text)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


class Segment:
    __slots__ = ("content", "doc_id", "kb_name", "position", "ngrams")

    def __init__(self, content: str, doc_id: str, kb_name: str, position: int):
        self.content = content
        self.doc_id = doc_id
        self.kb_name = kb_name
        self.position = position
        self.ngrams = _char_ngrams(content)


class KnowledgeRetriever:
    def __init__(self):
        self.segments: list[Segment] = []
        self._idf: dict[str, float] = {}
        self._idf_default: float = 1.0
        self._load_all()
        self._build_idf()

    def _load_all(self):
        """Load all segments from all KB directories."""
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

        print(f"[KB] Total: {len(self.segments)} segments across all KBs")

    def _build_idf(self):
        """按 gram 在全部段落中的出现次数计算平滑 IDF：越普遍的 gram 权重越低。"""
        total = len(self.segments)
        if not total:
            return
        df: Counter = Counter()
        for seg in self.segments:
            df.update(seg.ngrams)
        self._idf = {g: 1.0 + math.log((total + 1) / (d + 1)) for g, d in df.items()}
        self._idf_default = 1.0 + math.log(total + 1)

    def search(self, query: str, kb_names: Optional[list[str]] = None, top_k: int = 5) -> list[Segment]:
        """
        Search for relevant segments.
        IDF 加权查询覆盖率打分：段落覆盖的（加权）查询 gram 越多、越稀有，得分越高。
        Filters by kb_names if provided.
        """
        candidates = self.segments
        if kb_names:
            candidates = [s for s in self.segments if s.kb_name in kb_names]

        if not candidates:
            return []

        query_ngrams = _char_ngrams(query, n=2)
        if not query_ngrams:
            return candidates[:top_k]

        qweights = {g: self._idf.get(g, self._idf_default) for g in query_ngrams}
        qtotal = sum(qweights.values())
        if qtotal <= 0:
            return candidates[:top_k]

        scored = []
        for seg in candidates:
            if not seg.ngrams:
                continue
            matched = query_ngrams & seg.ngrams
            if not matched:
                continue
            score = sum(qweights[g] for g in matched) / qtotal
            scored.append((score, seg))

        # 覆盖率降序；同分按文档位置靠前优先
        scored.sort(key=lambda x: (-x[0], x[1].position))

        results: list[Segment] = []
        seen_prefix: set = set()
        for _, seg in scored:
            key = seg.content[:60]
            if key in seen_prefix:
                continue
            seen_prefix.add(key)
            results.append(seg)
            if len(results) >= top_k:
                break

        # 兜底：一条都没命中时保持旧行为（返回前 top_k 条），保证上下文不为空
        if not results:
            return candidates[:top_k]
        return results

    def get_kb_names_for_intent(self, intent: str) -> list[str]:
        """Get KB directory names for a given intent ID."""
        names = []
        primary = INTENT_KB_MAP.get(intent, "")
        if primary:
            names.append(primary)
        extra = EXTRA_KB_MAP.get(intent, [])
        names.extend(extra)
        return names

    def retrieve_for_intent(self, query: str, intent: str, top_k: int = 5) -> str:
        """Search and format results for a specific intent."""
        kb_names = self.get_kb_names_for_intent(intent)
        if not kb_names:
            return ""

        results = self.search(query, kb_names=kb_names, top_k=top_k)
        if not results:
            return ""

        lines = []
        for i, seg in enumerate(results, 1):
            # Truncate overly long segments
            content = seg.content[:2000]
            lines.append(f"{content}\n")
        return "\n".join(lines)


# Singleton
_retriever: Optional[KnowledgeRetriever] = None

def get_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever
