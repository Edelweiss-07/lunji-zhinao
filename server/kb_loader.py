"""
Knowledge base loader and retriever.
Loads all extracted Coze KB segments and provides search.
"""
import json
import re
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

class Segment:
    __slots__ = ("content", "doc_id", "kb_name", "position")
    def __init__(self, content: str, doc_id: str, kb_name: str, position: int):
        self.content = content
        self.doc_id = doc_id
        self.kb_name = kb_name
        self.position = position


class KnowledgeRetriever:
    def __init__(self):
        self.segments: list[Segment] = []
        self._load_all()

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

    def _char_ngrams(self, text: str, n: int = 2) -> set:
        """Generate character n-grams for Chinese text matching."""
        text = re.sub(r'\s+', '', text)
        if len(text) < n:
            return {text}
        return {text[i:i+n] for i in range(len(text) - n + 1)}

    def search(self, query: str, kb_names: Optional[list[str]] = None, top_k: int = 5) -> list[Segment]:
        """
        Search for relevant segments.
        Uses n-gram Jaccard similarity (works well for Chinese).
        Filters by kb_names if provided.
        """
        candidates = self.segments
        if kb_names:
            candidates = [s for s in self.segments if s.kb_name in kb_names]

        if not candidates:
            return []

        query_ngrams = self._char_ngrams(query, n=2)
        if not query_ngrams:
            return candidates[:top_k]

        scored = []
        for seg in candidates:
            seg_ngrams = self._char_ngrams(seg.content, n=2)
            if not seg_ngrams:
                continue
            intersection = query_ngrams & seg_ngrams
            union = query_ngrams | seg_ngrams
            score = len(intersection) / len(union) if union else 0
            scored.append((score, seg))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [seg for _, seg in scored[:top_k]]

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
