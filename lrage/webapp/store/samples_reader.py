"""Paginated, filterable access to the samples_*.jsonl files written by
EvaluationTracker. Files are immutable once a run completes, so parsed
contents are cached keyed by (path, mtime)."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lrage.utils import get_file_task_name

# Keys that are structural rather than per-sample metric values.
NON_METRIC_KEYS = {
    "doc_id",
    "doc",
    "target",
    "arguments",
    "resps",
    "filtered_resps",
    "doc_hash",
    "prompt_hash",
    "target_hash",
    "input_len",
    "Rating",
    "Explanation",
}

_CACHE_MAX = 4
_cache: Dict[Any, List[Dict[str, Any]]] = {}


def task_name_for(path: str) -> str:
    return get_file_task_name(Path(path).stem)


def load_samples(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _cache:
        with open(p, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        while len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = samples
    return _cache[key]


def _first_resp(sample: Dict[str, Any]) -> str:
    resps = sample.get("filtered_resps") or sample.get("resps") or []
    node = resps
    while isinstance(node, list) and node:
        node = node[0]
    return str(node) if node is not None and not isinstance(node, list) else ""


def metric_values(sample: Dict[str, Any]) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in sample.items()
        if k not in NON_METRIC_KEYS and isinstance(v, (int, float))
        and not isinstance(v, bool)
    }


def to_list_item(sample: Dict[str, Any]) -> Dict[str, Any]:
    doc = sample.get("doc")
    doc_text = ""
    if isinstance(doc, dict):
        for value in doc.values():
            if isinstance(value, str) and value.strip():
                doc_text = value
                break
    elif isinstance(doc, str):
        doc_text = doc
    return {
        "doc_id": sample.get("doc_id"),
        "input": doc_text[:280],
        "target": str(sample.get("target", ""))[:280],
        "resp": _first_resp(sample)[:280],
        "metrics": metric_values(sample),
        "rating": sample.get("Rating"),
    }


def filter_samples(
    samples: List[Dict[str, Any]],
    q: Optional[str] = None,
    incorrect_only: bool = False,
    judge_min: Optional[float] = None,
    judge_max: Optional[float] = None,
) -> List[Dict[str, Any]]:
    out = []
    needle = q.lower().strip() if q else None
    for sample in samples:
        if needle:
            haystack = " ".join(
                [
                    json.dumps(sample.get("doc"), ensure_ascii=False, default=str),
                    str(sample.get("target", "")),
                    _first_resp(sample),
                ]
            ).lower()
            if needle not in haystack:
                continue
        if incorrect_only:
            metrics = metric_values(sample)
            if not metrics or not any(v == 0.0 for v in metrics.values()):
                continue
        rating = sample.get("Rating")
        if judge_min is not None and (rating is None or rating < judge_min):
            continue
        if judge_max is not None and (rating is None or rating > judge_max):
            continue
        out.append(sample)
    return out


def find_sample(
    samples: List[Dict[str, Any]], doc_id: int
) -> Optional[Dict[str, Any]]:
    # doc_ids are written in order, so try direct indexing first.
    if 0 <= doc_id < len(samples) and samples[doc_id].get("doc_id") == doc_id:
        return samples[doc_id]
    for sample in samples:
        if sample.get("doc_id") == doc_id:
            return sample
    return None
