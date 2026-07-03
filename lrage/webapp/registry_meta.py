"""Metadata about available tasks, model backends, and devices.

Task listing imports `lrage.tasks` (transformers/datasets chain, seconds of
import + index time), so it is loaded lazily behind a lock and cached for the
process lifetime. Model/retriever/reranker names are a curated static list:
the live registries in `lrage.api.registry` only populate once `lrage.models`
(torch and friends) is imported, which the server avoids until a run starts.
"""
import threading
from typing import Dict, List, Optional

# Registry keys from @register_model decorators that make sense to drive from
# a web UI. The full registry (lrage/models/) also contains niche backends
# (nemo, neuronx, sparseml, ...) that stay CLI-only.
MODEL_TYPES: List[Dict[str, str]] = [
    {"key": "huggingface", "label": "Hugging Face"},
    {"key": "vllm", "label": "vLLM"},
    {"key": "openai-chat-completions", "label": "OpenAI (chat)"},
    {"key": "openai-completions", "label": "OpenAI (completions)"},
    {"key": "anthropic-chat-completions", "label": "Anthropic"},
    {"key": "local-chat-completions", "label": "OpenAI-compatible server"},
]

RETRIEVERS = ["pyserini"]
RETRIEVER_TYPES = ["bm25", "sparse", "dense", "hybrid"]
RERANKERS = ["rerankers"]
RERANKER_TYPES = ["colbert", "cross-encoder", "t5"]

_tasks_lock = threading.Lock()
_tasks_cache: Optional[List[Dict[str, str]]] = None


def list_tasks() -> List[Dict[str, str]]:
    """Return [{name, type}] for every indexed task/group. Slow on first call."""
    global _tasks_cache
    if _tasks_cache is None:
        with _tasks_lock:
            if _tasks_cache is None:
                from lrage.tasks import TaskManager

                tm = TaskManager()
                _tasks_cache = [
                    {"name": name, "type": tm.task_index[name].get("type", "task")}
                    for name in tm.all_tasks
                ]
    return _tasks_cache


def tasks_loaded() -> bool:
    return _tasks_cache is not None


def list_devices() -> List[str]:
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            devices += ["cuda"]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    return devices
