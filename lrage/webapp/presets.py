"""Presets served to the UI, read from the Gradio GUI's gui_args.yaml in place
so both UIs stay in sync."""
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def _gui_args_path() -> Path:
    import lrage.gui.gui_utils as gui_utils  # empty __init__, no gradio import

    return Path(gui_utils.__file__).parent / "gui_args.yaml"


@lru_cache(maxsize=1)
def load_presets() -> Dict[str, Any]:
    with open(_gui_args_path(), encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # {display label: registry key} -> [{key, label}]
    model_types = [
        {"key": key, "label": label}
        for label, key in raw.get("lm_eval_avil_model_types", {}).items()
    ]
    # {registry key: {preset label: args string}} -> {key: [{label, args}]}
    model_presets = {
        model_type: [{"label": label, "args": args} for label, args in presets.items()]
        for model_type, presets in raw.get("lm_eval_avil_model_args", {}).items()
    }
    retriever_presets = {
        retriever: list(presets)
        for retriever, presets in raw.get("retriever_args", {}).items()
    }
    reranker_presets = list(raw.get("reranker_args", []))

    return {
        "model_types": model_types,
        "model_presets": model_presets,
        "retriever_presets": retriever_presets,
        "reranker_presets": reranker_presets,
    }
