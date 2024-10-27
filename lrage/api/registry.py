import logging
from typing import Callable, Dict

import evaluate as hf_evaluate

from lrage.api.model import LM
from lrage.api.retriever import Retriever
from lrage.api.reranker import Reranker


eval_logger = logging.getLogger("lrage")

MODEL_REGISTRY = {}


def register_model(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, LM
            ), f"Model '{name}' ({cls.__name__}) must extend LM class"

            assert (
                name not in MODEL_REGISTRY
            ), f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead."

            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


def get_model(model_name):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load model '{model_name}', but no model for this name found! Supported model names: {', '.join(MODEL_REGISTRY.keys())}"
        )


RETRIEVER_REGISTRY = {}


def register_retriever(*names):
    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, Retriever
            ), f"Retriever '{name}' ({cls.__name__}) must extend Retriever class"

            assert (
                name not in RETRIEVER_REGISTRY
            ), f"Retriever named '{name}' conflicts with existing retriever! Please register with a non-conflicting alias instead."

            RETRIEVER_REGISTRY[name] = cls
        return cls

    return decorate


def get_retriever(retriever_name):
    try:
        return RETRIEVER_REGISTRY[retriever_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load retriever '{retriever_name}', but no retriever for this name found! Supported retriever names: {', '.join(RETRIEVER_REGISTRY.keys())}"
        )
    

RERANKER_REGISTRY = {}


def register_reranker(*names):
    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, Reranker
            ), f"Reranker '{name}' ({cls.__name__}) must extend Reranker class"

            assert (
                name not in RERANKER_REGISTRY
            ), f"Reranker named '{name}' conflicts with existing reranker! Please register with a non-conflicting alias instead."

            RERANKER_REGISTRY[name] = cls
        return cls

    return decorate



def get_reranker(reranker_name):
    try:
        return RERANKER_REGISTRY[reranker_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load reranker '{reranker_name}', but no reranker for this name found! Supported reranker names: {', '.join(RERANKER_REGISTRY.keys())}"
        )
    
    
TASK_REGISTRY = {}
GROUP_REGISTRY = {}
ALL_TASKS = set()
func2task_index = {}


def register_task(name):
    def decorate(fn):
        assert (
            name not in TASK_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        ALL_TASKS.add(name)
        func2task_index[fn.__name__] = name
        return fn

    return decorate


def register_group(name):
    def decorate(fn):
        func_name = func2task_index[fn.__name__]
        if name in GROUP_REGISTRY:
            GROUP_REGISTRY[name].append(func_name)
        else:
            GROUP_REGISTRY[name] = [func_name]
            ALL_TASKS.add(name)
        return fn

    return decorate


OUTPUT_TYPE_REGISTRY = {}
METRIC_REGISTRY = {}
METRIC_AGGREGATION_REGISTRY = {}
AGGREGATION_REGISTRY: Dict[str, Callable[[], Dict[str, Callable]]] = {}
HIGHER_IS_BETTER_REGISTRY = {}
FILTER_REGISTRY = {}

DEFAULT_METRIC_REGISTRY = {
    "loglikelihood": [
        "perplexity",
        "acc",
    ],
    "loglikelihood_rolling": ["word_perplexity", "byte_perplexity", "bits_per_byte"],
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}


def register_metric(**args):
    # TODO: do we want to enforce a certain interface to registered metrics?
    def decorate(fn):
        assert "metric" in args
        name = args["metric"]

        for key, registry in [
            ("metric", METRIC_REGISTRY),
            ("higher_is_better", HIGHER_IS_BETTER_REGISTRY),
            ("aggregation", METRIC_AGGREGATION_REGISTRY),
        ]:
            if key in args:
                value = args[key]
                assert (
                    value not in registry
                ), f"{key} named '{value}' conflicts with existing registered {key}!"

                if key == "metric":
                    registry[name] = fn
                elif key == "aggregation":
                    registry[name] = AGGREGATION_REGISTRY[value]
                else:
                    registry[name] = value

        return fn

    return decorate


def get_metric(name: str, hf_evaluate_metric=False) -> Callable:
    if not hf_evaluate_metric:
        if name in METRIC_REGISTRY:
            return METRIC_REGISTRY[name]
        else:
            eval_logger.warning(
                f"Could not find registered metric '{name}' in lrage, searching in HF Evaluate library..."
            )

    try:
        metric_object = hf_evaluate.load(name)
        return metric_object.compute
    except Exception:
        eval_logger.error(
            f"{name} not found in the evaluate library! Please check https://huggingface.co/evaluate-metric",
        )


def register_aggregation(name: str):
    def decorate(fn):
        assert (
            name not in AGGREGATION_REGISTRY
        ), f"aggregation named '{name}' conflicts with existing registered aggregation!"

        AGGREGATION_REGISTRY[name] = fn
        return fn

    return decorate


def get_aggregation(name: str) -> Callable[[], Dict[str, Callable]]:
    try:
        return AGGREGATION_REGISTRY[name]
    except KeyError:
        eval_logger.warning(f"{name} not a registered aggregation metric!")


def get_metric_aggregation(name: str) -> Callable[[], Dict[str, Callable]]:
    try:
        return METRIC_AGGREGATION_REGISTRY[name]
    except KeyError:
        eval_logger.warning(f"{name} metric is not assigned a default aggregation!")


def is_higher_better(metric_name) -> bool:
    try:
        return HIGHER_IS_BETTER_REGISTRY[metric_name]
    except KeyError:
        eval_logger.warning(
            f"higher_is_better not specified for metric '{metric_name}'!"
        )


def register_filter(name):
    def decorate(cls):
        if name in FILTER_REGISTRY:
            eval_logger.info(
                f"Registering filter `{name}` that is already in Registry {FILTER_REGISTRY}"
            )
        FILTER_REGISTRY[name] = cls
        return cls

    return decorate


def get_filter(filter_name: str) -> type:
    try:
        return FILTER_REGISTRY[filter_name]
    except KeyError:
        eval_logger.warning(f"filter `{filter_name}` is not registered!")
