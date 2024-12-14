import os

import pandas as pd
import datasets

from lrage.evaluator import simple_evaluate
from lrage.loggers import EvaluationTracker
from lrage.utils import simple_parse_args_string, handle_non_serializable

def build_table(result_dict, sort_results: bool = True):
    """Generate DataFrame of results."""
    data = []

    keys = result_dict["results"].keys()
    if sort_results:
        keys = sorted(keys)

    for k in keys:
        dic = result_dict["results"][k]
        n_shot = str(result_dict["n-shot"][k])

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        if sort_results:
            metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            stderr_value = dic.get(m + "_stderr" + "," + f, "")
            stderr_value = "%.4f" % stderr_value if stderr_value != "N/A" and stderr_value != "" else ""

            data.append({
                "tasks": k,
                "n-shot": n_shot,
                "metric": m,
                "value": "%.4f" % v,
                "stderr": stderr_value
            })

    df = pd.DataFrame(data, columns=["tasks", "n-shot", "metric", "value", "stderr"])
    return df

def eval_tasks(
        model, 
        model_args, 
        tasks, 
        api_key_name = None,
        api_key = None,
        judge_model = None,
        judge_model_args = None,
        retrieve_docs = False, 
        top_k = 3,
        retriever = None, 
        retriever_args = None,
        rerank = False,
        reranker = None,
        reranker_args = None,
        batch_size = 1, 
        device = ["cpu"], 
        use_cache = None,
        cache_requests = None,
        num_fewshot = 0,
        fewshot_as_multiturn = False,
        apply_chat_template = False,
        gen_kwargs = None,
        system_instruction = None,
        hf_hub_log_args = None,
        log_samples = False,
        output_path = None,):
    
    if use_cache == "":
        use_cache = None
    if gen_kwargs == "":
        gen_kwargs = None
    if output_path == "":
        output_path = None

    if api_key_name and api_key:
        os.environ[api_key_name] = api_key
    
    tasks_list = [task_name for task_name in tasks]
    if output_path:
        hf_hub_log_args += f",output_path={output_path}"
    if os.environ.get("HF_TOKEN", None):
        hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    results = simple_evaluate(
        model=model,
        model_args=model_args,
        judge_model=judge_model,
        judge_model_args=judge_model_args,
        tasks=tasks_list,
        num_fewshot=num_fewshot,
        fewshot_as_multiturn=fewshot_as_multiturn,
        use_cache=use_cache,
        cache_requests=cache_requests,
        gen_kwargs=gen_kwargs,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        retrieve_docs=retrieve_docs,
        top_k=top_k,
        retriever=retriever,
        retriever_args=retriever_args,
        rerank=rerank,
        reranker=reranker,
        reranker_args=reranker_args,
        batch_size=batch_size,
        device=device,
        evaluation_tracker=evaluation_tracker,
    )

    if results is not None:
        if log_samples:
            samples = results.pop("samples")

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if log_samples else None
        )

        if log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

    return build_table(results)