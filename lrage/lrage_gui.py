import sys
import os
import yaml
import json

import lrage
from lrage.loggers import EvaluationTracker
from lrage.utils import simple_parse_args_string, handle_non_serializable

import gradio as gr
import pandas as pd
import torch

def toggle_visiblity(value):
    if value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def toggle_interactive(value):
    if value:
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)
    
def get_all_tasks():
    task_manager = lrage.tasks.TaskManager()
    return task_manager.all_tasks
 
def get_all_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        devices += ["cuda"]
    return devices
    
def build_table(result_dict, sort_results: bool = True):
    """Generate DataFrame of results."""
    
    data = []

    keys = result_dict["results"].keys()
    if sort_results:
        # Sort entries alphabetically
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
        output_path = None,
        random_seed = 0,
        numpy_random_seed = 1234,
        torch_random_seed = 1234,
        fewshot_random_seed = 1234):
    
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

    results = lrage.evaluator.simple_evaluate(
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
        random_seed=random_seed,
        numpy_random_seed=numpy_random_seed,
        torch_random_seed=torch_random_seed,
        fewshot_random_seed=fewshot_random_seed,
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

all_tasks = get_all_tasks() 
devices = get_all_devices()
initial_df = pd.DataFrame(columns=["tasks", "n-shot","metric", "value", "stderr"])

with gr.Blocks(title="⚖️ LRAGE") as demo:
    gr.Markdown(
        """<h1 style='text-align: center'>⚖️ LRAGE: Legal Retrieval Augmented Generation Evaluation Tool</h1>""",
        elem_id="title"
    )
    with gr.Tab(label='evaluation'):
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model type", 
                    choices=[
                        "anthropic", 
                        "anthropic-chat-completions", 
                        "huggingface", 
                        "mamba_ssm", 
                        "nemo_lm", 
                        "sparseml", 
                        "neuronx", 
                        "openai-completions", 
                        "openai-chat-completions", 
                        "openvino", 
                        "textsynth", 
                        "vllm"])
                model_args = gr.Textbox(
                    label="Model args", 
                    placeholder="e.g. pretrained=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True,...",
                    lines=1)
                api_key_name = gr.Textbox(
                    label="API key name",
                    placeholder="Enter API key name for API models",
                    lines=1)
                api_key = gr.Textbox(
                    label="API key",
                    placeholder="Enter API key for API models",
                    lines=1)

                with gr.Row():
                    tasks = gr.Dropdown(
                        all_tasks, 
                        label="Task", 
                        multiselect=True)

                with gr.Row():
                    device = gr.Dropdown(
                        devices, 
                        value="cpu",
                        label="Device", 
                        scale=1)
                    batch_size = gr.Number(
                        value=1,
                        minimum=1,
                        label="batch size", 
                        scale=1)
                    max_batch_size = gr.Number(
                        value=0,
                        minimum=0,
                        label="max batch size", 
                        scale=1)
                
                with gr.Row():
                    random_seed = gr.Number(
                        value=0,
                        label="random seed",
                        scale=1,
                        min_width=100)
                    numpy_random_seed = gr.Number(
                        value=1234,
                        label="numpy seed",
                        scale=1,
                        min_width=100)
                    torch_random_seed = gr.Number(
                        value=1234,
                        label="torch seed",
                        scale=1,
                        min_width=100)
                    fewshot_random_seed = gr.Number(
                        value=1234,
                        label="fewshot seed",
                        scale=1,
                        min_width=100)

                with gr.Row():
                    with gr.Row():
                        num_fewshot = gr.Number(
                            value=0,
                            minimum=0,
                            label="num_fewshot")
                        fewshot_as_multiturn = gr.Checkbox(
                            value=False, 
                            label="fewshot_as_multiturn", 
                            interactive=False)

                        num_fewshot.change(
                            toggle_interactive,
                            inputs=[num_fewshot],
                            outputs=[fewshot_as_multiturn]
                        )
                    
                    with gr.Row():
                        apply_chat_template = gr.Checkbox(
                            value=False,
                            label="apply_chat_template")
                        template_name = gr.Textbox(
                            label="template_name", 
                            placeholder="If not provided, use default template.",
                            lines=1, 
                            interactive=False)

                        apply_chat_template.change(
                            toggle_interactive,
                            inputs=[apply_chat_template],
                            outputs=[template_name]
                        )

                gen_kwargs = gr.Textbox(
                    label="gen_kwargs", 
                    placeholder="e.g. max_length=100,temperature=0.5,...",
                    lines=1)
                
                system_instruction = gr.Textbox(
                    label="system_instruction",
                    placeholder="Enter system instruction",
                    lines=4)

                with gr.Row():
                    use_cache = gr.Textbox(
                        value=None,
                        label="use_cache", 
                        placeholder="Enter cache path",
                        lines=1)
                    cache_requests = gr.Dropdown(
                        ["true", "refresh" , "delete"], 
                        value=None,
                        label="cache_requests",  
                        interactive=False)

                    use_cache.change(
                        toggle_interactive,
                        inputs=[use_cache],
                        outputs=[cache_requests]
                    )

                with gr.Column():
                    output_path = gr.Textbox(
                        label="output path", 
                        lines=1
                        )
                    log_samples = gr.Checkbox(
                        value=False,
                        label="log_samples",
                        interactive=False
                        )

                    output_path.change(
                        toggle_interactive,
                        inputs=[output_path],
                        outputs=[log_samples]
                    )
                with gr.Group():
                    with gr.Column():
                        retrieve_docs = gr.Checkbox(
                            value=False,
                            label="retrieve_docs")
                        with gr.Row():
                            retriever = gr.Dropdown(
                                ["pyserini"], 
                                value="pyserini",
                                label="Retriever", 
                                visible=False)
                            top_k = gr.Number(
                                value=3,
                                label="Top k", 
                                visible=False)
                        retriever_args = gr.Textbox(
                            label="Retriever args", 
                            lines=1, 
                            visible=False)

                        retrieve_docs.change(
                            lambda visible: [gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)],
                            inputs=[retrieve_docs],
                            outputs=[retriever, top_k, retriever_args]
                        )

                with gr.Column():
                    rerank = gr.Checkbox(
                        value=False,
                        label="rerank")
                    reranker = gr.Dropdown(
                        ["rerankers"], 
                        value="rerankers",
                        label="Reranker", 
                        visible=False)
                    reranker_args = gr.Textbox(
                        label="Reranker args", 
                        lines=1, 
                        visible=False)

                    rerank.change(
                        lambda visible: [gr.update(visible=visible), gr.update(visible=visible)],
                        inputs=[rerank],
                        outputs=[reranker, reranker_args]
                    )
                with gr.Accordion(label="LLM-as-a-Judge", open=False):
                    with gr.Column():
                        judge_model = gr.Dropdown(
                            value=None,
                            label="Judge Model type", 
                            choices=[
                                "anthropic", 
                                "anthropic-chat-completions", 
                                "huggingface", 
                                "mamba_ssm", 
                                "nemo_lm", 
                                "sparseml", 
                                "neuronx", 
                                "openai-completions", 
                                "openai-chat-completions", 
                                "openvino", 
                                "textsynth", 
                                "vllm"])
                        judge_model_args = gr.Textbox(
                            value=None,
                            label="Judge Model args", 
                            placeholder="e.g. pretrained=meta-llama/Meta-Llama-3-8B-Instruct,trust_remote_code=True,...",
                            lines=1)
                
                hf_hub_log_args = gr.Textbox(
                    value=None,
                    label="hf_hub_log_args", 
                    placeholder="Enter comma separated key=value pairs",
                    lines=1)

                eval_button = gr.Button("Evlaute")

            with gr.Column():
                results = gr.DataFrame(initial_df, label="Results")

            eval_button.click(
                eval_tasks,
                inputs=[
                    model, 
                    model_args, 
                    tasks, 
                    api_key_name,
                    api_key,
                    judge_model,
                    judge_model_args,
                    retrieve_docs, 
                    top_k,
                    retriever, 
                    retriever_args,
                    rerank,
                    reranker,
                    reranker_args,
                    batch_size, 
                    device, 
                    use_cache,
                    cache_requests,
                    num_fewshot,
                    fewshot_as_multiturn,
                    apply_chat_template,
                    gen_kwargs,
                    system_instruction,
                    hf_hub_log_args,
                    log_samples,
                    output_path,
                    random_seed,
                    numpy_random_seed,
                    torch_random_seed,
                    fewshot_random_seed
                ],
                outputs=[results]
            )

demo.launch()