import os
import argparse

import pandas as pd
import gradio as gr

from lrage.gui.gui_utils.utils import (update_dropdown, toggle_interaction_retriver,
                                       lm_eval_avil_model_types, lm_eval_avil_model_args,
                                       retriever_args, reranker_args)
from lrage.gui.gui_utils.tasks import get_all_tasks, get_all_devices
from lrage.gui.gui_utils.evaluation import eval_tasks

def create_retriever_arg_labels(retriever_type, raw_args_list):
    result = {}
    for raw_arg in raw_args_list:
        parts = raw_arg.split(',')
        arg_dict = {}
        for part in parts:
            key, value = part.split('=', 1)
            arg_dict[key] = value
        
        if retriever_type == "pyserini":
            if arg_dict.get("retriever_type") == "bm25":
                index_name = os.path.basename(arg_dict.get("bm25_index_path", "unknown"))
                label = f"BM25 - {index_name}"
            elif arg_dict.get("retriever_type") == "dense":
                index_name = os.path.basename(arg_dict.get("faiss_index_path", "unknown"))
                encoder_name = os.path.basename(arg_dict.get("encoder_path", "unknown"))
                label = f"Dense ({encoder_name}) - {index_name}"
            else:
                label = f"Unknown - {raw_arg[:30]}..."
        else:
            label = raw_arg
        
        result[label] = raw_arg
    
    return result

pyserini_labels = create_retriever_arg_labels("pyserini", retriever_args["pyserini"])

reranker_labels = {}
for raw_arg in reranker_args:
    parts = raw_arg.split(',')
    arg_dict = {}
    for part in parts:
        key, value = part.split('=', 1)
        arg_dict[key] = value
    
    if "reranker_type" in arg_dict:
        label = f"{arg_dict['reranker_type'].capitalize()} Reranker"
        reranker_labels[label] = raw_arg
    else:
        reranker_labels[raw_arg] = raw_arg

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()

if __name__ == '__main__':

    all_tasks = get_all_tasks()
    devices = get_all_devices()
    initial_df = pd.DataFrame(columns=["tasks", "n-shot", "metric", "value", "stderr"])

    with gr.Blocks(title="⚖️ LRAGE") as demo:
        gr.Markdown(
            """<h1 style='text-align: center'>⚖️ LRAGE: Legal Retrieval Augmented Generation Evaluation Tool</h1>""",
            elem_id="title"
        )
        
        tasks = gr.State([])
        model = gr.State(None)
        model_args = gr.State(None)
        openai_key = gr.State(os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else "")
        hf_token = gr.State(os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else "")
        system_instruction = gr.State("")
        max_new_tokens = gr.State(512)
        temperature = gr.State(0.5)
        do_sample = gr.State(False)
        retrieve_docs = gr.State(False)
        retriever = gr.State("pyserini")
        top_k = gr.State(3)
        retriever_args_val = gr.State(None)
        rerank = gr.State(False)
        reranker = gr.State("rerankers")
        reranker_args_val = gr.State(None)
        judge_model = gr.State(None)
        judge_model_args = gr.State(None)
        
        def toggle_interaction_retriever_friendly(is_checked):
            if is_checked:
                return [
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True)
                ]
            else:
                return [
                    gr.update(interactive=False),
                    gr.update(interactive=False, value=3),
                    gr.update(interactive=False)
                ]
        
        def get_raw_retriever_arg(friendly_label):
            return pyserini_labels.get(friendly_label, None)
        
        def get_raw_reranker_arg(friendly_label):
            return reranker_labels.get(friendly_label, None)
        
        with gr.Tabs():
            with gr.Tab(label='Task Selection'):
                tasks_input = gr.Dropdown(
                    all_tasks,
                    label="Tasks",
                    multiselect=True,
                    info="Select tasks to evaluate")
                
                tasks_update_btn = gr.Button("Save Task Settings")
                tasks_update_btn.click(
                    lambda x: x,
                    inputs=[tasks_input],
                    outputs=[tasks]
                )
            
            with gr.Tab(label='Model Settings'):
                with gr.Group():
                    with gr.Row():
                        model_input = gr.Radio(
                            choices=lm_eval_avil_model_types, 
                            label="Choose model type")
                        model_args_input = gr.Dropdown(
                            lm_eval_avil_model_args,
                            value=None,
                            label="Model",
                            scale=1)
                        model_input.change(update_dropdown, inputs=model_input, outputs=model_args_input)

                with gr.Group():
                    openai_key_input = gr.Textbox(
                        label="OpenAI API key", 
                        lines=1,
                        placeholder="Enter your OpenAI API key: sk-...",
                        value=os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else "")
                    hf_token_input = gr.Textbox(
                        label="Hugging Face token",
                        lines=1,
                        placeholder="Enter your Hugging Face token: hf_...",
                        value=os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else "")
                
                model_settings_update_btn = gr.Button("Save Model Settings")
                model_settings_update_btn.click(
                    lambda m, ma, ok, hf: [m, ma, ok, hf],
                    inputs=[model_input, model_args_input, openai_key_input, hf_token_input],
                    outputs=[model, model_args, openai_key, hf_token]
                )
            
            with gr.Tab(label='Generation Settings'):
                system_instruction_input = gr.Textbox(
                    label="System Instruction",
                    placeholder="Enter system instruction",
                    lines=6)
                
                with gr.Group():
                    max_new_tokens_input = gr.Slider(
                        0,
                        1024,
                        value=512,
                        step=128,
                        label="Max Generation Length")
                    
                    temperature_input = gr.Slider(
                        0.0,
                        1.0,
                        value=0.5,
                        step=0.1,
                        label="Temperature")
                    do_sample_input = gr.Checkbox(
                        value=False,
                        label="Do Sample")
                
                gen_settings_update_btn = gr.Button("Save Generation Settings")
                gen_settings_update_btn.click(
                    lambda si, mnt, t, ds: [si, mnt, t, ds],
                    inputs=[system_instruction_input, max_new_tokens_input, temperature_input, do_sample_input],
                    outputs=[system_instruction, max_new_tokens, temperature, do_sample]
                )
            
            with gr.Tab(label='Retrieval Settings'):
                with gr.Group():
                    gr.Markdown("### Document Retrieval Settings")
                    retrieve_docs_input = gr.Checkbox(
                        value=False,
                        label="Retrieve Documents")
                    
                    with gr.Row():
                        retriever_input = gr.Dropdown(
                            ["pyserini"], 
                            value="pyserini",
                            label="Retriever", 
                            interactive=False)
                        top_k_input = gr.Number(
                            value=3,
                            label="Top K", 
                            interactive=False)
                    
                    retriever_args_input = gr.Dropdown(
                        choices=list(pyserini_labels.keys()),
                        value=None,
                        label="Retriever Configuration",
                        interactive=False,
                        info="Select a retrieval configuration")
                    
                    retrieve_docs_input.change(
                        toggle_interaction_retriever_friendly,
                        inputs=[retrieve_docs_input],
                        outputs=[retriever_input, top_k_input, retriever_args_input]
                    )
                
                with gr.Group():
                    gr.Markdown("### Reranking Settings")
                    rerank_input = gr.Checkbox(
                        value=False,
                        label="Rerank")
                    
                    reranker_input = gr.Dropdown(
                        ["rerankers"], 
                        value="rerankers",
                        label="Reranker", 
                        visible=False)
                    reranker_args_input = gr.Dropdown(
                        choices=list(reranker_labels.keys()),
                        label="Reranker Configuration",
                        visible=False,
                        info="Select a reranking configuration")
                    
                    rerank_input.change(
                        lambda visible: [gr.update(visible=visible), gr.update(visible=visible)],
                        inputs=[rerank_input],
                        outputs=[reranker_input, reranker_args_input]
                    )
                
                def save_retrieval_settings(rd, r, tk, ra_friendly, rr, rk, rka_friendly):
                    ra_raw = get_raw_retriever_arg(ra_friendly) if ra_friendly else None
                    rka_raw = get_raw_reranker_arg(rka_friendly) if rka_friendly else None
                    return [rd, r, tk, ra_raw, rr, rk, rka_raw]
                
                retrieval_settings_update_btn = gr.Button("Save Retrieval Settings")
                retrieval_settings_update_btn.click(
                    save_retrieval_settings,
                    inputs=[retrieve_docs_input, retriever_input, top_k_input, retriever_args_input, 
                           rerank_input, reranker_input, reranker_args_input],
                    outputs=[retrieve_docs, retriever, top_k, retriever_args_val, rerank, reranker, reranker_args_val]
                )
            
            with gr.Tab(label='Evaluation Settings'):
                with gr.Group():
                    gr.Markdown("### LLM-as-a-Judge Settings")
                    with gr.Row():
                        judge_model_input = gr.Radio(
                            choices=lm_eval_avil_model_types,
                            label="Judge Model type")
                        judge_model_args_input = gr.Dropdown(
                            lm_eval_avil_model_args,
                            value=None,
                            label="Judge Model",
                            scale=1)
                        judge_model_input.change(update_dropdown, inputs=judge_model_input, outputs=judge_model_args_input)
                
                judge_settings_update_btn = gr.Button("Save Evaluation Settings")
                judge_settings_update_btn.click(
                    lambda jm, jma: [jm, jma],
                    inputs=[judge_model_input, judge_model_args_input],
                    outputs=[judge_model, judge_model_args]
                )
            
            with gr.Tab(label='Results and Execution'):
                with gr.Row():
                    with gr.Column():
                        current_settings = gr.JSON(label="Current Settings", value={})
                        
                        def update_settings_summary(tasks, model, model_args, 
                                                   openai_key, hf_token, 
                                                   system_instruction, max_new_tokens, temperature, do_sample,
                                                   retrieve_docs, retriever, top_k, retriever_args_raw,
                                                   rerank, reranker, reranker_args_raw,
                                                   judge_model, judge_model_args):
                            retriever_friendly = next((k for k, v in pyserini_labels.items() if v == retriever_args_raw), retriever_args_raw)
                            reranker_friendly = next((k for k, v in reranker_labels.items() if v == reranker_args_raw), reranker_args_raw)
                            
                            return {
                                "Tasks": tasks,
                                "Model": {"Type": model, "Args": model_args},
                                "API Keys": {"OpenAI": "Set" if openai_key else "Not set", 
                                          "HF": "Set" if hf_token else "Not set"},
                                "Generation Settings": {
                                    "System Instruction": system_instruction[:30] + "..." if system_instruction and len(system_instruction) > 30 else system_instruction,
                                    "Max Tokens": max_new_tokens,
                                    "Temperature": temperature,
                                    "Do Sample": do_sample
                                },
                                "Retrieval Settings": {
                                    "Retrieve Documents": retrieve_docs,
                                    "Retriever": retriever if retrieve_docs else "Not used",
                                    "Top K": top_k if retrieve_docs else "Not used",
                                    "Retriever Config": retriever_friendly if retrieve_docs and retriever_args_raw else "Not selected",
                                    "Rerank": rerank,
                                    "Reranker": reranker if rerank else "Not used",
                                    "Reranker Config": reranker_friendly if rerank and reranker_args_raw else "Not selected"
                                },
                                "Evaluation Settings": {
                                    "Judge Model": judge_model,
                                    "Judge Model Args": judge_model_args
                                }
                            }
                        
                        update_summary_btn = gr.Button("Update Settings Summary")
                        update_summary_btn.click(
                            update_settings_summary,
                            inputs=[
                                tasks, model, model_args, 
                                openai_key, hf_token, 
                                system_instruction, max_new_tokens, temperature, do_sample,
                                retrieve_docs, retriever, top_k, retriever_args_val,
                                rerank, reranker, reranker_args_val,
                                judge_model, judge_model_args
                            ],
                            outputs=[current_settings]
                        )
                    
                    with gr.Column():
                        results = gr.DataFrame(initial_df, label="Evaluation Results")
                
                eval_button = gr.Button("Run Evaluation", variant="primary", size="lg")
                eval_button.click(
                    eval_tasks,
                    inputs=[
                        model, 
                        model_args, 
                        tasks,
                        openai_key,
                        hf_token,
                        judge_model,
                        judge_model_args,
                        retrieve_docs, 
                        top_k,
                        retriever, 
                        retriever_args_val,
                        rerank,
                        reranker,
                        reranker_args_val,
                        max_new_tokens,
                        temperature,
                        do_sample,
                        system_instruction,
                    ],
                    outputs=[results]
                )

        demo.launch(server_port=args.port, server_name="0.0.0.0")