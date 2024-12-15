import os
import argparse

import pandas as pd
import gradio as gr

from lrage.gui.gui_utils.utils import (update_dropdown, toggle_interaction_retriver,
                                       lm_eval_avil_model_types, lm_eval_avil_model_args,
                                       retriever_args, reranker_args)
from lrage.gui.gui_utils.tasks import get_all_tasks, get_all_devices
from lrage.gui.gui_utils.evaluation import eval_tasks

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()

if __name__ == '__main__':

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
                    tasks = gr.Dropdown(
                            all_tasks,
                            label="Task",
                            multiselect=True)
                    
                    with gr.Group():
                        model = gr.Radio(
                            choices=lm_eval_avil_model_types, 
                            label="Choose model type")
                        model_args = gr.Dropdown(
                            lm_eval_avil_model_args,
                            value=None,
                            label="Model",
                            scale=1)
                        model.change(update_dropdown, inputs=model, outputs=model_args)

                    with gr.Group():
                    
                        openai_key = gr.Textbox(
                            label="OpenAI API key", 
                            lines=1,
                            value=os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") else "Enter your OpenAI API key: sk-...")
                        hf_token = gr.Textbox(
                            label="Hugging Face token",
                            lines=1,
                            value=os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else "Enter your Hugging Face token: hf_...")
                    
                    with gr.Row():
                        system_instruction = gr.Textbox(
                            label="system_instruction",
                            placeholder="Enter system instruction",
                            lines=6)

                        with gr.Group():
                            max_new_tokens = gr.Slider(
                                0,
                                1024,
                                value=512,
                                step=128,
                                label="Max Generation Length")
                            
                            temperature = gr.Slider(
                                0.0,
                                1.0,
                                value=0.5,
                                step=0.1,
                                label="Temperature")
                            do_sample = gr.Checkbox(
                                value=False,
                                label="Do Sample")
                
                with gr.Column():
                    with gr.Group():
                        with gr.Group():
                            with gr.Column():
                                retrieve_docs = gr.Checkbox(
                                    value=False,
                                    label="retrieve_docs"
                                )

                            with gr.Row():
                                retriever = gr.Dropdown(
                                    ["pyserini"], 
                                    value="pyserini",
                                    label="Retriever", 
                                    interactive=False
                                )
                                top_k = gr.Number(
                                    value=3,
                                    label="Top k", 
                                    interactive=False
                                )

                            retriever_args = gr.Dropdown(
                                retriever_args[retriever.value],
                                value=None,
                                label="Retriever args",
                                interactive=False
                            )

                            retrieve_docs.change(
                                toggle_interaction_retriver,
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
                        reranker_args = gr.Dropdown(
                            reranker_args,
                            label="Reranker args",
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
                                choices=lm_eval_avil_model_types)
                            judge_model_args = gr.Dropdown(
                                lm_eval_avil_model_args,
                                value=None,
                                label="Judge Model",
                                scale=1)
                            judge_model.change(update_dropdown, inputs=judge_model, outputs=judge_model_args)
                    
                    with gr.Column():
                        results = gr.DataFrame(initial_df, label="Results")
                        eval_button = gr.Button("Evlaute")

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
                        retriever_args,
                        rerank,
                        reranker,
                        reranker_args,
                        max_new_tokens,
                        temperature,
                        do_sample,
                        system_instruction,
                    ],
                    outputs=[results]
                )

        demo.launch(server_port=args.port, server_name="0.0.0.0")