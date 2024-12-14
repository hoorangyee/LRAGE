import argparse
import pandas as pd
import gradio as gr

from gui_utils.utils import toggle_visiblity, toggle_interactive
from gui_utils.tasks import get_all_tasks, get_all_devices
from gui_utils.evaluation import eval_tasks

all_tasks = get_all_tasks() 
devices = get_all_devices()
initial_df = pd.DataFrame(columns=["tasks", "n-shot","metric", "value", "stderr"])

lm_eval_avil_models = ["huggingface",
                       "openai-chat-completions"]

with gr.Blocks(title="⚖️ LRAGE") as demo:
    gr.Markdown(
        """<h1 style='text-align: center'>⚖️ LRAGE: Legal Retrieval Augmented Generation Evaluation Tool</h1>""",
        elem_id="title"
    )
    with gr.Tab(label='evaluation'):
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    model = gr.Dropdown(
                        label="Model type", 
                        choices=lm_eval_avil_models)
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
                with gr.Column():
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
                            choices=lm_eval_avil_models)
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

            with gr.Column():
                with gr.Row():
                    tasks = gr.Dropdown(
                        all_tasks, 
                        label="Task", 
                        multiselect=True)
                
                gen_kwargs = gr.Textbox(
                    label="gen_kwargs",
                    placeholder="e.g. max_length=100,temperature=0.5,...",
                    lines=1)
                
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
                ],
                outputs=[results]
            )

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()

if __name__ == '__main__':
    demo.launch(server_port=args.port)