import gradio as gr

def toggle_visiblity(value):
    if value:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
    
def toggle_interaction_retriver(is_checked):
    if is_checked:
        return [
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True)
        ]
    else:
        return [
            gr.update(interactive=False, value=None),
            gr.update(interactive=False, value=None),
            gr.update(interactive=False, value="")
        ]

def update_dropdown(selected_model_type):
    return gr.update(choices=lm_eval_avil_model_args[lm_eval_avil_model_types[selected_model_type]])

custom_css = """
.gr-textbox input:disabled {
    background-color: #f0f0f0;
    color: #888;
    cursor: not-allowed;
}
"""

lm_eval_avil_model_types = {"🤗 Hugging Face": "huggingface",
                            "OpenAI": "openai-chat-completions"}

lm_eval_avil_model_args = {"huggingface": {"Llama-3.1-8B": "pretrained=meta-llama/Llama-3.1-8B-Instruct",
                                            "Qwen2.5-7B": "pretrained=Qwen/Qwen2.5-7B-Instruct",
                                            "SaulLM-7B": "pretrained=meta-llama/Llama-3.1-8B",
                                            "Gemma-2-9B": "pretrained=google/gemma-2-9b-it",
                                            "Phi-3.5-mini": "pretrained=microsoft/Phi-3.5-mini-instruct"},
                           "openai-chat-completions": {"gpt-4o": "model=gpt-4o",
                                                       "gpt-4o-2024-11-20": "model=gpt-4o-2024-11-20",
                                                       "gpt-4o-2024-08-06": "model=gpt-4o-2024-08-06",
                                                       "gpt-4o-mini": "model=gpt-4o-mini",
                                                       "gpt-4o-mini-2024-07-18": "model=gpt-4o-mini-2024-07-18",
                                                       "o1-preview": "model=o1-preview",}}