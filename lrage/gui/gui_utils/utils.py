import os
import yaml
import gradio as gr
from pathlib import Path

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

current_file = Path(__file__).resolve()
config_file = os.path.join(current_file.parent, "gui_args.yaml")
config = load_yaml_config(config_file)

lm_eval_avil_model_types = config["lm_eval_avil_model_types"]
lm_eval_avil_model_args = config["lm_eval_avil_model_args"]

retriever_args = config["retriever_args"]

reranker_args = config["reranker_args"]

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
