import gradio as gr

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