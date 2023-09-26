import modules.scripts as scripts
import random

import torch
import gradio as gr

from scripts.generate_model import Model
from modules import script_callbacks

if torch.cuda.is_available():
    print("** CUDA support : " + str(torch.cuda.is_available()) + " **")
    print("** Version : " + str(torch.version.cuda) + " **")
else:
    print("** CUDA not supported , Shap-E will use CPU.  **")

MAX_SEED = 2147483647
model = None
model_img = None

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column():
                with gr.Tab("TextTo3D"):
                    prompt = gr.Textbox(label="Prompt")
                    with gr.Row():
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=0,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        with gr.Column():
                            guidance_scale = gr.Slider(
                                label="Guidance scale",
                                minimum=1,
                                maximum=20,
                                step=0.1,
                                value=15.0,
                            )
                        with gr.Column():
                            num_inference_steps = gr.Slider(
                                label="Number of inference steps",
                                minimum=2,
                                maximum=200,
                                step=1,
                                value=64,
                            )
                    with gr.Row():
                        generate_button = gr.Button("Generate")
                with gr.Tab("ImgTo3D"):
                    image = gr.Image(label="Input image", show_label=False, type="pil")
                    with gr.Row():
                        seed_img = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=0,
                        )
                        randomize_seed_img = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        with gr.Column():
                            guidance_scale_img = gr.Slider(
                                label="Guidance scale",
                                minimum=1,
                                maximum=20,
                                step=0.1,
                                value=5.0,
                            )
                        with gr.Column():
                            num_inference_steps_img = gr.Slider(
                                label="Number of inference steps",
                                minimum=2,
                                maximum=200,
                                step=1,
                                value=64,
                            )
                    with gr.Row():
                        generate_button_img = gr.Button("Generate")
            with gr.Column():
                with gr.Tab("Output"):
                    with gr.Row():
                        t3d_output = gr.Model3D(show_download_button=False)
            generate_button.click(randomize_seed_fn, inputs=[seed, randomize_seed], outputs=seed, queue=False, api_name=False).then(run_text, inputs=[prompt, seed, guidance_scale, num_inference_steps ], outputs=t3d_output)
            generate_button_img.click(randomize_seed_fn, inputs=[seed_img, randomize_seed_img], outputs=seed_img, queue=False, api_name=False).then(run_image, inputs=[image, seed_img, guidance_scale_img, num_inference_steps_img ], outputs=t3d_output)
        return [(ui_component, "Shap-E", "shap_e_ui_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def run_text(prompt: str, seed: int = 0, guidance_scale: float = 15.0, num_steps: int = 64) -> str:
    global model
    if model is None:
        model = Model(True)
    return model.run_text(prompt, seed, guidance_scale, num_steps)

def run_image(image, seed: int = 0, guidance_scale: float = 3.0, num_steps: int = 64) -> str:
    global model_img
    if model_img is None:
        model_img = Model(False)
    return model_img.run_image(image, seed, guidance_scale, num_steps)

class Script(scripts.Script):
    def __init__(self) -> None:
        super().__init__()

    def title(self):
        return "Shap-E"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return ()
        