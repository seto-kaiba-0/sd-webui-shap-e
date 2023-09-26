import os, datetime
from pathlib import Path


import numpy as np
import PIL.Image
import torch
import trimesh
from diffusers import ShapEImg2ImgPipeline, ShapEPipeline
from diffusers.utils import export_to_ply

ROOT_DIR = str(Path().absolute())
output_folder = ROOT_DIR + "\outputs\shap-e"

class Model:
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model: # Disable multi load if not used
            #self.pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, cache_dir=ROOT_DIR + "/.cache",)
            self.pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16)
            self.pipe.to(self.device)
        else:
            #self.pipe_img = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16, cache_dir=ROOT_DIR + "/.cache",)
            self.pipe_img = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16)
            self.pipe_img.to(self.device)
        
    def run_text(self, prompt: str, seed: int = 0, guidance_scale: float = 15.0, num_steps: int = 64) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        return self.save_to_file_n(images)
        
    def run_image(self, image: PIL.Image.Image, seed: int = 0, guidance_scale: float = 3.0, num_steps: int = 64) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe_img(
            image,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        return self.save_to_file_n(images)
        

    def save_to_file_n(self, images):
        file = str(self.output_file_n("shap-e"))
        ply_path = file + ".ply"
        export_to_ply(images[0], ply_path)
        mesh = trimesh.load(ply_path)
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        mesh = mesh.apply_transform(rot)
        mesh_path = file + ".glb"
        mesh.export(mesh_path, file_type="glb")
        
        return mesh_path


    def output_file_n(self, file):
        if not os.path.exists(output_folder):
           os.makedirs(output_folder)
        curday = str(datetime.datetime.now().date()) + '-' + str(datetime.datetime.now().time()).replace(':', '').replace('.', '')
        return output_folder + "\\" + file + "_" + str(curday)

    
    
