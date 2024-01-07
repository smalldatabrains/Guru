import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import os
import random
torch.cuda.empty_cache()

#check if cuda is well installed on the local machine
print(torch.cuda.is_available())

model_id='./model/ghibli_style_offset.safetensors'

#load stable diffusion model
#pipe = StableDiffusionPipeline.from_single_file(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_single_file('./model/ghibli_style_offset.safetensors',torch_dtype=torch.float16,safety_checker=None)
pipe = pipe.to("cuda")

images=[]
background=[]
style=[]
appearance=[]
position=[]
camera=[]
init_image=None

for i in range(100):
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    prompt="ghibli style, freeze corleone, central cee, with dark hoodie and cap "
    print(prompt)
    image=pipe(
        prompt=prompt,  
        negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",  
        image=init_image,
        width=800,  
        height=600,
        num_inference_steps=160 
               ).images[0]
    torch.cuda.empty_cache()
    image.save(f'./results/picture_{i}.jpg')
    images.append(image)
    
