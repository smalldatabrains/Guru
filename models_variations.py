import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import os
import random
torch.cuda.empty_cache()

#check if cuda is well installed on the local machine
print(torch.cuda.is_available())

model_id='./model/dreamlike-photoreal-2.0.ckpt'

#load stable diffusion model
#pipe = StableDiffusionPipeline.from_single_file(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_single_file('./model/realisticVisionV60B1_v60B1VAE.safetensors',torch_dtype=torch.float16,safety_checker=None)
pipe = pipe.to("cuda")

images=[]
background=['bedroom','nature','stadium', 'office']
style=['portrait', 'close portrait', 'full body','suggestive position']
position=['standing','sitting','laying on bed','doggystyle']
camera=[]

for i in range(100):
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    prompt="sexy naked young woman,"+ random.choice(position) +",long hair, bronzed skin,"+ random.choice(style) +", lingerie,"+ random.choice(background) +" background, 8k uhd, high quality, dramatic, cinematic"
    print(prompt)
    image=pipe(
        prompt=prompt,  
        negative_prompt="(deformed iris, deformed face,deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers",  
        width=600,  
        height=800,
        num_inference_steps=140  
               ).images[0]
    torch.cuda.empty_cache()
    image.save(f'./results/sexyc/picture_{i}.jpg')
    images.append(image)
    