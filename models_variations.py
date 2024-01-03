import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
import os
torch.cuda.empty_cache()

#check if cuda is well installed on the local machine
print(torch.cuda.is_available())

model_id='./model/dreamlike-photoreal-2.0.ckpt'

#load stable diffusion model
#pipe = StableDiffusionPipeline.from_single_file(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_single_file('./model/realisticVisionV60B1_v60B1VAE.safetensors',safety_checker=None)
pipe = pipe.to("cuda")
prompts = ["Alice in Wonderland, Ultra HD, realistic, futuristic, detailed, octane render, photoshopped, photorealistic, soft, pastel, Aesthetic, Magical background",
          "Anime style Alice in Wonderland, 90's vintage style, digital art, ultra HD, 8k, photoshopped, sharp focus, surrealism, akira style, detailed line art",
          "Beautiful, abstract art of Chesire cat of Alice in wonderland, 3D, highly detailed, 8K, aesthetic"]

images=[]

for i in range(100):
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    image=pipe(
        prompt="futuristic 40 yo man with beard and moustache, portrait, sunglasses, colorful,night city background, 8k uhd, high quality, dramatic, cinematic",  
        negative_prompt="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers",  
        width=600,  
        height=800,
        num_inference_steps=120  
               ).images[0]
    torch.cuda.empty_cache()
    image.save(f'./results/guru/picture_{i}.jpg')
    images.append(image)
    