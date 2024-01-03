import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionXLPipeline
import os
torch.cuda.empty_cache()

#check if cuda is well installed on the local machine
print(torch.cuda.is_available())

model_id='./model/dreamlike-photoreal-2.0.ckpt'

#load stable diffusion model
#pipe = StableDiffusionPipeline.from_single_file(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_single_file('./model/juggernautXL_v7Rundiffusion.safetensors')
pipe = pipe.to("cuda")

prompts = ["Alice in Wonderland, Ultra HD, realistic, futuristic, detailed, octane render, photoshopped, photorealistic, soft, pastel, Aesthetic, Magical background",
          "Anime style Alice in Wonderland, 90's vintage style, digital art, ultra HD, 8k, photoshopped, sharp focus, surrealism, akira style, detailed line art",
          "Beautiful, abstract art of Chesire cat of Alice in wonderland, 3D, highly detailed, 8K, aesthetic"]

images=[]

for i, prompt in enumerate(prompts):
    image=pipe(
        prompt,
        height=160,
        width=80
               ).images[0]
    image.save(f'picture_{i}.jpg')
    images.append(image)
    torch.cuda.empty_cache()