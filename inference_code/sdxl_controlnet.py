import cv2
import os
import json
import sys
import warnings
import numpy as np
import torch
import traceback
from diffusers import (ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipeline)
from PIL import Image
from tqdm import tqdm
warnings.filterwarnings("ignore")


def image_generation(prompt, path, pipe, condition_scale, index, generator=None):
    control_images = []
    filename = prompt[:100]
    depth_filename = path + "/rendering/depth/depth0000001.png"
    original_filename = path + "/rendering/frame000.png"
    depth_image = Image.open(depth_filename).convert("RGB")
    ori_image = Image.open(original_filename).convert("RGB")
    canny_image = cv2.Canny(np.array(ori_image), 50, 150, apertureSize=5)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    control_images.append(canny_image)
    canny_image.save(path + '/' + filename + "_canny.png")
    control_images.append(depth_image)
    depth_image.save(path + '/' + filename + "_depth.png")

    negative_prompt = "sketches, blurry, false, low quality, bad quality"
    print("prompt: ", prompt, '\t', condition_scale)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=50, image=control_images, controlnet_conditioning_scale=[condition_scale, condition_scale]).images[0]
    saving_path = f"{path}/scale_{condition_scale}.png"
    print(f"generated image saved to {saving_path}")
    image.save(saving_path)

# Load Multiple ControlNets
controlnet_depth = ControlNetModel.from_pretrained("../model/SDXL-ControlNet-Zoe-Depth", torch_dtype=torch.float16).to('cuda')
controlnet_canny = ControlNetModel.from_pretrained("../model/SDXL-ControlNet-Mistoline", torch_dtype=torch.float16, variant='fp16').to('cuda')
vae = AutoencoderKL.from_pretrained("../model/SDXL-Vae", torch_dtype=torch.float16).to('cuda')
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "../model/SDXL-Base",
    # controlnet=controlnet_canny,
    # controlnet=controlnet_depth,
    controlnet=[controlnet_canny, controlnet_depth],
    vae=vae,
    torch_dtype=torch.float16,
).to('cuda')
pipe.enable_xformers_memory_efficient_attention()


# Error Logging
error_file = f'../output/error_line_{sys.argv[2]}.txt'
# Read Prompts
with open(sys.argv[1], 'r') as file:
    all_lines = file.readlines()
all_lines = [line.strip() for line in all_lines]

# Condition Scale
# paras = [0.05, 0.15, 0.25, 0.35, 0.45]
paras = [0.35]

with open(error_file, "a") as ef:
    for index, text in enumerate(tqdm(all_lines, total=len(all_lines), desc='Progress'), start=1):
        try:
            prompt = text
            line = text[:100]
            print(index, line)
            path = f"../output/obj_output_{sys.argv[2]}/{index}_{line}"
            if not os.path.exists(f'{path}/entity_info.json') or not os.path.exists(f'{path}/default_orientation_info.json'):
                print('=' * 100)
                continue
            
            for condition_scale in paras:
                image_generation(prompt, path, pipe, condition_scale, index)

        # Debug
        except Exception as e:
            error_msg = traceback.format_exc()
            ef.write(f"depth_canny_infer.py\n User Input {index}: {prompt}\nError: {error_msg}\n\n")

