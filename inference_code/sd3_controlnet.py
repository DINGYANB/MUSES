import cv2
import os
import json
import sys
import warnings
import numpy as np
import torch
import traceback
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
from PIL import Image
from tqdm import tqdm
warnings.filterwarnings("ignore")


def image_generation(prompt, ppath, output_folder, pipe, condition_scale, index, generator=None):
    filename = prompt[:100]
    original_filename = ppath + "/rendering/frame000.png"
    ori_image = Image.open(original_filename).convert("RGB")
    canny_image = cv2.Canny(np.array(ori_image), 50, 150, apertureSize=5)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    canny_image.save(ppath + '/' + filename + "_canny.png")
    
    negative_prompt = "sketches, blurry, false, low quality, bad quality"
    print("prompt: ", prompt, '\t', condition_scale)
    image = pipe(prompt, negative_prompt=negative_prompt, control_image=canny_image, controlnet_conditioning_scale=condition_scale, num_inference_steps=20).images[0]
    saving_path = output_folder + '/' + filename +  '_0.png'
    print(saving_path)
    image.save(saving_path)



# Load Multiple ControlNets
controlnet_canny = SD3ControlNetModel.from_pretrained("../model/SD3-ControlNet-Canny", torch_dtype=torch.float16).to('cuda')
# controlnet_tile = SD3ControlNetModel.from_pretrained("../model/SD3-ControlNet-Tile")
# pipe = StableDiffusion3ControlNetPipeline.from_pretrained("../model/SD3-Base", controlnet=controlnet_tile)
pipe = StableDiffusion3ControlNetPipeline.from_pretrained("../model/SD3-Base", controlnet=controlnet_canny, torch_dtype=torch.float16).to('cuda')

# Error Logging
error_file = f'../output/error_line_{sys.argv[2]}.txt'
# Read Prompts
with open(sys.argv[1], 'r') as file:
    all_lines = file.readlines()
all_lines = [line.strip() for line in all_lines]

# Condition Scale
paras = [0.5, 0.6, 0.7, 0.8, 0.9]
os.makedirs(f"../output/{sys.argv[2]}", exist_ok=True)
for condition_scale in paras:
    output_folder = f"../output/{sys.argv[2]}/{str(condition_scale)}"
    os.makedirs(output_folder, exist_ok=True)


with open(error_file, "a") as ef:
    for index, text in enumerate(tqdm(all_lines, total=len(all_lines), desc='Progress'), start=1):
        try:
            prompt = text
            line = text[:100]
            print(index, line)
            path = "../output/obj_output_" + sys.argv[2] + "/" + str(index) + "_" + line
            if not os.path.exists(f'{path}/entity_info.json') or not os.path.exists(f'{path}/default_orientation_info.json'):
                print('-' * 50)
                continue

            for condition_scale in paras:
                output_folder = f"../output/{sys.argv[2]}/{str(condition_scale)}"
                image_generation(prompt, path, output_folder, pipe, condition_scale, index)

        # Debug
        except Exception as e:
            error_msg = traceback.format_exc()
            ef.write(f"depth_canny_infer.py\n User Input {index}: {line}\nError: {error_msg}\n\n")

