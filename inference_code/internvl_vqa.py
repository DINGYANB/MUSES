import os
import torch
from transformers.generation import GenerationConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import warnings
import re
import sys
import argparse
from torchvision.transforms.functional import InterpolationMode
warnings.filterwarnings("ignore")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
parser = argparse.ArgumentParser(description='Process some images and a text file.')
parser.add_argument('--img_dir', type=str, required=True, help='Path for the images.')
parser.add_argument('--text_file', type=str, required=True, help='Path to the text file.')
args = parser.parse_args()
img_dir = args.img_dir
text_file = args.text_file

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def extract_number(s):
    return int(s.split('_')[0])


# Read Images
condition_scale = [0.5, 0.6, 0.7, 0.8, 0.9]
img_dirs = [img_dir + "_" + str(cs) for cs in condition_scale]
print(img_dirs)
image_files = [sorted(os.listdir(img_dir), key=extract_number) for img_dir in img_dirs]

# Read Text Prompts
with open(text_file, 'r') as file:
    texts = file.readlines()
texts = [line.strip() for line in texts]

similarity_results = []

# Loac Checkpoint
path = "../model/InternVL"
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)


def extract_score_or_number(s):
    # Attempt to find the specific pattern '("score", '
    pattern = '("score", '
    start = s.find(pattern)
    if start != -1:
        start += len(pattern)
        end = start
        # Extract the number after the pattern, including the decimal part
        while end < len(s) and (s[end].isdigit() or s[end] == '.'):
            end += 1
        try:
            return float(s[start:end])
        except:
            return 0.0
    else:
        # If the specific pattern is not found, use regex to find any number in the string
        match = re.search(r'\d+\.\d+', s)
        return float(match.group(0)) if match else None


# Start Evaluation
for i in range(len(image_files[0])):
    max_score = float(0.0)
    for j in range(5):
        image = os.path.join(img_dirs[j], image_files[j][i])
        # print(image)
        pixel_values = load_image(image, max_num=6).to(torch.bfloat16).cuda()
        image_index = image.split('/')[-1].split('_')[0]
        print(image_index)
        text = texts[int(image_index)]
        # question = '''Text: {}\nHow well does the image match the text? You need to consider (1) object count, (2) object orientation, (3) 3D spatial relationship between objects, (4) camera view. Return a tuple ("score", X.XXXX), with the float number between 0 and 1, and higher scores representing higher text-image alignment.'''.format(text)
        question = '''Text: {}\nHow well does the image match the text? You need to consider camera view. Return a tuple ("score", X.XXXX), with the float number between 0 and 1, and higher scores representing higher text-image alignment.'''.format(text)
        response = model.chat(tokenizer, pixel_values, question, generation_config)

        score = extract_score_or_number(response)
        if score == None:
            continue
        max_score = max(float(score), max_score)
        print(f"'{text}\n': {max_score}")
        if max_score >= 0.99999:
            break
    similarity_results.append((text, max_score))


print('-' * 300, '\n')
total_sim = 0
for i, result in enumerate(similarity_results):
    total_sim += result[1]
    print(f"{i} - Text: {result[0]} - Similarity: {result[1]}")
print(len(similarity_results), len(image_files[0]))
print(total_sim / len(image_files[0]))