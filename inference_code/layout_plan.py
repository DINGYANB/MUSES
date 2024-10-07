import ast
import os
import json
import re
import clip
import traceback
import torch
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import shutil
from transformers import CLIPTokenizer, CLIPModel
from llama import Dialog, Llama
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(prog='MUSES', description='Use Llama 3 to generate 3D image scene layouts.')
parser.add_argument('--user_input', type=str, default='')
parser.add_argument('--text_file', type=str, default='test.txt')
parser.add_argument('--append', type=str, default='no_append_info')
args = parser.parse_args()


def reorder_list(key_order, lst):
    key_index = {key: idx for idx, key in enumerate(key_order)}
    print("key_index", key_index)
    sorted_lst = sorted(lst, key=lambda x: key_index.get(x[0], float('inf')))
    
    return sorted_lst


def draw_bounding_box(bboxes, texts):
    # Create a white background image
    image = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    # Draw bounding boxes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 128, 128), (0, 0, 0)]
    for i, (bbox, text) in enumerate(zip(bboxes, texts)):
        x = int(bbox[0] * 1024)
        y = int(bbox[1] * 1024)
        width = int(bbox[2] * 1024)
        height = int(bbox[3] * 1024)
        color = colors[i % len(colors)]
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 4)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x + (width - text_size[0]) // 2
        text_y = y + 20  # Place the text slightly below the bounding box
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness)

    return image


def create_3d_depth_examplar_prompt(json_path):
    prompt = ""
    depth_examples = json.load(open(json_path))
    for dic in depth_examples:
        prompt += 'User Input: ' + dic['User Input'] + '\n' + 'Layout Information: ' + str(dic['Layout Information']) + '\n' +  'Output: ' + str(dic['Output']) + '\n\n'
    return prompt


def create_3d_orientation_examplar_prompt(json_path):
    prompt = ""
    orientation_examples = json.load(open(json_path))
    for dic in orientation_examples:
        prompt += 'User Input: ' + dic['User Input'] + '\n' + 'Layout Information: ' + str(dic['Layout Information']) + '\n' +  'Output: ' + str(dic['Output']) + '\n\n'
    return prompt


def create_3d_camera_view_examplar_prompt(json_path):
    prompt = ""
    camera_examples = json.load(open(json_path))
    for dic in camera_examples:
        prompt += 'User Input: ' + dic['User Input'] + '\n' +  'Output: ' + str(dic['Output']) + '\n\n'
    return prompt


def create_exemplar_prompt(caption, object_list, canvas_size, is_chat=False):
    if is_chat:
        prompt = ''
    else:
        prompt = f'\nPrompt: {caption}\nLayout:\n'

    for obj_info in object_list:
        category, bbox = obj_info
        coord_list = [int(i*canvas_size) for i in bbox]
        x, y, w, h = coord_list
        prompt += f'{category} {{height: {h}px; width: {w}px; top: {y}px; left: {x}px; }}\n'
    return prompt


def f_form_prompt(text_input, current_feature, top_k, examples, features):
    rtn_prompt = 'Instruction: You are a master of image layout planning. Your task is to plan the realistic layout of the image according to the given prompt. ' \
                 'The generated layout should follow the CSS style, where each line starts with the object description and is followed by its absolute position. ' \
                 'Formally, each line should be like "object {{width: ?px; height: ?px; left: ?px; top: ?px; }}", with each object extracted from the given prompt. ' \
                 'The image is 256px wide and 256px high. Therefore, all properties of the positions should not exceed 256px, including the addition of left and width and the addition of top and height. ' \
                 'Some examples are given below.\n'
    last_example = f'\nPrompt: {text_input}'
    prompting_examples = ''

    # Find most related supporting examples
    similarities = (100.0 * current_feature @ features.T).softmax(dim=-1)
    _, indices = similarities[0].topk(100)
    supporting_examples = [examples[idx] for idx in indices]

    count = 0
    # Loop through the related supporting examples
    for i in range(len(supporting_examples)):
        if i > 0 and supporting_examples[i]['prompt'] == supporting_examples[i-1]['prompt']:
            continue
        if "object_list" in supporting_examples[i]:
            current_prompting_example = create_exemplar_prompt(
                caption=supporting_examples[i]['prompt'],
                object_list=supporting_examples[i]['object_list'],
                canvas_size=256,
            )
        else:
            current_prompting_example = create_exemplar_prompt(
                caption=supporting_examples[i]['prompt'],
                object_list=[supporting_examples[i]['obj1'], supporting_examples[i]['obj2']],
                canvas_size=256,
            )
        prompting_examples = current_prompting_example + prompting_examples
        count += 1
        if count == top_k:
            break
    
    # concatename prompts
    prompting_examples += "\nNow, it's your turn." + last_example
    rtn_prompt += prompting_examples
    
    return rtn_prompt


def llm_plan(model, user_input, layout_plan_prompt, index):
    # 111 --- 2D layout plan
    print(layout_plan_prompt)
    dialogs: List[Dialog] = [[{"role": "user", "content": layout_plan_prompt}]]
    results = model.chat_completion(
        dialogs,
        max_gen_len=None,
        temperature=0.2,
        top_p=0.1,
    )
    response = results[0]['generation']['content']
    print('11111-2D Layout:', results[0]['generation']['content'])

    # Parse Output
    predicted_object_list = []
    line_list = response.split('\n')
    bboxes = []
    texts = []
    for line in line_list:
        if line == '':
            continue
        # print('line: ', line)
        css_string = line
        main_key = re.match(r'([^{]+)\{', css_string)
        if not main_key:
            continue
        main_key = (main_key.group(1))[:-1]  
        texts.append(main_key)
        css_properties_string = re.search(r'\{([^}]+)\}', css_string).group(1)

        css_properties = {}
        nested_properties = {}
        for prop in css_properties_string.split(';'):
            if prop.strip():
                key, value = prop.split(':')
                nested_properties[key.strip()] = float(int(value.strip().replace('px', ''))) / 256
                # print(nested_properties)

        # 2D layout check and adjustment
        nested_properties['left'] = max(0, nested_properties['left'])
        nested_properties['top'] = max(0, nested_properties['top'])
        if (nested_properties['left'] + nested_properties['width']) >= 1.0:
            if nested_properties['width'] >= 0.9:
                nested_properties['width'] /= 1.5
            nested_properties['left'] = 1.0 - nested_properties['width']
        if (nested_properties['top'] + nested_properties['height']) >= 1.0:
            if nested_properties['height'] >= 0.9:
                nested_properties['height'] /= 1.5
            nested_properties['top'] = 1.0 - nested_properties['height']

        css_properties[main_key] = nested_properties
        bbox = [nested_properties['left'], nested_properties['top'], nested_properties['width'], nested_properties['height']]
        bboxes.append(bbox)
        # print('css_properties: ', css_properties)
        predicted_object_list.append(css_properties)
    obj_list = [key for obj in predicted_object_list for key in obj.keys()]

    # Saving 2D layout
    result_image = draw_bounding_box(bboxes, texts)
    saving_directory = "../output/obj_output_" + args.append + "/" + str(index) + "_" + user_input[:100]
    if os.path.exists(saving_directory):
        shutil.rmtree(saving_directory)
    os.makedirs(saving_directory, exist_ok=True)
    saving_path = os.path.join(saving_directory, "bbox.png")
    print("Saving Path:", saving_path)
    success = cv2.imwrite(saving_path, result_image)
    if success:
        print("Image saved successfully.", '\n' + '-' * 30)
    else:
        print("Failed to save image.", '\n' + '-' * 30)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # 222 --- 3D layout plan
    depth_prompt = 'User Input: {}\nLayout Information: {}\n\n' \
                   'Your task is to determine the depth value (greater than 0 and less than 1) ' \
                   'for each object contained in the layout information based on the user input. ' \
                   'Actually, depth value represents the distance of the object from the viewer. First, carefully look for the words ' \
                   'among "back", "behind", "front" and "hidden" in the user input. If not found, directly set all depth values to "0.0".' \
                   'If find one, go to the second step with the following setting rules:' \
                   '\n- If "Object1 in front of Object2", the depth value of Object1 is smaller and can be "0.1" and the depth value of Object2 can be "0.9".' \
                   '\n- If "Object1 behind (hidden by) Object2", the depth value of Object1 is bigger and can be "0.9" and the depth value of Object2 can be "0.1".' \
                   '\n- If "Object in the back", the depth value of Object is big and can be "0.9".' \
                   '\n- If "Object in the front", the depth value of Object is small and can be "0.1".' \
                   '\n\nYour final answer must be a nested list as [[object name, depth]], where object name is the same as layout information. Follow the above two steps and give some explanation. Do not include any code.'.format(user_input, predicted_object_list)
    orientation_prompt = 'User Input: {}\nLayout Information: {}\n\n' \
                         'Your task is to determine the orientation value for each object contained in the layout information based on the user input. ' \
                         'First, carefully look for the words among "facing", "toward" and "heading" in the user input. If not found, ' \
                         'directly set all orientation values to "forward". If find one, go to the second step with the following setting rule:' \
                         '\n- Extract the directions among "forward", "backward", "left", "right", "upward" or "downward" from the user input as the orientation values.' \
                         '\n\nYour final answer must be a nested list as [[object name, orientation]], where object name is the same as layout information. Follow the above two steps and give some explanation. Do not include any code.'.format(user_input, predicted_object_list)
    camera_prompt = 'User Input: {}\n\n' \
                    'Your task is to extract the camera view from the user input. First, carefully look for the word "view" in the user input. If not found, ' \
                    'directly set the camera view to "front view". If find one, go to the second step with the following setting rule:' \
                    '\n- Extract strings among "front view", "left view", "right view" or "top view" from the user input as the camera view.' \
                    '\n\nYour final answer must be in JSON format, where key is "camera view" and value is one of the strings "front view", "left view", "right view" or "top view". Follow the above two steps and give some explanation. Do not include any code.'.format(user_input)
    print(depth_prompt, '\n', orientation_prompt, '\n', camera_prompt)

    dialogs: List[Dialog] = [[{"role": "user", "content": depth_prompt}], [{"role": "user", "content": orientation_prompt}], [{"role": "user", "content": camera_prompt}]]
    results = model.chat_completion(dialogs, temperature=0.2, top_p=0.1)
    depth_info, orientation_info, camera_info = results[0]['generation']['content'], results[1]['generation']['content'], results[2]['generation']['content']
    print(depth_info, '\n' + '-' * 30 + '\n', orientation_info, '\n' + '-' * 30 + '\n', camera_info, '\n' + '-' * 30 + '\n')
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    json_pattern_1 = re.compile(r"\[\[.*?\]\]", re.DOTALL)      # For Depth/Orientation
    json_pattern_2 = re.compile(r'\{.*?\}', re.DOTALL)          # For Camera
    json_mathes_depth = json_pattern_1.findall(depth_info)
    json_mathes_orientation = json_pattern_1.findall(orientation_info)
    json_mathes_camera = json_pattern_2.findall(camera_info)
    depth_values = []
    if json_mathes_depth:
        json_str = json_mathes_depth[-1].strip().replace('"', "'")
        try:
            json_str = re.sub(r'\[(\w+),\s*([\d.]+)\]', r"['\1', \2]", json_str)
            depth_values = ast.literal_eval(json_str)
        except:
            for obj in predicted_object_list:
                for key in obj.keys():
                    depth_values.append([key, 0.0])
    else:
        for obj in predicted_object_list:
            for key in obj.keys():
                depth_values.append([key, 0.0])
    print(obj_list)
    # print(depth_values)
    # depth_values = reorder_list(obj_list, depth_values)
    print(depth_values)
    orientation_values = []
    if json_mathes_orientation:
        json_str = json_mathes_orientation[-1].strip().replace('"', "'")
        try:
            json_str = re.sub(r'(?<![\w\'\"])\b(\w+)\b(?![\w\'\"])', r"'\1'", json_str)
            orientation_values = ast.literal_eval(json_str)
        except:
            for obj in predicted_object_list:
                for key in obj.keys():
                    orientation_values.append([key, "forward"])
    else:
        for obj in predicted_object_list:
            for key in obj.keys():
                orientation_values.append([key, "forward"])
    # print(orientation_values)
    # orientation_values = reorder_list(obj_list, orientation_values)
    print(orientation_values)
    camera_value = {"camera view": "front view"}
    if json_mathes_camera:
        json_str = json_mathes_camera[-1].strip().replace("'", '"')
        try:
            camera_value = json.loads(json_str) 
        except:
            camera_value = {"camera view": "front view"}
    else:
        camera_value = {"camera view": "front view"}
    print(camera_value)

    for dic in predicted_object_list:
        for key, value in dic.items():
            try:
                for ii in range(len(depth_values)):
                    if key.lower() == depth_values[ii][0].lower():
                        value["depth"] = depth_values[ii][1]
                        depth_values.pop(ii)
                        break
                for jj in range(len(orientation_values)):
                    if key.lower() == orientation_values[jj][0].lower():
                        value["orientation"] = orientation_values[jj][1]
                        orientation_values.pop(jj)
                        break
            except:
                value["depth"] = 0.0
                value["orientation"] = "forward"
            value["camera"] = camera_value["camera view"]

    # Saving 3D layout
    ppath = os.path.join(saving_directory, "entity_info.json")
    with open(ppath, 'w') as file:
        json.dump(predicted_object_list, file)
    print(f'results written to {ppath}')


def _main(args):
    # Load CLIP
    clip_model = CLIPModel.from_pretrained("../model/CLIP")
    tokenizer = CLIPTokenizer.from_pretrained("../model/CLIP")
    clip_model = clip_model.to('cuda')

    # Load Features
    loaded_examples_features = np.load('../dataset/examples_features.npy')
    loaded_examples_features = torch.tensor(loaded_examples_features)
    loaded_examples_features = loaded_examples_features.to('cuda')
    print("loaded examples features: ", loaded_examples_features.shape, loaded_examples_features.dtype)

    # Load 2D Layout Examples
    train_example_files_counting = "../dataset/NSR-1K-Expand/counting/counting.json"
    train_example_files_spatial = "../dataset/NSR-1K-Expand/spatial/spatial.json"
    train_example_files_3d_spatial = "../dataset/NSR-1K-Expand/3d_spatial/3d_spatial.json"
    train_example_files_complex = "../dataset/NSR-1K-Expand/complex/complex.json"
    train_examples_counting = json.load(open(train_example_files_counting))
    train_examples_spatial = json.load(open(train_example_files_spatial))
    train_examples_3d_spatial = json.load(open(train_example_files_3d_spatial))
    train_examples_complex = json.load(open(train_example_files_complex))
    supporting_examples = train_examples_counting + train_examples_spatial + train_examples_3d_spatial + train_examples_complex

    # print(type(train_examples_counting), len(train_examples_counting) , '*', len(train_examples_counting[0]))
    # print(type(train_examples_spatial), len(train_examples_spatial), '*', len(train_examples_spatial[0]))
    # print(type(train_examples_3d_spatial), len(train_examples_3d_spatial), '*', len(train_examples_3d_spatial[0]))
    # print(type(train_examples_complex), len(train_examples_complex), '*', len(train_examples_complex[0]))
    # print(type(supporting_examples), len(supporting_examples), '*', len(supporting_examples[0]))
    # exit(0)

    # compute features
    # features_list = []
    # example_prompts = [example['prompt'] for example in supporting_examples]
    # for prompt in tqdm(example_prompts, total=len(example_prompts), desc='Progress'):
    #     examples_input = tokenizer(prompt, padding=True, return_tensors="pt")
    #     examples_input = {key: value for key, value in examples_input.items()}
    #     feature = clip_model.get_text_features(**examples_input)
    #     feature /= feature.norm(dim=-1, keepdim=True) 
    #     features_list.append(feature.cpu().detach().numpy())
    # examples_input = tokenizer(example_prompts, padding=True, return_tensors="pt")
    # print(examples_input['input_ids'].shape, examples_input['attention_mask'].shape)  # [n, 61]

    # examples_features = np.stack(features_list, axis=0)
    # examples_features = np.squeeze(examples_features, axis=1)
    # print(examples_features.shape, examples_features.dtype)
    # np.save('examples_features.npy', examples_features)
    # exit(0)

    # Debug
    error_file = f'../output/error_line_{args.append}.txt'
    if os.path.exists(error_file):
        os.remove(error_file)
    else:
        open(error_file, 'w').close() 

    # Read user's prompt
    if (args.user_input != ''):
        all_lines = [args.user_input]
    else:
        with open(args.text_file, 'r') as file:
            all_lines = file.readlines()

    all_lines = [line.strip() for line in all_lines]
    input_features = []
    for line in all_lines:
        text_input = tokenizer(line, padding=True, return_tensors="pt")
        text_input = {key: value.to('cuda') for key, value in text_input.items()}
        text_feature = clip_model.get_text_features(**text_input)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        input_features.append(text_feature)

    # Build Llama3
    model = Llama.build(
        ckpt_dir='../model/Llama3',
        tokenizer_path='../model/Llama3/tokenizer.model',
        max_seq_len=8192,
        max_batch_size=3,
    )

    with open(error_file, "a") as ef:
        for index, text in enumerate(tqdm(all_lines, total=len(all_lines), desc='Progress'), start=1):
            try:
                prompt_for_layout = f_form_prompt(text, input_features[index-1], 5, supporting_examples, loaded_examples_features)
                llm_plan(model, text, prompt_for_layout, index)
                print('\n' + '-'*30 + '\n')
            except Exception as e:
                error_msg = traceback.format_exc()
                ef.write(f"layout_plan.py\n User Input {index}: {text}\nError: {error_msg}\n\n")

   
if __name__ == '__main__':
    _main(args)
