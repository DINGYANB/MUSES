import bpy
import requests
import clip
import json
import rarfile
import zipfile
import os
import re
import warnings
import torch
import shutil
import random
import sys
import math
import mathutils
import traceback
import warnings
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_config, load_model
from shap_e.util.data_util import load_or_create_multimodal_batch
from shap_e.util.notebooks import decode_latent_images, decode_latent_mesh
warnings.filterwarnings("ignore")


def add_ground_below_object(obj, ground_thickness):
    bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    min_z = min([v.z for v in bbox])

    # Add a ground under the 3D model
    bpy.ops.mesh.primitive_plane_add(size=4, enter_editmode=False, align='WORLD', location=(0, 0, min_z - ground_thickness / 2))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.scale = (10, 10, ground_thickness)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ground_mat = bpy.data.materials.new(name="Ground_Material")
    ground_mat.diffuse_color = (0.8, 0.8, 1, 1)
    ground.data.materials.append(ground_mat)


def set_obj_color(obj, color):
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="Obj_Material")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]
    
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf is None:
        bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = color


def scale_and_center_obj(obj, target_size):
    obj_dimensions = obj.dimensions
    average_dimension = max(obj_dimensions)
    scale_factor = target_size / average_dimension
    obj.scale = (scale_factor,) * 3
    
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
    obj.location = (0, 0, 0)
    bpy.context.view_layer.update()


def multi_view_render(obj, rotation_angle, log_file, directory, prompt, i, model, tokenizer):
    open(log_file, 'a').close()
    old = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.close(sys.stdout.fileno())
    fd = os.open(log_file, os.O_WRONLY)

    obj.rotation_euler = rotation_angle
    output_image_path = os.path.join(directory, f"{prompt}_{i}.png")
    bpy.context.scene.render.filepath = output_image_path
    bpy.context.view_layer.update()
    ground_thickness = 0.01
    add_ground_below_object(obj, ground_thickness)

    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(bpy.data.objects["Ground"], do_unlink=True)

    os.close(fd)
    os.dup(old)
    os.close(old)

    with torch.no_grad():
        text = '''{} faces camera'''.format(add_article(prompt))
        text_input = clip.tokenize(text).to('cuda')
        image_input = tokenizer(Image.open(output_image_path)).unsqueeze(0).to('cuda')
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity_score = 100.0 * image_features @ text_features.T

    del text_input, image_input, image_features, text_features
    torch.cuda.empty_cache()

    # print(float(similarity_score))
    return similarity_score, output_image_path



def remove_trailing_parentheses(text):
    return re.sub(r'\s*\(.*\)\s*$', '', text)


def add_article(word):
    special_cases = {
        'hour': 'an',
        'honest': 'an',
        'honor': 'an',
        'heir': 'an'
    }
    if word.lower() in special_cases:
        return f'{special_cases[word.lower()]} {word}'
    vowel_sound_regex = re.compile(r'^[aeiou]', re.IGNORECASE)
    special_vowel_sound_regex = re.compile(r'^(hour|honest|honor|heir)', re.IGNORECASE)
    if special_vowel_sound_regex.match(word) or vowel_sound_regex.match(word):
        return f'an {word}'
    else:
        return f'a {word}'


def extract_rar(rar_path, extract_to):
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(extract_to)


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


def search_obj_in_zip_or_rar(main_directory, search_directory, search_info, file_name):
    flag = False
    for root, _, files in os.walk(search_directory):
        for file in files:
            if file.lower().endswith(".obj"):
                new_file_path = os.path.join(main_directory, file_name)
                shutil.move(os.path.join(root, file), new_file_path)
                print(f'Download an obj file in the compressed package successfully. ')
                flag = True
                break
    if not flag:
        print(f'Failed to find an obj file in the compressed package! ')
    shutil.rmtree(search_directory)
    return flag


def search_existing_models_in_folder(search_index, search_main_directory):
    models = os.listdir(search_main_directory)
    model_names = [(os.path.splitext(model)[0]).split('_')[1].lower() for model in models]
    ok_models = []
    for i, name in enumerate(model_names):
        if search_index.lower() == name:
            ok_models.append(i)
    if len(ok_models) > 0:
        ok_models = [os.path.join(search_main_directory, models[index]) for index in ok_models]
    return ok_models


def search_on_website(new_directory, search_index, base_search_url, final_download_name, clip_model):
    pattern = search_index.replace(" ", ".*")
    search_url = base_search_url + search_index.replace(" ", "%20")
    search_response = session.get(search_url)
    if search_response.status_code == 200:
        print(f'Search on "{search_url}" successfully')
    else:
        print("Search online unsuccessfully!")
        return -1

    soup = BeautifulSoup(search_response.text, 'html.parser')
    json_dict = json.loads(str(soup))
    model_url = []
    model_name = []
    if len(json_dict['data']) > 0:
        for value in json_dict['data']:
            name = value['attributes']['title']
            if re.search(pattern, name, re.IGNORECASE):
                url = value['attributes']['url']
                model_url.append(url)
                model_name.append(name)
        # print(model_name)
    else:
        print("No available model!")
        return -1

    if len(model_url) == 0:
        print("No suitable model!")
        return -1
    
    
    search_feature = clip.tokenize("low poly " + search_index + " 3d mesh model")
    similarity = []
    model_info = []
    file_name = [" " for j in range(len(model_url))]
    find = [-1 for j in range(len(model_url))]
    
    for i, (name, url) in enumerate(zip(model_name, model_url)):
        text_inputs = torch.cat([search_feature, clip.tokenize(name.lower().replace('_', ' '))]).to('cuda')
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity.append(text_features[0] @ text_features[1].T)
        if i == 10:
            break
        model_page_response = session.get(url)
        soup = BeautifulSoup(model_page_response.text, 'html.parser')

        model_item = (soup.find_all(class_='actions'))[0]
        if model_item:
            download_link = model_item.find('a')['href']
            full_download_link = "https://www.cgtrader.com" + download_link
            download_page_response = session.get(full_download_link)
            soup = BeautifulSoup(download_page_response.text, 'html.parser')
            info = soup.select('ul.details-box__list li')
            button_names = [li.get_text(strip=True).lower() for li in info]
            # print(button_names)
            found_files = {ext: None for ext in exts}
            model_info.append(info)
            for ii, button in enumerate(button_names):
                for ext in exts:
                    if ext in button:
                        # print(button)
                        end = button.find(ext) + len(ext)
                        filename = button[:end]
                        found_files[ext] = [filename, ii]
                        break
            # print(found_files)
            for ext in found_files:
                if found_files[ext]:
                    file_name[i], find[i] = found_files[ext][0], found_files[ext][1]
                    break
        else:
            continue
    
    similarity = sorted(similarity, reverse=True)
    succeed = False
    for value in similarity:
        if succeed:
            break
        index = similarity.index(value)
        if find[index] >= 0: 
            print('''Retrieved the relevant model "{}({})": "{}"'''.format(model_name[index], file_name[index], model_url[index]))
            a_tag = model_info[index][find[index]].find('a')
            # print(model_info[index], find[index])
            actual_download_link = "https://www.cgtrader.com" + a_tag['href']
            print(f'Try to download it from: "{actual_download_link}"...', end=" ")
            response = requests.head(actual_download_link, allow_redirects=True, headers=headers)
            if response.status_code == 200:
                redirect_link = response.url
                file_response = requests.get(redirect_link)
                if file_response.status_code == 200:
                    # print(file_name[index])
                    if file_name[index][-4:] == ".zip" or file_name[index][-4:] == ".rar":
                        download_name = os.path.join(new_directory, prompt + file_name[index][-4:])
                    else:
                        download_name = os.path.join(new_directory, final_download_name)
                    # print(download_name)
                    with open(download_name, 'wb') as file:
                        file.write(file_response.content)
                    if download_name[-4:] == ".rar":
                        search_directory = f"{new_directory}/rar_" + prompt
                        os.makedirs(search_directory, exist_ok=True)
                        extract_rar(download_name, search_directory)
                        os.remove(download_name)
                        succeed = search_obj_in_zip_or_rar(new_directory, search_directory, prompt, final_download_name)
                    elif download_name[-4:] == ".zip":
                        search_directory = f"{new_directory}/zip_" + prompt
                        os.makedirs(search_directory, exist_ok=True)
                        extract_zip(download_name, search_directory)
                        os.remove(download_name)
                        succeed = search_obj_in_zip_or_rar(new_directory, search_directory, prompt, final_download_name)
                    else:
                        succeed = True
                        print("Download 3d model in obj file format successfully. ")
                else:
                    print(f"Download failed: {response.status_code}!")
                    continue
            else:
                print(f"Download failed: {response.status_code}!")
                continue

    return succeed


def text_to_3d_generation(obj_file_path, prompt):
    # Using opanai shap-e model
    xm = load_model('transmitter', device='cuda')
    text_to_3d_model = load_model('text300M', device='cuda')
    diffusion = diffusion_from_config(load_config('diffusion'))
    latents = sample_latents(
                batch_size=1,
                model=text_to_3d_model,
                diffusion=diffusion,
                guidance_scale=15.0,
                model_kwargs=dict(texts=[add_article(prompt)]),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=64,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
                )

    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    with open(obj_file_path, 'w') as f:
        t.write_obj(f)

    del xm, text_to_3d_model
    torch.cuda.empty_cache()


def face_camera_view_indentify(obj_directory, png_directory, file_path, log_file, prompt, model, tokenizer):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    bpy.ops.wm.obj_import(filepath=file_path)
    # Merge extra components of a single obj file
    if (len(bpy.data.objects) > 3):
        # print(len(bpy.data.objects))
        obs = bpy.context.scene.objects[2:]
        for ob in obs:
            set_obj_color(ob, (0.5, 0.5, 0.7, 1))
        ctx = bpy.context.copy()
        ctx['selected_objects'] = obs
        bpy.ops.object.join()
        # print(len(bpy.data.objects))
    
    obj = bpy.context.selected_objects[0]
    set_obj_color(obj, (0.5, 0.5, 0.7, 1))
    target_size = 2.5
    scale_and_center_obj(obj, target_size)

    rotation_euler = \
    [(0, 0, 0), (0, 0, 90 * math.pi / 180), (0, 0, -90 * math.pi / 180), (0, 0, 180 * math.pi / 180),
     (90 * math.pi / 180, 0, 0), (90 * math.pi / 180, 0, 90 * math.pi / 180), (90 * math.pi / 180, 0, -90 * math.pi / 180), (90 * math.pi / 180, 0, 180 * math.pi / 180),
     (-90 * math.pi / 180, 0, 0), (-90 * math.pi / 180, 0, 90 * math.pi / 180), (-90 * math.pi / 180, 0, -90 * math.pi / 180), (-90 * math.pi / 180, 0, 180 * math.pi / 180)]

    highest_similarity_score = -1
    right_image_path = ""
    torch.cuda.empty_cache()
    for j, current_angle in enumerate(rotation_euler):
        similarity_score, image_path = multi_view_render(obj, current_angle, log_file, png_directory, prompt, j, model, tokenizer)
        if similarity_score > highest_similarity_score:
            highest_similarity_score = similarity_score
            right_angle = current_angle
            if right_image_path != "":
                os.remove(right_image_path)
            right_image_path = image_path
        else:
            os.remove(image_path)

    return right_angle


if __name__ == "__main__":
    # Load CLIP Model
    clip_model_1, preprocess_1 = clip.load("ViT-B/32", device='cuda', jit=False)
    clip_model_2, preprocess_2 = clip.load("ViT-B/32", device='cuda', jit=False)
    clip_model_2.load_state_dict(torch.load("../model/CLIP/finetuned_clip_epoch_20.pth"))
    clip_model_2.eval()

    # Relevant Parameters
    search_main_directory =  "../3d_model_shop/exist_models"
    base_search_url = "https://www.cgtrader.com/search?free=1&file_types[]=12&keywords="
    exts = [".obj.zip", ".obj.rar", "obj.zip", "obj.rar", ".obj"]
    log_file = 'blender.log'
    headers = {
        "Cookie": '''##################'''      # replace it with your cookie
    }
    session = requests.Session()

    # Blender Initialize
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    # Camera, Light
    camera = bpy.data.objects['Camera']
    light = bpy.data.objects['Light']
    camera.location = (0, -6, 2)
    camera.rotation_euler = (75 * math.pi / 180, 0, 0)  
    light.location = (5, -5, 5)

    # Error Logging
    error_file = f'../output/error_line_{sys.argv[2]}.txt'
    with open(sys.argv[1], 'r') as file:
        all_lines = file.readlines()
    all_lines = [line.strip() for line in all_lines]

    # Start!
    with open(error_file, "a") as ef:
        for index, text in enumerate(tqdm(all_lines, total=len(all_lines), desc='Progress'), start=1):
            try:
                text = text[:100]
                if not os.path.exists(f'../output/obj_output_{sys.argv[2]}/{index}_{text}/entity_info.json'):
                    continue
                with open(f'../output/obj_output_{sys.argv[2]}/{index}_{text}/entity_info.json', 'r') as file:
                    data = json.load(file)
                print(f'{index}: {text}')

                prompts = []
                answers = []
                for item_info in data:
                    for key, value in item_info.items():
                        if 'name' in value.keys():
                            prompts.append(value['name'])
                        else:
                            if (key[-1] >= '0' and key[-1] <= '9') or (key[-1] >= 'A' and key[-1] <= 'J'):
                                if key[-2] == ' ':
                                    prompts.append((key.replace('_', ' '))[:-2])
                                elif (key[-2] >= '0' and key[-2] <= '9') or (key[-2] >= 'A' and key[-2] <= 'J'):
                                    if key[-3] == ' ':
                                        prompts.append((key.replace('_', ' '))[:-3])
                                    else:
                                        prompts.append((key.replace('_', ' '))[:-2])
                                else:
                                    prompts.append((key.replace('_', ' '))[:-1])
                            else:
                                prompts.append(remove_trailing_parentheses(key).replace('_', ' '))
                print("Need 3D Models: ", prompts)

                output_obj_path = f'../output/obj_output_{sys.argv[2]}/{index}_{text}/obj'
                output_png_path = f'../output/obj_output_{sys.argv[2]}/{index}_{text}/png'
                if os.path.exists(output_obj_path):
                    shutil.rmtree(output_obj_path)
                if os.path.exists(output_png_path):
                    shutil.rmtree(output_png_path)
                os.makedirs(output_obj_path, exist_ok=True)
                os.makedirs(output_png_path, exist_ok=True)

                for num, prompt in enumerate(prompts):
                    torch.cuda.empty_cache()
                    sss = chr(65+num)
                    random_num = random.randint(1, 100)
                    obj_file_name = f'{sss}_{prompt}.obj'
                    obj_file_path = os.path.join(output_obj_path, obj_file_name)
                    model_shop_name = f'{str(random_num)}_{prompt}.obj'
                    model_get = False
                    is_copy = False
                    print(f'Get 3D Model "{obj_file_name}"...')

                    # To make sure the same name has the same model
                    if num > 0 and prompt == prompts[num-1]:
                        shutil.copy2(f"{output_obj_path}/" + chr(64+num) + f"_{prompt}.obj", obj_file_path)
                        model_get = True
                        is_copy = True

                    '''
                    1. Search in directory
                    '''
                    if not model_get:
                        search_existing_model_result = search_existing_models_in_folder(prompt, search_main_directory)
                        if len(search_existing_model_result) > 0:
                            print(f'Found eisting models: {search_existing_model_result}')
                            choice = random.choice(search_existing_model_result)
                            print("choose model:", choice)
                            shutil.copy2(choice, obj_file_path)
                            model_get = True

                    '''
                    2. Search on website
                    '''
                    try:
                        if not model_get:
                            search_online_result = search_on_website(output_obj_path, prompt, base_search_url, obj_file_name, clip_model_1)
                            if search_online_result == True:
                                shutil.copy2(obj_file_path, os.path.join(search_main_directory, model_shop_name))
                                model_get = True
                    except Exception as e:
                        print("Download error occured, and use the TextTo3D model. ")
                        text_to_3d_generation(obj_file_path, prompt)
                        model_get = True
        
                    '''
                    3. Text to 3D Generation
                    '''
                    if not model_get:
                        print("No exsiting models and online models found, and use the TextTo3D model. ")
                        text_to_3d_generation(obj_file_path, prompt)

                    '''
                    4. Render image and Identify Face-Camera Image
                    '''
                    if is_copy:
                        answers.append(answers[-1])
                    else:
                        angle = face_camera_view_indentify(output_obj_path, output_png_path, obj_file_path, log_file, prompt, clip_model_2, preprocess_2)
                        answers.append(list(angle))

                # Saving Orientation Results
                with open(f'../output/obj_output_{sys.argv[2]}/{index}_{text}/default_orientation_info.json', 'w') as file:
                    json.dump(answers, file)

            # Debug
            except Exception as e:
                error_msg = traceback.format_exc()
                ef.write(f"model_engineer.py\n User Input {index}: {text}\nError: {error_msg}\n\n")