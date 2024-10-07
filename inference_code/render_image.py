import json
import math
import os
import random
import shutil
import sys
import bpy
import cv2
import numpy as np


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
        text_y = y + 20
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness)

    return image


def initialize(scene, output_img_dir, main_directory, key_data, item_data):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # Environment Settings
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node is None:
        bg_node = world.node_tree.nodes.new('ShaderNodeBackground')
    bg_node.inputs[0].default_value = (0.1, 0.1, 0.1, 1)
    bg_node.inputs[1].default_value = 1.0
    
    # Rendering Settings
    scene.render.engine = 'CYCLES'
    scene.view_layers[0].use_pass_z = True
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.use_nodes = True
    scene.render.filepath = os.path.join(output_img_dir, "frame000") + ".png"
    scene.render.image_settings.file_format = 'PNG'

    camera = bpy.data.objects['Camera']
    light = bpy.data.objects['Light']
    depth = [item.get("depth", 0) for item in item_data]
    area = [[item["left"], item["top"], item["left"] + item["width"], item["top"] + item["height"]] for item in item_data]
    average_depth = (max(depth) + min(depth)) / 2
    average_area = [[max(item), min(item)] for item in area]

    # Camera View
    if "camera" in item_data[0]:
        if item_data[0]['camera'] == "left view":
            randon_x = random.randint(-10, -5)
        elif item_data[0]['camera'] == "right view":
            randon_x = random.randint(5, 10)
        else:
            randon_x = random.randint(-1, 1)
    else: 
        randon_x = random.randint(-1, 1)
    if "camera" in item_data[0] and item_data[0]['camera'] == "top view":
        randon_y = random.randint(-10, -5)
        randon_y = average_depth * 10 - 3
        # randon_z = random.randint(15, 20)
        camera.location = (0, randon_y, math.sqrt(250 - randon_y**2))
    else:
        camera.location = (randon_x, -math.sqrt(200 - randon_x**2), 5)
    # camera.location = (5, -10, 10)
    camera_x, camera_y, camera_z = camera.location
    camera.rotation_euler = (math.atan(math.sqrt(camera_x**2 + camera_y**2)/ (camera_z + 1e-5)), 0, -math.atan(camera_x / (camera_y + 1e-5)))
    
    # Light Configuration
    light.location = (0, -5, 10)
    light.rotation_euler = (0, 0, 0)
    
    # Clear Nodes
    tree = scene.node_tree
    links = tree.links
    for n in tree.nodes:
        tree.nodes.remove(n)
    
    # Depth Image
    render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')
    invert_node = tree.nodes.new(type='CompositorNodeInvert')
    depth_output_node = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output_node.base_path = output_img_dir + "/depth"
    depth_output_node.file_slots[0].path = "depth000"

    map_node = tree.nodes.new('CompositorNodeMapValue')
    map_node.offset = [-20]
    map_node.size = [0.1]

    links.new(render_layers_node.outputs[2], map_node.inputs[0])
    links.new(map_node.outputs[0], invert_node.inputs[1])
    links.new(invert_node.outputs[0], depth_output_node.inputs[0])



def render_3D_to_2D(scene, llm_info, vllm_info, rotation_data, output_img_dir, obj_files, log_file, main_directory):
    key_data = [key for item_info in llm_info for key, _ in item_info.items()]
    item_data = [value for item_info in llm_info for _, value in item_info.items()]
    bboxs = [[item["left"], item["top"], item["width"], item["height"]] for item in item_data]
    current_bbox = draw_bounding_box(bboxs, key_data)
    cv2.imwrite(main_directory + '/bbox.png', current_bbox)
    initialize(scene, output_img_dir, main_directory, key_data, item_data)
    
    for i in range(len(obj_files)): 
        # Import 3D Model
        bpy.ops.wm.obj_import(filepath=obj_files[i])
        # Merge Extra Components
        if (len(bpy.data.objects) > 3 + i):
            # print(len(bpy.data.objects))
            obs = bpy.context.scene.objects[2:]
            ctx = bpy.context.copy()
            ctx['selected_objects'] = obs
            bpy.ops.object.join()
        obj = bpy.context.selected_objects[0] 

        # Check Value First
        if ('depth' not in item_data[i]) or not isinstance(item_data[i]['depth'], float):
            item_data[i]['depth'] = 0.0
        if ('orientation' not in item_data[i]) or (item_data[i]['orientation'] not in ["forward", "backward", "left", "right", "upward", "downward"]):
            item_data[i]['orientation'] = "forward"

        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='BOUNDS')
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
        bpy.context.view_layer.update()

        # Set Orientation
        obj.rotation_euler = [x + y for x, y in zip(vllm_info[i], rotation_data[item_data[i]['orientation']])]
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.context.view_layer.update()

        # Set Size
        height_width_ratio = obj.dimensions[2] / obj.dimensions[0]
        height_depth_ratio = obj.dimensions[2] / obj.dimensions[1]
        width_depth_ratio = obj.dimensions[0] / obj.dimensions[1]
        max_obj_size = max(obj.dimensions[0], obj.dimensions[2])
        max_item_size = max(item_data[i]['width'], item_data[i]['height']) * 10 + item_data[i]['depth']
        if max_obj_size == obj.dimensions[0]:
            obj.dimensions = (max_item_size, max_item_size / width_depth_ratio, max_item_size * height_width_ratio)
        else:
            obj.dimensions = (max_item_size / height_width_ratio, max_item_size / height_depth_ratio, max_item_size)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bpy.context.view_layer.update()
        if item_data[i]['depth'] < 0.5:
            while sum(obj.dimensions) > 25:
                obj.dimensions = obj.dimensions / 1.5
                bpy.context.view_layer.update()
                print('--------', obj.dimensions)

        # Set Location
        obj.delta_location[0] = (item_data[i]['left'] + item_data[i]['width'] / 2 - 0.5) * 10
        obj.delta_location[1] = item_data[i]['depth'] * (5 + obj.dimensions[1])
        obj.delta_location[2] = - (item_data[i]['top'] + item_data[i]['height'] / 2 - 0.5) * 10 - item_data[i]['depth']

    # Redirect Output
    open(log_file, 'a').close()
    old = os.dup(sys.stdout.fileno())
    sys.stdout.flush()
    os.close(sys.stdout.fileno())
    fd = os.open(log_file, os.O_WRONLY)

    # Render Image
    bpy.ops.render.render(write_still=True)
    os.close(fd)
    os.dup(old)
    os.close(old)



if __name__ == "__main__":
    with open(sys.argv[1], 'r') as file:
        all_lines = file.readlines()
    all_lines = [line.strip() for line in all_lines]


    rotation = {
        "forward": [0, 0, 0], 
        "backward": [0, 0, 180 / 180 * math.pi], 
        "left": [0, 0, -90 / 180 * math.pi], 
        "right": [0, 0, 90 / 180 * math.pi], 
        "upward": [-90 / 180 * math.pi, 0, 0], 
        "downward": [90 / 180 * math.pi, 0, 0]
    }

    # Start Render!
    for index, text in enumerate(all_lines, start=1):
        text = text[:100]
        path = f'../output/obj_output_{sys.argv[2]}/{index}_{text}'
        if not os.path.exists(f'{path}/entity_info.json') or not os.path.exists(f'{path}/default_orientation_info.json'):
            continue
        with open(f'{path}/entity_info.json', 'r') as file:
            llm_info = json.load(file)
        with open(f'{path}/default_orientation_info.json', 'r') as file:
            vllm_info = json.load(file)
        print(f"Render Image —— Line {index}: {text}")

        other_objs_path = path + "/obj"
        other_objs = [os.path.join(other_objs_path, f) for f in sorted(os.listdir(other_objs_path))]
        output_img_dir = path + "/rendering"
        os.makedirs(output_img_dir, exist_ok=True)

        # Logging File
        log_file = 'blender.log'
        render_3D_to_2D(bpy.context.scene, llm_info, vllm_info, rotation, output_img_dir, other_objs, log_file, path)