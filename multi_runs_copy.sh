input_file=$1
info=$2
camera=$3
random_number=$((20000 + RANDOM % 10001))
start_time=$(date +%s)

cd ./inference_code

# LLM 3D Layout Plan
torchrun --nproc_per_node 1 --master-port $random_number layout_plan.py  --text_file "$input_file" --append "$info"

# 3D Model Get and Align 
python model_engineer.py "$input_file" "$info"

# 3D Scene Compose and to 2D Condition Images
python render_image.py "$input_file" "$info" "$camera"

# Image Generation
python sd3_controlnet.py "$input_file" "$info"
# python sdxl_controlnet.py "$input_file" "$info"

end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Total runtime: ${total_time} seconds"