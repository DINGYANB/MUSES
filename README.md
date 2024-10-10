<div align="center">

<h2><a href="https://arxiv.org/abs/2408.10605">MUSES: 3D-Controllable Image Generation via Multi-Modal Agent Collaboration</a></h2>


[Yanbo Ding*](https://github.com/DINGYANB),
[Shaobin Zhuang](https://scholar.google.com/citations?user=PGaDirMAAAAJ&hl=zh-CN&oi=ao), 
[Kunchang Li](https://scholar.google.com/citations?user=D4tLSbsAAAAJ), 
[Zhengrong Yue](https://arxiv.org/search/?searchtype=author&query=Zhengrong%20Yue), 
[Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl), 
[Yali Wang†](https://scholar.google.com/citations?user=hD948dkAAAAJ)

[![arXiv](https://img.shields.io/badge/arXiv-2408.10605-b31b1b.svg)](https://arxiv.org/abs/2408.10605)
[![GitHub](https://img.shields.io/badge/GitHub-MUSES-blue?logo=github)](https://github.com/DINGYANB/MUSES)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/yanboding/MUSES/)

</div>

## 💡 Motivation

Despite recent advancements in text-to-image generation, most existing methods struggle to create images with multiple objects and complex spatial relationships in 3D world. To tackle this limitation, we introduce a generic AI system, namely MUSES, for 3D-controllable image generation from user queries.

<img width="800" alt="image" src="https://github.com/DINGYANB/MUSES/blob/main/assets/demo.png">
</a>


## 🤖 Architecture

Our MUSES realize 3D controllable image generation by developing a progressive workflow with three key components, including:
1. Layout Manager for 2D-to-3D layout lifting;
2. Model Engineer for 3D object acquisition and calibration;
3. Image Artist for 3D-to-2D image rendering.

By mimicking the collaboration of human professionals, this multi-modal agent pipeline facilitates the effective and automatic creation of images with 3D-controllable objects, through an explainable integration of top-down planning and bottom-up generation. 

<img width="800" alt="image" src="https://github.com/DINGYANB/MUSES/blob/main/assets/overview.png">
</a>


## 🔨 Installation

1. Clone this GitHub repository and install the required packages:

    ``` shell
    git clone https://github.com/DINGYANB/MUSES.git
    cd MUSES

    conda create -n MUSES python=3.10
    conda activate MUSES

    pip install -r requirements.txt
    ```


2. Download other required models:

    | Model                |     Storage Path     |    Description    |
    |----------------------|----------------------|-------------------|
    | [OpenAI ViT-L-14](https://huggingface.co/openai/clip-vit-large-patch14) | `model/CLIP/` | Similarity Comparison |
    | [Meta Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | `model/Llama3/` | 3D Layout Planning | 
    | [stabilityai stable-diffusion-3-medium (SD3)](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | `model/SD3-Base/` | Image Generation |
    | [InstantX SD3-Canny-ControlNet](https://huggingface.co/InstantX/SD3-Controlnet-Canny) | `model/SD3-ControlNet-Canny/` | Controllable Image Generation |
    | [examples_features.npy](https://huggingface.co/yanboding/MUSES/upload/main) | `/dataset/` | In-Context Learning |
    | [finetuned_clip_epoch_20.pth](https://huggingface.co/yanboding/MUSES/upload/main) | `/model/CLIP/` | Orientation Calibration |

    Since our MUSES is a training-free multi-model collaboration system,  feel free to replace the generative models with other competitive ones. For example, we recommend users to replace the Llama-3-8B with more powerful LLMs like [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) and [GPT 4o](https://platform.openai.com/docs/models/gpt-4o).

3. *Optional* Downloads:
- Download our self-built 3D model shop at this [link](https://huggingface.co/yanboding/MUSES/upload/main), which includes 300 high-quality 3D models, and 1500 images of various objects with different orientations for fine-tuing the [CLIP](https://huggingface.co/openai/clip-vit-base-patch32).
- Download multiple ControlNets such as [SD3-Tile-ControlNet](https://huggingface.co/InstantX/SD3-Controlnet-Tile), [SDXL-Canny-ControlNet](https://huggingface.co/TheMistoAI/MistoLine), [SDXL-Depth-ControlNet](https://huggingface.co/diffusers/controlnet-zoe-depth-sdxl-1.0), and other image generation models, e.g., [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).


## 🌟 Usage
Use the following command to generate images.
``` shell
cd MUSES && bash multi_runs.sh "test_prompt.txt" "test"
```
Where the **first argument** is the input txt file containing the prompts in rows, and the **second argument** is the identifier of the current run, which will be appended to the output folder name. For SD3-Canny-ControlNet, each prompt results in 5 images of different control scales.


## 📊 Dataset & Benchmark
### Expanded NSR-1K
Since the original [NSR-1K](https://github.com/Karine-Huang/T2I-CompBench) dataset lacks layouts in 3D scenes and complex scenes, so we manually add some
prompts with corresponding layouts.
Our expanded NSR-1K dataset is in the directory `dataset/NSR-1K-Expand`.

### Benchmark Evaluation

For *T2I-CompBench* evaluation, we follow its official evaluation codes in this [link](https://github.com/Karine-Huang/T2I-CompBench). Note that we choose the best score among the 5 images as the final score. 

Since T2I-CompBench lacks detailed descriptions of complex 3D spatial relationships of multiple objects, we construct our T2I-3DisBench (`dataset/T2I-3DisBench.txt`), which describes diverse 3D image scenes with 50 detailed prompts. 
For *T2I-3DisBench* evaluation, we employ [Mini-InternVL-2B-1.5](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) to score the generated images from 0.0 to 1.0 across four dimensions, including object count, object orientation, 3D spatial relationship, and camera view. You can download the weights at this [link](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5) and put them into the folder `model/InternVL/`.

``` shell
python inference_code/internvl_vqa.py
```

After running it, it will output an average score.
Our MUSES demonstrates state-of-the-art performance on both benchmarks, verifying its effectiveness.

## 💙 Acknowledgement
MUSES is built upon 
[Llama](https://github.com/meta-llama/llama3), 
[NSR-1K](https://github.com/Karine-Huang/T2I-CompBench), 
[Shap-e](https://github.com/openai/shap-e), 
[CLIP](https://github.com/openai/CLIP), 
[SD](https://github.com/Stability-AI/generative-models),
[ControlNet](https://github.com/lllyasviel/ControlNet).
We acknowledge these open-source codes and models, and the website [CGTrader](https://www.cgtrader.com) for supporting 3D model free downloads.
We appreciate as well the valuable insights from the researchers
at the Shenzhen Institute of Advanced Technology and the
Shanghai AI Laboratory.


## 📝 Citation
If our MUSES system helps your work, please cite this paper:
```bib
@article{ding2024muses,
      title={MUSES: 3D-Controllable Image Generation via Multi-Modal Agent Collaboration}, 
      author={Yanbo Ding and Shaobin Zhuang and Kunchang Li and Zhengrong Yue and Yu Qiao and Yali Wang},
      journal={arXiv preprint arXiv:2408.10605},
      year={2024},
}
```
