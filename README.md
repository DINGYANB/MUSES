<div align="center">

<h2><a href="https://arxiv.org/abs/2408.10605">MUSES: 3D-Controllable Image Generation via Multi-Modal Agent Collaboration</a></h2>


[Yanbo Ding*](https://github.com/DINGYANB),
[Shaobin Zhuang](https://scholar.google.com/citations?user=PGaDirMAAAAJ&hl=zh-CN&oi=ao), 
[Kunchang Li](https://scholar.google.com/citations?user=D4tLSbsAAAAJ), 
[Zhengrong Yue](https://arxiv.org/search/?searchtype=author&query=Zhengrong%20Yue), 
[Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl), 
[Yali Wang†](https://scholar.google.com/citations?user=hD948dkAAAAJ)

[![arXiv](https://img.shields.io/badge/arXiv-2401.09414-b31b1b.svg)](https://arxiv.org/abs/2401.09414)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/yanboding/MUSES/)
<!-- [![YouTube Video](https://img.shields.io/badge/YouTube-Video-red)](https://youtu.be/ZRD1-jHbEGk) -->
<!-- [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhuangshaobin%2FVlogger&count_bg=%23F59352&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com) -->
<!-- [![Project Page](https://img.shields.io/badge/Vlogger-Website-green)](https://zhuangshaobin.github.io/Vlogger.github.io/) -->


</div>

## 💡 Motivation

Despite recent advancements in text-to-image generation, most existing methods struggle to create images with multiple objects and complex spatial relationships in 3D world. To tackle this limitation, we introduce a generic AI system, namely MUSES, for 3D-controllable image generation from user queries.

<img width="800" alt="image" src="https://github.com/DINGYANB/MUSES/blob/main/assets/demo.png">
</a>


## 🤖 Architecture

Our MUSES realize 3D controllable image generation by developing a progressive workflow with three key components, including 
1. Layout Manager for 2D-to-3D layout lifting
2. Model Engineer for 3D object acquisition and calibration
3. Image Artist for 3D-to-2D image rendering

By mimicking the collaboration of human professionals, this multi-modal agent pipeline facilitates the effective and automatic creation of images with 3D-controllable objects, through an explainable integration of top-down planning and bottom-up generation. 

<!-- Additionally, we find that existing benchmarks lack detailed descriptions of complex 3D spatial relationships of multiple objects. To fill this gap, we further construct a new benchmark of T2I-3DisBench (3D image scene), which describes diverse 3D image scenes with 50 detailed prompts.  -->

<img width="800" alt="image" src="https://github.com/DINGYANB/MUSES/blob/main/assets/overview.png">
</a>


## 🔨 Installation

1. Clone this github repository and install the required packages:

    ```shell
    git clone https://github.com/DINGYANB/MUSES.git
    cd MUSES

    conda create -n MUSES python=3.10
    conda activate MUSES

    pip install -r requirements.txt
    ```

<!-- 2. Download encoded features and the fine-tuned CLIP model:
    
- Put [examples_features.npy](https://huggingface.co/yanboding/MUSES/upload/main) into the folder `dataset/`
- Put [finetuned_clip_epoch_20.pth](https://huggingface.co/yanboding/MUSES/upload/main) into the folder `model/CLIP/` -->

2. Download other required models:

    | Model                |     Storage Path     |    Role    |
    |----------------------|----------------------|-------------|
    | [OpenAI ViT-L-14](https://huggingface.co/openai/clip-vit-large-patch14) | `model/CLIP/` | Similarity Comparison |
    | [Meta Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) | `model/Llama3/` | 3D Layout Planning | 
    | [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) | `model/SD3-Base/` | Image Generation |
    | [SD3-Canny-ControlNet](https://huggingface.co/InstantX/SD3-Controlnet-Canny) | `model/SD3-ControlNet-Canny/` | Controllable Image Generation |
    | [examples_features.npy](https://huggingface.co/yanboding/MUSES/upload/main) | `/dataset/` | In-Context Learning |
    | [finetuned_clip_epoch_20.pth](https://huggingface.co/yanboding/MUSES/upload/main) | `/model/CLIP/` | Orientation Calibration |

    Since our MUSES is a training-free multi-model collaboration system,  feel free to replace the generated models with other competitive ones. For example, we recommend users to replace the Llama-3-8B with more powerful LLM like [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) and [GPT 4o](https://platform.openai.com/docs/models/gpt-4o).

3. Optional Downloads:
- Download our self-built 3D model shop at this [link](https://huggingface.co/yanboding/MUSES/upload/main), which includes 300 high quality 3D models, and 1500 images of various objects with different orientations for fine-tuing the [CLIP](https://huggingface.co/openai/clip-vit-base-patch32).
- Download multiple ControlNets such as [SD3-Tile-ControlNet](https://huggingface.co/InstantX/SD3-Controlnet-Tile), [SDXL-Canny-ControlNet](https://huggingface.co/TheMistoAI/MistoLine), [SDXL-Depth-ControlNet](https://huggingface.co/diffusers/controlnet-zoe-depth-sdxl-1.0), and other image generation models, e.g., [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) with [VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix).

## 📊 Dataset & Benchmark



## 💙 Acknowledgement
MUSES is built upon 
[Llama](https://github.com/meta-llama/llama3), 
[Shap-e](https://github.com/openai/shap-e), 
[CLIP](https://github.com/openai/CLIP), 
[SD](https://github.com/Stability-AI/generative-models),
[ControlNet](https://github.com/lllyasviel/ControlNet).
We acknowledge these open-source codes and models, and the website [CGTrader](https://www.cgtrader.com) for supporting 3D model free downloads.
We appreciate as well the valuable insights from researchers
at the Shenzhen Institute of Advanced Technology and the
Shanghai AI Laboratory.


## 📝 Citation
```bib
@article{ding2024muses,
      title={MUSES: 3D-Controllable Image Generation via Multi-Modal Agent Collaboration}, 
      author={Yanbo Ding and Shaobin Zhuang and Kunchang Li and Zhengrong Yue and Yu Qiao and Yali Wang},
      journal={arXiv preprint arXiv:2408.10605},
      year={2024},
}
```
