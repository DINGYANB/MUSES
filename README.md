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

<img width="743" alt="image" src="https://github.com/DINGYANB/MUSES/blob/main/assets/demo.png">
</a>


##  Method

Our MUSES realize 3D controllable image generation by developing a progressive workflow with three key components, including 
1. Layout Manager for 2D-to-3D layout lifting
2. Model Engineer for 3D object acquisition and calibration
3. Image Artist for 3D-to-2D image rendering

By mimicking the collaboration of human professionals, this multi-modal agent pipeline facilitates the effective and automatic creation of images with 3D-controllable objects, through an explainable integration of top-down planning and bottom-up generation. 

<!-- Additionally, we find that existing benchmarks lack detailed descriptions of complex 3D spatial relationships of multiple objects. To fill this gap, we further construct a new benchmark of T2I-3DisBench (3D image scene), which describes diverse 3D image scenes with 50 detailed prompts.  -->

<img width="743" alt="image" src="https://github.com/DINGYANB/MUSES/blob/main/assets/overview.png">
</a>

## 🔨 Installation

Clone this github repository and install the required packages:

```shell
git clone https://github.com/DINGYANB/MUSES.git
cd MUSES

conda create -n MUSES python=3.10
conda activate MUSES

pip install -r requirements.txt
```

## Dataset


## 💙 Acknowledgement
MUSES is built upon [Llama 3.0](https://github.com/mihirp1998/Diffusion-TTA), [Shap-e](https://github.com/tsb0601/MMVP), [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [timm](https://github.com/huggingface/pytorch-image-models/). 
We acknowledge these open-source models, and
[CGTrader](https://www.cgtrader.com) for supporting 3D model crawler free downloads.
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
