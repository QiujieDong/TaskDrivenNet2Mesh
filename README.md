# **A Task-driven Network for Mesh Classification and Semantic Part Segmentation**

### [Project](https://qiujiedong.github.io/publications/TaskDrivenNet2Mesh/) | [Paper](https://arxiv.org/abs/2306.05246)

**This repository is the official PyTorch implementation of our paper,  *A Task-driven Network for Mesh Classification and Semantic Part Segmentation*.**

<div align=center><img src='./assets/Mesh_MLP.webp'></div>

## Requirements
- Python 3.7
- CUDA 12.0
- PyTorch 1.11.0
- potpourri3d (pip install potpourri3d)
- robust_laplacian (pip install robust_laplacian)

## Installation

```angular2html
git clone https://github.com/QiujieDong/TaskDrivenNet2Mesh.git
cd TaskDrivenNet2Mesh
```

## Fetch Data

The URLs of the datasets used in this paper are listed in ```./data/README.md```. 

## Training

```angular2html
sh ./scripts/<DATASET_NAME>/train.sh
```

## Cite

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@article{Dong2024TaskDrivenNet2Mesh,
title = {A task-driven network for mesh classification and semantic part segmentation},
author = {Qiujie Dong and Xiaoran Gong and Rui Xu and Zixiong Wang and Junjie Gao and Shuangmin Chen and Shiqing Xin and Changhe Tu and Wenping Wang},
journal = {Computer Aided Geometric Design},
volume = {111},
pages = {102304},
year = {2024},
issn = {0167-8396},
doi = {https://doi.org/10.1016/j.cagd.2024.102304},
keywords = {Geometric deep learning, Mesh classification, Semantic part segmentation, Task-driven neural network}
}
```


## Acknowledgments
Our code is inspired by [Laplacian2Mesh](https://github.com/QiujieDong/Laplacian2Mesh) and [DiffusionNet](https://github.com/nmwsharp/diffusion-net)
