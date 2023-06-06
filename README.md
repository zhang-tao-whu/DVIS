<div align="center">

# [DVIS: Decoupled Video Instance Segmentation Framework]()
[Tao Zhang](https://scholar.google.com/citations?user=3xu4a5oAAAAJ&hl=zh-CN), XingYe Tian, [Yu Wu](https://scholar.google.com/citations?hl=zh-CN&user=23SZHUwAAAAJ), [ShunPing Ji](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=zh-CN), Xuebo Wang, Yuan Zhang, Pengfei Wan
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/pipeline.png" width="800"/>
</div>

## News
- DVIS achieved **1st place** in the VPS Track of the PVUW challenge at CVPR 2023. `2023.5.25`

## Features
- DVIS is a universal video segmentation framework that supports VIS, VPS and VSS.
- DVIS can run in both online and offline modes. 
- DVIS achieved SOTA performance on YTVIS, OVIS, VIPSeg and VSPW datasets.
- DVIS can complete training and inference on GPUs with only 11G memory. 

## Demos
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_0.gif" width="400"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_1.gif" width="370"/>
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_2.gif" width="215"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_4.gif" width="290"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_5.gif" width="290"/>
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_6.gif" width="400"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/demo_7.gif" width="400"/>

## Installation

See [Installation Instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for DVIS](datasets/README.md).

See [Getting Started with DVIS](GETTING_STARTED.md).

## Model Zoo

Trained models are available for download in the [DVIS Model Zoo](MODEL_ZOO.md).

## Acknowledgement

This repo is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), [MinVIS](https://github.com/NVlabs/MinVIS) and [VITA](https://github.com/sukjunhwang/VITA).
Thanks for their excellent works.