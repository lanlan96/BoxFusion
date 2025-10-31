<div align="center">
<h1>BoxFusion: Reconstruction-Free Open-Vocabulary 3D Object Detection via Real-Time Multi-View Box Fusion</h1>


<a href="https://arxiv.org/pdf/2506.15610"><img src="https://img.shields.io/badge/arXiv-2506.15610-b31b1b" alt="arXiv"></a>
<a href="https://lanlan96.github.io/BoxFusion/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

[Yuqing Lan](https://scholar.google.com/citations?user=laTrw7AAAAAJ&hl=en&oi=ao), [Chenyang Zhu](https://www.zhuchenyang.net/), [Zhirui Gao](https://scholar.google.com/citations?hl=en&user=IqtwGzYAAAAJ), [Jiazhao Zhang](https://jzhzhang.github.io/), [Yihan Cao](https://github.com/yhanCao), [Renjiao Yi](https://renjiaoyi.github.io/), [Yijie Wang](https://ieeexplore.ieee.org/author/37540196000), [Kai Xu](https://kevinkaixu.net/)
</div>

This repository includes the public implementation of BoxFusion.

## 📢 News
- **2025-10-31**: Guidelines for ROS2 demo (to be oganized).
- **2025-08-30**: Code is released.
- **2025-08-10**: BoxFusion is accepted by Pacific Graphics 2025 (Journal Track), the **top 5%** paper.


## 📋 TODO

- [x] Release the codes and demos.
- [x] Release the online ROS demo for detecting neighboring objects while the user/agent is scanning. 
- [ ] Release the evaluation code (before 11.30).

## 1. Installation

Please create the virtual environment with python3.10 and a recent 2.x build of PyTorch. The code has been tested on Ubuntu22.04 and CUDA 11.8. The environment can be created like:

```
conda create -n boxfusion python=3.10 
conda activate boxfusion
```
Install PyTorch:
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```
Then you can install the dependencies:
```
pip install -r requirements.txt
pip install -e .
```

## 2. Quick Start

1.Download the pre-trained RGB-D model [Cubify Anything](https://github.com/apple/ml-cubifyanything?tab=readme-ov-file#running-the-cutr-models). Please the follow the license of Cubify Anything. Similarly, you need to download the [CLIP](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/open_clip_pytorch_model.bin). After you download these models, please move them into the `models` folder.
```
models
  |-- cutr_rgbd.pth
  |-- open_clip_pytorch_model.bin
```
2.Download the [example data](https://drive.google.com/file/d/15l5e_pcN-vm6iIx58gkObdNHaog_KZpH/view?usp=sharing) from google drive. Move it into `data` folder and unzip the data. 

3.Run the demo.py using the example data for a quick start. This demo will load the data automatically, and the visualization will present the sequential RGB, depth, 3D object boxes, with the camera trajtory. You can change the configuration to the customized data. Note that, the online visualization process will slightly slow down the system FPS, and you can switch to `rerun=False` in the config file for acceleration.
```
python demo.py CA1M --model-path ./models/cutr_rgbd.pth  --config ./config/ca1m.yaml --device cuda --seq 42898867
```



## 3. Data Preparation
Basically, we organize the data like most SLAM methods. There are two datasets we utilize in the benchmark: [CA-1M](https://github.com/apple/ml-cubifyanything) and [ScanNetV2](http://www.scan-net.org/). If you want to test all sequences on these two datasets, please follow the steps in this section.

### CA-1M

1.Following [Cubify Anything](https://github.com/apple/ml-cubifyanything), please download the data with the links in `data/val.txt`. You can use `wget`, `curl` or any tool to download the data you want. As for the evaluation, all sequences in `data/val.txt` are required.

2.Prepare the data structure according to [README](./data_process/README.md).

The data structure is like this:

<details>
<summary>[Structure for CA-1M dataset (click to expand)]</summary>

```
CA-1M/
├── 48458654/                
│   ├── depth/               # Folder containing depth images
│   ├── rgb/                 # Folder containing RGB color images
│   ├── after_filter_boxes.npy  # Filtered gt 3D bounding boxes 
│   ├── all_poses.npy        # Camera poses for a sequence of frames [N,4,4]
│   ├── instances.json       # Instance segmentation or object detection results
│   ├── K_depth.txt          # Intrinsic camera matrix for the depth sensor
│   ├── K_rgb.txt            # Intrinsic camera matrix for the RGB sensor
│   ├── mesh.ply             # Reconstructed 3D mesh
│   └── T_gravity.npy        # Transformation matrix for gravity alignment
```
</details>

### ScanNetV2
Please follow the process of [ScanNetV2](http://www.scan-net.org/) to download the validation sets. We use the default data structure like this:
<details>
<summary>[Structure for ScanNetV2 dataset (click to expand)]</summary>

```
ScanNet/
├── scene0xxx_0x/
│   ├── frames/
│   │   ├── color/
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   ├── depth/
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   └── ...
│   │   ├── pose/
│   │   │   ├── 0.txt
│   │   │   ├── 1.txt
│   │   │   └── ...
│   │   ├── intrinsic/
│   │   │   └── intrinsic_depth.txt
│   └── scene0xxx_0x.txt
│   └── scene0xxx_0x_vh_clean_2.ply
└── ...
```
</details>

## 4. Run
In this section, we introduce how to run a given sequence (CA-1M dataset or ScanNetV2). After you have prepared the datasets according to the instructions above, you can run the following commands to try BoxFusion on the specific sequence.
### CA-1M
Please change the `datadir` in the `config/ca1m.yaml` to root of your processed CA-1M dataset. Customize the `--seq` to the sequence you want to try.
```
python demo.py CA1M --model-path ./models/cutr_rgbd.pth  --config ./config/ca1m.yaml --device cuda --seq 42898867
```

### ScanNetV2
Please change the `datadir` in the `config/scannet.yaml` to root of your processed ScanNetV2 dataset. Customize the `--seq` to the sequence you want to try.
```
python demo.py scannet --model-path ./models/cutr_rgbd.pth  --config ./config/scannet.yaml --device cuda --seq scene0169_00
```
### Others
We recommend to prepare the data like ScanNetV2. Once you have prepared the data, you can instantiate a dataset object in this [file](./cubifyanything/capture_stream.py), and use the similar command to try on your data.

## 5. ROS2 demo guideline
We update the basic code for the ROS2 version for BoxFusion. If you use `online` dataset, the dataloader will automatically listen the topic `/rgb/image_raw` for RGB, `/rgb/image_raw` for depth and `/trajectory` for camera pose. You can try any method to obtain the posed RGB-D data, given the raw RGB-D images (We recommend the sparse method [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)).

After you have started the node for online pose estimation, you can run the following command to visualize the online detection of BoxFusion.

```
python demo.py online --model-path ./models/cutr_rgbd.pth  --config ./config/online.yaml --device cuda 
```

## Acknowledgement
Parts of the code are modified from [Cubify Anything](https://github.com/apple/ml-cubifyanything). Thanks to the authors and please consider citing their papers.


## Citation
If you find our work useful in your research, please consider giving a star ✨ and citing the following paper:
```
@article{lan2025boxfusion,
  title={BoxFusion: Reconstruction-Free Open-Vocabulary 3D Object Detection via Real-Time Multi-View Box Fusion},
  author={Lan, Yuqing and Zhu, Chenyang and Gao, Zhirui and Zhang, Jiazhao and Cao, Yihan and Yi, Renjiao and Wang, Yijie and Xu, Kai},
  journal={arXiv preprint arXiv:2506.15610},
  year={2025}
}
```
