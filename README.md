<p align="center">
  <h1 align="center">DroneSplat: 3D Gaussian Splatting for Robust 3D Reconstruction from In-the-Wild Drone Imagery</h1>
<p align="center">

<p align="center">
  <a href="https://dronesplat.github.io/">Project page</a> | 
  <a href="https://drive.google.com/drive/folders/1DWm-foUQC2QBsrr3QC6Tx8bDmWTfgAzu?usp=sharing">Dataset</a> 
</p>

## Get Started

### Installation
1. Clone DroneSplat and download pre-trained model.
```bash
git clone --recursive https://github.com/DroneSplat/anonymous_code.git
cd DroneSplat
git submodule update --init --recursive
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/
```

2. Create the environment.
```bash
conda create -n dronesplat python=3.11
conda activate dronesplat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
```

3. Build SAM2 Model
```bash
cd submodules/sam2
pip install -e .
cd ..
```

4. Optional but highly suggested, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd submodules/dust3r/croco/models/curope/
python setup.py build_ext --inplace
```

### Data preparation
We provide six scenes of [DroneSplat dataset](https://drive.google.com/drive/folders/1DWm-foUQC2QBsrr3QC6Tx8bDmWTfgAzu?usp=sharing) for evaluation.

In addition, we also conduct experiments on [NeRF On-the-go](https://github.com/cvg/nerf-on-the-go) and [UrbanScene3D](https://vcc.tech/UrbanScene3D).

## Usage

### Training
Run the following command to train on Simingshan.
```
python preprocess.py --img_base_path data/Simingshan --colmap_path data/Simingshan/sparse/0 --preset_pose
```
```
python train.py -s data/Simingshan -m output/Simingshan --scene Simingshan --iter 7000 --use_masks --mask_start_iter 500 --threshold_local 0.4 --video_segment --video_seg_start_iter 500 --threshold_global 2.8
```
### Rendering
Run the following script to render train and  test images:
```
python render.py -s data/Simingshan -m output/Simingshan --iter 7000
```
Run the following script to render a video:
```
python render_video.py -s data/Simingshan -m output/Simingshan --iter 7000 --n_views 600 --fps 30
```

### Evaluation
```
python metrics.py --rendering output/Simingshan/render_test --gt data/Simingshan --output output/Simingshan/metrics.json
```

## Shoutouts and Credits
This project is built on top of open-source code. We thank the open-source research community and credit our use of parts of [3D Guassian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [DUSt3R](https://github.com/naver/dust3r), and [InstantSplat](https://github.com/NVlabs/InstantSplat). 
