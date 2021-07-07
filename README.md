<h2 align="center">RLP-Net: A Recursive Light Propagation Network for 3-D Virtual Refocusing</h2>
<p align="center">
<img width="90%" src="demo/RLPnet_demo_final_210617.gif">
</p>

### [Paper]()
Official source codes for "RLP-Net: A Recursive Light Propagation Network for 3-D Virtual Refocusing", MICCAI2021.

We propose a recursive light propagation network (RLP-Net) that infers the 3-D volume from two adjacent 2-D wide-field fluorescence images via virtual refocusing.
Specifically, we propose a recursive inference scheme in which the network progressively predicts the subsequent planes along the axial direction. 
This recursive inference scheme re ects that the law of physics for the light propagation remains spatially invariant and therefore a fixed function (i.e., a neural network) for a short distance light propagation can be recursively applied for a longer distance light propagation.
## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Requirements
```markdown
python==3.6
torch==1.7.0
torchvision==0.8.1
numpy==1.19.4
skimage==0.17.2
scipy==1.5.3
```

### Update log

07/08/2021: initial commit

### Getting started

- Clone this repo:
```bash
git clone https://https://github.com/NICALab/rlpnet rlpnet
cd rlpnet
```

## Train
```bash
python train.py --dataset_name "NAME OF DATASET" --root2 "SAVING PATH OF TRAINING RESULTS"
```

## Test

