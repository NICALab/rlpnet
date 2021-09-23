<h2 align="center">RLP-Net: A Recursive Light Propagation Network for 3-D Virtual Refocusing</h2>
<p align="center">
<img width="90%" src="demo/RLPnet_demo_final_210617.gif">
</p>
<h6 align="center"> 3-D video reconstruction of C. elegans expressing NLS-GCaMP5K pan-neuronally using RLP-Net. 
  
  (left) video recording of two adjacent focal planes applied as the input to RLP-Net. (right) maximum intensity projection of the 3-D reconstructed video.</h6>

### [Paper]()
Official source codes for "RLP-Net: A Recursive Light Propagation Network for 3-D Virtual Refocusing", MICCAI 2021.

We propose a recursive light propagation network (RLP-Net) that infers the 3-D volume from two adjacent 2-D wide-field fluorescence images via virtual refocusing.
Specifically, we propose a recursive inference scheme in which the network progressively predicts the subsequent planes along the axial direction. 
This recursive inference scheme reflects that the law of physics for the light propagation remains spatially invariant and therefore a fixed function (i.e., a neural network) for a short distance light propagation can be recursively applied for a longer distance light propagation.


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
```bash
python test.py  --test_data_path "PATH OF TEST INPUT" --output_path "OUTPUT PATH FOR SAVING" --saved_model_path "SAVED MODEL PATH" --test_name "NAME OF TEST"
```


## Example Results

### Virtual refocusing of a fluorescent bead image
<img width="100%" src="figure/Figure3_ver4.PNG">

(a) Two adjacent images applied as the input. 
(b) Refocusing result obtained with RLP-Net. 
A 3-D view of the reconstructed volume (left). 
Refocused to z=-5um. The insets below zoom in on the three boxes (center).
Refocused to z=8um. The insets below zoom in on the boxes (right).
(c) As in (b), but for the ground truth image.
Scale bars, 2um.

## Citation
```bash
@inproceedings{shin2021rlpnet,
  title={RLP-Net: A Recursive Light Propagation Network for 3-D Virtual Refocusing},
  author={Changyeop Shin and Hyun Ryu and Eun-Seo Cho and Young-Gyu Yoon},
  booktitle={Medical Image Computing and Computer Assisted Intervention – MICCAI 2021},
  pages = {181--190},
  volume = {12906},
  year={2021},
  doi={https://doi.org/10.1007/978-3-030-87231-1_18},
  url={https://link.springer.com/chapter/10.1007%2F978-3-030-87231-1_18#citeas},
}
```

