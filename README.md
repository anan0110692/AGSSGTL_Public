# Attention Guided Semi-Supervised Generative Transfer Learning for Hyperspectral Image Analysis
## Paper
![](./framework.png)

[Attention Guided Semi-Supervised Generative Transfer Learning for Hyperspectral Image Analysis](https://ieeexplore.ieee.org/document/10731899)  
 [Anan Yaghmour](https://github.com/anan0110692),  Saurabh Prasad, Melba M. Crawford


If you find this code useful for your research, please cite our [paper](https://ieeexplore.ieee.org/document/10731899):

```
@ARTICLE{10731899,
  author={Yaghmour, Anan and Prasad, Saurabh and Crawford, Melba M.},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Attention Guided Semisupervised Generative Transfer Learning for Hyperspectral Image Analysis}, 
  year={2024},
  volume={17},
  number={},
  pages={19884-19899},
  keywords={Adaptation models;Attention mechanisms;Semantics;Image analysis;Geospatial analysis;Deep learning;Hyperspectral imaging;Analytical models;Training;Semantic segmentation;Attention GANs;domain adaptation;generative adversarial learning;hyperspectral;remote sensing;semantic segmentation},
  doi={10.1109/JSTARS.2024.3485528}}

```
## Abstract
In geospatial image analysis, domain shifts caused by differences between datasets often undermine the performance of deep learning models due to their limited generalization ability. This issue is particularly pronounced in hyperspectral imagery, given the high dimensionality of the per-pixel reflectance vectors and the complexity of the resulting deep learning models. We introduce a semi-supervised domain adaptation technique that improves on the adversarial discriminative framework, incorporating a novel multi-class discriminator to address low discriminability and negative transfer issues from which current approaches suffer. Significantly, our method addresses mode collapse by incorporating limited labeled data from the target domain for targeted guidance during adaptation. Additionally, we integrate an attention mechanism that focuses on challenging spatial regions for the target mode. We tested our approach on three unique hyperspectral remote sensing datasets to demonstrate its efficacy in diverse conditions (e.g., cloud shadows, atmospheric variability, and terrain). This strategy improves discrimination and reduces negative transfer in domain adaptation for geospatial image analysis

## Preparation

### Pre-requisites
* Python 3.9.18
* PyTorch Version: 2.5.1
* CUDA Version: 12.4
### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/anan0110692/AGSSGTL.git
$ cd AGSSGTL
```

1. Setting Up the Environment
```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```
### Datasets

Ensure the `Datafiles` folder exists. If it does not, create it using the following command (Linux):

```bash
mkdir -p <root_dir>/Datasets/Datafiles
```
Once the folder is created, all data files should be put  in:
```<root_dir>/Datasets/Datafiles```
* **Download Datafiles Folder**


The datafiles folder  can be downloaded from the following link: [Google Drive Link](https://drive.google.com/drive/folders/1A1yYsHM48_Om39orlCQQQAw030ZuMb0w?usp=sharing).

The datasets included are:

- **UH Dataset** (refer to [Link](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/))
- **Botswana Dataset** (refer to [Hsiuhan Lexie Yang and Melba M. Crawford, “Domain Adaptation with
Preservation of Manifold Geometry for Hyperspectral Image Classifica-
tion,” IEEE Journal of Selected Topics in Applied Earth Observations
and Remote Sensing, vol. 9, no. 2, pp. 543–555, 2016.])
- **C2Sseg-AB Dataset** (refer to [Link](https://github.com/danfenghong))

For reproducibility, it is highly recommended to download the datafiles folder for consistent results.

  
