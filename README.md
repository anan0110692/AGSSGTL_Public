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
- **Botswana Dataset** (refer to [1])
- **C2Sseg-AB Dataset** (refer to [Link](https://github.com/danfenghong))


For reproducibility, it is highly recommended to download the provided datafiles folder for consistent results.
## Running the Code

The code is structured to represent the three main phases illustrated in the diagram above:

1. **Pre-training Phase**: Implemented in the `Pre-train` notebook.
2. **Adversarial Training Phase**: Implemented in the `Adversarial-train` notebook.
3. **Inference Phase**: Implemented in the `Test` notebook.

The notebooks should be executed in the order listed above to ensure proper workflow and reproducibility.

### Pre-train Notebook Guide

1. **Select the Dataset**  
   In the second cell of the notebook, choose the appropriate dataset file by specifying `Dataset.<Dataset file>`. For example:
   ```python
   Dataset.UH_Dataset
2. **Set the Experiment Name**  
   In the third cell (*interface cell* ), assign a value to `Exp_name`. By default, logs and snapshots will be stored in the following directory structure:   

```bash
<root_dir>/Results/<Dataset file>/DA/Exp_<Exp_name>
<root_dir>/Results/<Dataset file>/Exp_<Exp_name>
<root_dir>/Results/<Dataset file>/Source/Exp_<Exp_name>
<root_dir>/Results/<Dataset file>/lightning_logs/Exp_<Exp_name>
```

- The `DA` folder contains the checkpoints for the adversarial training phase.  
- The `Source` folder contains the checkpoints for the pre-training phase.  
- The `Exp_<Exp_name>` folder contains a list of ready-to-use models.  
- The `lightning_logs` folder contains TensorBoard logs for tracking experiments and runs.
3. Run all cells

  ### Adversarial-train Notebook Guide

1. **Select the Dataset**  
   In the second cell of the notebook, choose the appropriate dataset file by specifying `Dataset.<Dataset file>`. For example:
   ```python
   Dataset.UH_Dataset
2. **Set Pre-trained Models Path**
   Typically, this path should be provided after running the Pre-training notebook. For example, for an experiment with the name `dummy_Pre_train`, the path would look like:

```bash
<root_dir>/Results/UH/dummy_Pre_train/UH_lists.pkl
```
   
3. **Set the Experiment Name**  
   In the third cell (*interface cell*), assign a value to `Exp_name`. 
4. Run all cells

### Test Notebook Guide

1. **Select the Dataset**  
   In the second cell of the notebook, choose the appropriate dataset file by specifying `Dataset.<Dataset file>`. For example:
   ```python
   Dataset.UH_Dataset
2. **Set Adapted Models Path**
   Typically, this path should be provided after running the Adversarial-train notebook. For example, for an experiment with the name `dummy_DA`, the path would look like:

```bash
<root_dir>/Results/UH/dummy_DA/UH_lists.pkl
```
   
 
3. Run the desired evaluation cell.

### Pre-trained parameters
Pre-trained parameters can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1bVt_3WdXlyMq80Y9cF7XoustsMOKj949?usp=sharing). Each dataset includes two `.pkl` files, corresponding to the pre-training phase and the adversarial training phase. Each `.pkl` file contains a list of model parameters , with a length equal to the number of runs (10).

## References
[1]: Hsiuhan Lexie Yang and Melba M. Crawford, “Domain Adaptation with Preservation of Manifold Geometry for Hyperspectral Image Classification,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 9, no. 2, pp. 543–555, 2016.
