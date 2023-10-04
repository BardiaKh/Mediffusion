# Mediffusion

Actively being documented. Please check back soon.

## Setup and Installation

### Step 1: Create a Conda Environment

If you haven't installed Conda yet, you can download it from [here](https://docs.anaconda.com/anaconda/install/). After installing, create a new Conda environment by running:

```bash
conda create --name mediffusion python=3.10
```

Activate the environment:

```bash
conda activate mediffusion
```

### Step 2: Install PyTorch

Install PyTorch specifically for CUDA 11.8 by running:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install The Package

You can install the latest version from github using:

```bash
pip install git+https://github.com/BardiaKh/Mediffusion.git -u
```

This will install all the necessary packages.

## Usage


## Citation

If you find this work useful, please consider citing the parent project:

```
@article{KHOSRAVI2023107832,
    title = {Few-shot biomedical image segmentation using diffusion models: Beyond image generation},
    journal = {Computer Methods and Programs in Biomedicine},
    volume = {242},
    pages = {107832},
    year = {2023},
    issn = {0169-2607},
    doi = {https://doi.org/10.1016/j.cmpb.2023.107832},
    url = {https://www.sciencedirect.com/science/article/pii/S0169260723004984},
    author = {Bardia Khosravi and Pouria Rouzrokh and John P. Mickley and Shahriar Faghani and Kellen Mulford and Linjun Yang and A. Noelle Larson and Benjamin M. Howe and Bradley J. Erickson and Michael J. Taunton and Cody C. Wyles},
}
```