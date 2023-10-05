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
## Preparing Data

To prepare the data, you need to create a dataset where each element is a dictionary. The dictionary should have the key "img" and may also contain additional keys like "cls" and "concat" depending on the type of condition. One way to do this is by using MONAI. Below is a sample code snippet:

```python
import monai as mn

train_data_dicts = [
    {"img": "./image1.dcm", "cls": 2},
    {"img": "./image2.dcm", "cls": 0},
    # Add more data dictionaries
]

valid_data_dicts = [
    {"img": "./image9.dcm", "cls": 1},
    # Add more data dictionaries
]

transforms = mn.transforms.Compose([
    mn.transforms.LoadImageD(keys="img"),
    mn.transforms.SelectItemsD(keys=["img","cls"]),
    mn.transforms.ToTensorD(keys=["img","cls"], dtype=torch.float, track_meta=False),
])

train_ds = Dataset(data=train_data_dicts, transform=transforms)  # Add your MONAI transforms here if needed
valid_ds = Dataset(data=valid_data_dicts, transform=transforms)  # For demonstration, using the same data for validation
```

At the end of this step, you should have `train_ds` and `val_ds`.

## Configuring Model

### Configuration Fields Explanation

- **diffusion**: Configurations related to the diffusion process
  - **timesteps**: The number of timesteps in the diffusion process.
  - **schedule_name**: The name of the schedule (e.g., "cosine").
  - **enforce_zero_terminal_snr**: A boolean indicating whether to enforce zero terminal Signal-to-Noise Ratio (SNR).
  - **schedule_params**: Parameters for the schedule.
  - **mean_type** and **var_type**: Types of mean and variance models.
  - **loss_type**: The loss type to use (e.g., "MSE").

- **optimizer**: Optimization configurations.
  - **lr**: The learning rate for the optimizer.
  - **type**: The type of optimizer to use.

- **validation**: Configurations for validation.
  - **classifier_cond_scale**, **protocol**, **log_original**: Various validation-related settings.

- **model**: The configurations for the neural network model.
  - **input_size**, **dims**, **attention_resolutions**, etc: Parameters related to the architecture.

### Instantiating Model

You can instantiate the model using the configuration file and dataset as follows:

```python
from mediffusion import DiffusionModule

model = DiffusionModule(
    "./config.yaml",
    train_ds=train_ds,
    val_ds=valid_ds,
    dl_workers=2,
    train_sampler=None,     # Replace if you're using a custom sampler
    batch_size=32,          # Replace with your train batch size
    val_batch_size=16       # Replace with your validation batch size (recommended size is half of batch_size)
)
```

## Setting up Trainer

Firstly, ensure that the `WANDB_API_KEY` is in your environment variables. You can add it using:

```bash
export WANDB_API_KEY=your-api-key-here
```

Afterward, you can set up the trainer using the `Trainer` class:

```python
from mediffusion import Trainer

trainer = Trainer(
    max_steps=10000,
    val_check_interval=5000,
    root_directory="./outputs",
    precision="16-mixed",
    devices=-1,
    nodes=1,
    wandb_project="Your_Project_Name",
    logger_instance="Your_Logger_Instance",
)
```

## Train!

Finally, to train your model, you simply call:

```python
trainer.fit(model)
```

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