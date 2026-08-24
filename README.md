# Tesis MDS

## About the code and documentation
...
## Installation 

This repository manages two python versions.
- Python 3.12: main env used in experiments of DINOv2 features and model training
- Python 3.10: used exclusively as a local sub-module to extract [dGedi](https://github.com/tev-fbk/dGeDi/) features.

This project was build using the uv package manager, an extremely fast and simple Python package and project manager.

### System requirements
- Tested with CUDA 12.9 for 3.12 env and 11.8 for 3.10 gGedi
- uv for .venv installation
- Windows or linux based OS as macOS is not supported by torch wheels

### Environment Setup 

1. Clone this repository in your local machine. This will clone the dGedi submodule as well. 
```bash
git clone --recurse-submodules https://github.com/Charqican/tesis-mds.git
cd tesis-mds
```
2. Install uv:

on windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

on macOs & Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

More details here [uv installation page](https://docs.astral.sh/uv/getting-started/installation/).

3. Setup the uv python env (3.12):
```bash
uv sync --all-groups
```

4. Submodule Setup: 

[dGedi](https://github.com/tev-fbk/dGeDi/) has its own uv setup. The following is heavily based on the original instructions:
```bash
uv venv external/dgedi_env --python 3.10
source external/dgedi_env/bin/activate

cd external/dgedi
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install numpy==1.26.4 scipy h5py==3.14.0 pyyaml
uv pip install setuptools wheel
uv pip install nina tensorboard==2.11.0 tensorboardX==2.6.2.2 \
    timm==0.6.13 addict einops plyfile termcolor yapf \
    loguru opencv-python-headless==4.10.0.84
uv pip install torch-geometric
uv pip install torch-scatter torch-sparse torch-cluster \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
uv pip install spconv-cu118==2.3.8
uv pip install open3d==0.17.0
uv pip install huggingface_hub tqdm scikit-learn

python download_ckpts.py

deactivate
```

If you are running this in a consumer GPU you probably want to deactivate enable_flash in the dGedi configuration located at external/dgedi/config_dgedi.yaml by changing its value to false.
## Usage

1. The lmo dataset needs to be downloaded separately from the [original source](https://bop.felk.cvut.cz/datasets/). The data processing scripts targets the directory of the dataset using arguments or by configuring an .env file. An .env-example has been provided as an example. 

2. To extract the partial pointclouds from the lmo dataset the following script is provided:
```bash
uv run ./scripts/pose6d_extract_pT.py --dataset DATASET_ROOT --root PREPROCESSING_ROOT
```

3. To propagate the symmetry field from canonical objects to their partial pointclouds the following script is provided:
```bash
uv run ./scripts/pose6d_extract_symmetry_features.py --dataset DATASET_ROOT --root PREPROCESSING_ROOT
```

4. To extract the dGedi features from the partial point clouds the following script is provided:

```bash
external/dgedi_env/bin/python .external/extract_dgedi_features.py
```

5. A marimo notebook can be deployed with:

```bash
uv run marimo edit ./notebooks/Pose6d_symmetry.py
```

### File structure

... 
