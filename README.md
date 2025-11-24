# Evaluation of the VN-EGNN Model for Allosteric Site Prediction

[![](https://img.shields.io/badge/dataset-zenodo-orange?style=plastic&logo=zenodo)](https://zenodo.org/records/17365855)

# Overview

The VN-EGNN model can be found on [GitHub](https://github.com/ml-jku/vnegnn).

This branch is currently WIP. It is designed to evaluate the VN-EGNN model for allosteric site prediction on the ASD (
Allosteric Database) dataset.

# Installation

## Setup

### OS Requirements

Note that I recommend using a Linux-based OS for compatibility and ease of setup. Windows users may consider using
WSL2 (Windows Subsystem for Linux). For more info, see [here](https://learn.microsoft.com/en-us/windows/wsl/install).

### Repository

Clone this repository.

```bash
git clone https://github.com/tobias-nolz/vnegnn-allosteric-sites
````

### Environment

I provide a modified `environment.yml` file for conda to set up the required environment. Create the conda environment
as follows:

```bash
conda env create -f environment.yml
conda activate vnegnn-allosteric-sites
```

In addition, you need perform the following steps, as also described in the original VN-EGNN repository:

1. Install `PyTorch` (with CUDA support if applicable) by following the instructions
   at [pytorch.org](https://pytorch.org/get-started/locally/). Note that in the next step you will need to install
   `PyTorch Geometric` and related packages, which depend on your `PyTorch` version and might not be available for the
   latest `PyTorch` version right away. For example, for `PyTorch 2.9.0` with `CUDA 12.9`, you can run:
    ```bash
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
    ```
2. Install `PyTorch Geometric` and related packages. As noted above, make sure to select the correct versions
   compatible with your `PyTorch` installation. Therefore, first check your installed `PyTorch` version:
    ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
    ```
   Then, install the required packages. An example for `PyTorch 2.8.0` with `CUDA 12.9` is shown below. Adjust the URL
   accordingly for your setup.
   ```bash
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
    ```
   For a list of available versions, see
   the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. Verify the installation by running the following command:
   ```bash
    python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}, PyG: {torch_geometric.__version__}')"
   ```

# Data

## Setup Data

### Allosteric Database

1. Download the ASD dataset from the [Allosteric Database](http://mdl.shsmu.edu.cn/ASD/).

2. Execute the provided script to download the required `protein.pdb` files and pre-process them.
   ```bash
    python scripts/allosteric-sites/setup_data.py \
   --asd-file path/to/ASD_Release_xxxx_AS.txt \
    --output-dir path/to/store/pdb_files \
    --jobs <num_parallel_workers> \
    [--no-skip]
   ```
    
    | Argument       | Short | Type   | Required | Default                 | Description                                                                                                     |
    |----------------|-------|--------|----------|-------------------------|-----------------------------------------------------------------------------------------------------------------|
    | `--asd-file`   | `-a`  | `str`  | ✅ Yes    | —                       | Path to ASD dataset file (CSV/TSV) containing columns `allosteric_pdb`, `modulator_chain`, and `modulator_resi` |
    | `--output-dir` | `-o`  | `str`  | ❌ No     | `data/allosteric-sites` | Directory where allosteric-site data will be stored.                                                            |
    | `--jobs`       | `-j`  | `int`  | ❌ No     | `1`                     | Number of parallel workers                                                                                      |
    | `--no-skip`    | —     | `flag` | ❌ No     | `False`                 | If set, do **not** skip already extracted ligand files                                                          |

This will create a directory structure suitable for evaluation.

### Other Datasets

To compare the results to the original publication, you can also process the other datasets used in the VN-EGNN paper.
The datasets are processed and be downloaded from [this](https://zenodo.org/records/17365855) link. Place the datasets
in the folder ``data/data``. Also note that you will need the `sc-pdb` dataset for training the model.

## Prepare Data for VN-EGNN

Execute the following script to process the data for the VN-EGNN model. The script will use provided scripts from the
VN-EGNN repository to generate esm embeddings and extract the binding info.

```bash
python scripts/process_data.py \
--data-dir path/to/stored/pdb_files \
--jobs <num_parallel_workers> \
--device <cuda_or_cpu> \
--batch <batch_size> \
--threshold <distance_threshold>
```

| Argument      | Short | Type    | Required | Default                 | Description                                                           |
|---------------|-------|---------|----------|-------------------------|-----------------------------------------------------------------------|
| `--data-dir`  | `-p`  | `str`   | ❌ No     | `data/allosteric-sites` | Data folder containing protein subfolders                             |
| `--jobs`      | `-j`  | `int`   | ❌ No     | `1`                     | Number of parallel jobs to pass to scripts                            |
| `--device`    | `-d`  | `str`   | ❌ No     | `"auto"`                | Device to use for ESM embedding generation (`auto`, `cpu`, or `cuda`) |
| `--batch`     | `-b`  | `int`   | ❌ No     | `1`                     | Batch size for ESM embedding generation                               |
| `--threshold` | `-t`  | `float` | ❌ No     | `4.0`                   | Distance threshold for binding site detection                         |

Without parameters, the script will process the data for the allosteric site prediction. For the other datasets, you
need to specify the data directory.

## Data Structure

```
📁 data
├── 📁 allosteric-sites    # Proccced ASD dataset for allosteric site prediction
│   ├── 📁 allosteric       # ASD dataset
│   │   ├── 📁 splits        # Dataset splits
│   │   └── 📁 raw           # Raw data
├── 📁 data                # Processed datasets from VN-EGNN publication
│   ├── 📁 coach420         # COACH420 dataset
│   │   ├── 📁 splits        # Dataset splits
│   │   └── 📁 raw           # Raw data
│   ├── 📁 holo4k          # Holo4K dataset
│   │   ├── 📁 splits        # Dataset splits
│   │   └── 📁 raw           # Raw data
│   ├── 📁 pdbbind2020     # PDBBind2020 dataset
│   │   ├── 📁 splits        # Dataset splits
│   │   └── 📁 raw           # Raw data
│   └── 📁 sc-pdb          # scPDB dataset
│       ├── 📁 splits        # Dataset splits
│       └── 📁 raw           # Raw data
└── 📁 equipocket          # Equipocket dataset for equipocket baseline (optional)
    └── ...
```

# Experiments

Experiment are logged via Weights and Biases, use the [RUN_ID] to evaluate the model. The evaluation metrics are logged
in wandb and can then be exported as csv for further processing.

## Model Training

First, you will have to train the model to obtain the model weights. Everything is configured via Hydra configs and
saved automatically.

```bash
# Train
python src/train.py experiment=vnegnn
```

## Model Evaluation

### Benchmark Datasets

To check your model performance, you can run the evaluation with the same datasets as used in the paper to compare your
results. For this, run

```bash
python src/eval.py wandb_run_id=[RUN_ID]
```

### Allosteric Site Prediction

To evaluate the model on the allosteric site prediction task, run the command with an override for the data config:

```bash
python src/eval.py +data=allosteric wandb_run_id=[RUN_ID]
```

# Project structure

## Configruation

Configuration is managed with Hydra configs, structured as follows.

```

📁 configs
├── 📁 callbacks # Callbacks (e.g. checkpointing, ...)
├── 📁 data # Dataset configs
├── 📁 debug # Debug configs
├── 📁 experiment # Contains all experiments reported in the publication.
├── 📁 extras # Extra configurations.
├── 📁 hydra # Hydra configurations.
├── 📁 local # Local setup files.
├── 📁 logger # Logger setup (wandb logger was used for all experiments)
├── 📁 model # Model configurations
├── 📁 paths # Paths setup.
├── 📁 trainer # Lighting trainer configuration
├── 📄 eval.yaml # Train config.
└── 📄 train.yaml # Eval config.

```

## Source code

The following shows the structure of the source code. The training pipeline is setup
with [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).

```

📁 src
├── 📁 datasets # Dataset implementations
│ ├── 📄 binding_dataset.py # Binding site dataset class
│ ├── 📄 equipocket_dataset.py # Equipocket dataset class
│ └── 📄 utils.py # Dataset utilities
├── 📁 models # Model architectures
│ ├── 📁 equipocket # Equipocket baseline models
│ │ ├── 📄 baseline_models.py # Baseline model implementations
│ │ ├── 📄 egnn_clean.py # Clean EGNN implementation
│ │ ├── 📄 equipocket.py # Equipocket model
│ │ └── 📄 surface_egnn.py # Surface-based EGNN
│ └── 📁 vnegnn # VN-EGNN models
│ ├── 📄 aggregation.py # Aggregation layers
│ ├── 📄 utils.py # Model utilities
│ └── 📄 vnegnn.py # VN-EGNN implementation
├── 📁 modules # Training components
│ ├── 📄 callbacks.py # Custom Lightning callbacks
│ ├── 📄 cluster.py # Clustering utilities
│ ├── 📄 ema.py # Exponential moving average
│ ├── 📄 losses.py # Loss functions
│ ├── 📄 metrics.py # Evaluation metrics
│ └── 📄 schedulers.py # Learning rate schedulers
├── 📁 utils # Utility functions
│ ├── 📄 constants.py # Constants and definitions
│ ├── 📄 graph.py # Graph processing utilities
│ ├── 📄 instantiators.py # Hydra instantiation helpers
│ ├── 📄 logging_utils.py # Logging utilities
│ ├── 📄 misc.py # Miscellaneous utilities
│ ├── 📄 protein.py # Protein processing
│ ├── 📄 pylogger.py # Python logger
│ ├── 📄 rich_utils.py # Rich text formatting
│ ├── 📄 tensor_utils.py # Tensor manipulation
│ ├── 📄 torch_utils.py # PyTorch utilities
│ └── 📄 utils.py # General utilities
├── 📁 wrappers # Lightning module wrappers
│ ├── 📄 base.py # Base wrapper class
│ ├── 📄 bindingsites.py # VNEGNN wrapper
│ └── 📄 equipocket.py # Equipocket wrapper
├── 📄 train.py # Training script
└── 📄 eval.py # Evaluation script

```

# Additional Information from the original VN-EGNN repository

## Citation

```

@misc{sestak2024vnegnn,
title={VN-EGNN: E(3)-Equivariant Graph Neural Networks with Virtual Nodes Enhance Protein Binding Site Identification},
author={Florian Sestak and Lisa Schneckenreiter and Johannes Brandstetter and Sepp Hochreiter and Andreas Mayr and
Günter Klambauer},
year={2024},
eprint={2404.07194},
archivePrefix={arXiv},
primaryClass={cs.LG}
}

```

## License

MIT

## Acknowledgements

The ELLIS Unit Linz, the LIT AI Lab, the Institute for Ma-
chine Learning, are supported by the Federal State Upper
Austria. We thank the projects AI-MOTION (LIT-2018-
6-YOU-212), DeepFlood (LIT-2019-8-YOU-213), Medi-
cal Cognitive Computing Center (MC3), INCONTROL-
RL (FFG-881064), PRIMAL (FFG-873979), S3AI (FFG-
872172), DL for GranularFlow (FFG-871302), EPILEP-
SIA (FFG-892171), AIRI FG 9-N (FWF-36284, FWF-
36235), AI4GreenHeatingGrids(FFG- 899943), INTE-
GRATE (FFG-892418), ELISE (H2020-ICT-2019-3 ID:
951847), Stars4Waters (HORIZON-CL6-2021-CLIMATE-
01-01). We thank Audi.JKU Deep Learning Center,
TGW LOGISTICS GROUP GMBH, Silicon Austria Labs
(SAL), FILL Gesellschaft mbH, Anyline GmbH, Google,
ZF Friedrichshafen AG, Robert Bosch GmbH, UCB Bio-
pharma SRL, Merck Healthcare KGaA, Verbund AG, GLS
(Univ. Waterloo) Software Competence Center Hagen-
berg GmbH, TÜV Austria, Frauscher Sensonic, TRUMPF
and the NVIDIA Corporation. We acknowledge EuroHPC
Joint Undertaking for awarding us access to Karolina at
IT4Innovations, Czech Republic; MeluXina at LuxProvide,
Luxembourg; LUMI at CSC, Finland.

![](visualizations/1odi_3lpk.png)
