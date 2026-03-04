<div align="center">
   
# Artifacts for DiskVecLab
</div>

### Our Code is Open Sourced in https://github.com/alibaba/DiskVecLab.

## Artifacts

This repository contains the evaluation artifacts. The framework source code lives in the `codes` submodule. The repository is organized as follows:

```text
.
├── assets   # README assets
├── codes    # Evaluation framework (submodule)
├── data     # Dataset scripts and resources
├── logs     # Log parsing scripts and example outputs
├── tests    # Batch scripts for running experiments
├── tools    # Utilities (e.g., data processing)
├── .gitignore
├── .gitsubmodules
├── LICENSE
└── README.md
```
## Dataset Preparation

1. Download datasets using the links provided in the `codes` submodule. Most datasets are already in `fbin` or `bin` format.
2. LAION: convert downloaded `npy` files to `fbin`  
   python data/npy_to_fbin.py
3. SPACEV: convert the original dataset to `fbin`  
   python data/spacev_to_fbin.py

## Running the Evaluation

- The evaluation source code is located in the `codes` submodule.
- Batch scripts for running experiments are provided in `tests`.
- Before running, update the paths in the scripts (source code, binaries, and datasets).
- For algorithm details and the full evaluation workflow, refer to the documentation in the `codes` submodule.

## Results Processing

- Scripts for parsing logs and processing evaluation outputs are located in `logs`.
- The `logs` directory also contains example results for reference.
