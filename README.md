# CL-OOD: Efficient Domain Generalization via LoRA Adapter Merging

This project implements efficient domain generalization through LoRA (Low-Rank Adaptation) adapter merging for out-of-distribution (OOD) generalization in computer vision tasks.

## Overview

The project trains domain-specific LoRA adapters on different source domains and learns to merge them optimally for effective generalization with minimal target domain data (as little as 10%). This approach extends the "Model Ratatouille" concept with a more parameter-efficient LoRA-based method.

## Tech Stack

- **Deep Learning**: PyTorch, PEFT (LoRA), timm (Vision Transformers)
- **Base Model**: ViT-Base (`vit_base_patch16_224`)
- **Datasets**: PACS, VLCS, OfficeHome, SVIRO, DomainNet
- **Infrastructure**: SLURM for HPC cluster execution, CUDA for GPU acceleration

## Project Structure

```
cl-ood/
├── configs/                          # Configuration files
│   ├── baseline_configs.yaml
│   ├── class_incremental_merging_loras.yaml
│   ├── custom_config.yaml
│   └── merging_loras.yaml
│
├── data/                             # Dataset handling
│   └── cl_benchmark_generator.py     # Custom dataset loader with augmentation
│
├── experiments/                      # Experiment implementations
│   ├── baselines/                    # Baseline experiments
│   │   ├── baseline_3.py
│   │   ├── baseline_4.py
│   │   ├── baseline5.py              # Main baseline (learned coefficients)
│   │   ├── baseline_4_test.py
│   │   ├── baseline_experiments.py
│   │   ├── baseline_merging_loras.py
│   │   ├── class_inc_merging_loras.py
│   │   └── run_baseline.py
│   │
│   ├── merging/                      # LoRA merging algorithms
│   │   ├── ties.py                   # TIES merging algorithm
│   │   ├── exp.py                    # Alternative TIES implementation
│   │   ├── weighted_avg.py           # Weighted average merging
│   │   └── weight_pruning.py         # Weight pruning methods
│   │
│   └── utils/                        # Training/evaluation utilities
│       ├── train.py
│       ├── train_adapters.py
│       ├── evaluate.py
│       ├── hyperparamsearch.py
│       └── lora_hyperparameter_search.py
│
├── src/                              # Core source code (future utilities)
│
├── scripts/                          # Bash scripts
│   └── run.sh                        # SLURM batch script for HPC
│
├── docs/                             # Documentation and results
│   ├── PACS.md                       # PACS benchmark results
│   ├── PACS.html
│   ├── latest.md                     # Latest experiment results
│   ├── latest.html
│   └── RESULTS.ipynb                 # Jupyter notebook analysis
│
├── outputs/                          # Generated outputs
│   ├── lora_coefficients.png
│   └── ties_lora_results_*.json
│
├── results/                          # Experimental results
├── logs/                             # Training logs
├── outs/                             # SLURM output files
│
└── main.py                           # Main entry point

```

## Usage

### Running Experiments

**Basic usage:**
```bash
python main.py --config configs/custom_config.yaml
```

**Running specific experiments:**
```bash
# Main baseline with learned coefficients
python experiments/baselines/baseline5.py

# TIES merging algorithm
python experiments/merging/ties.py

# Class-incremental learning
python experiments/baselines/class_inc_merging_loras.py
```

**Using SLURM (HPC):**
```bash
sbatch scripts/run.sh
```

## Merging Algorithms

The project implements and compares multiple LoRA merging strategies:

- **Learned Coefficients**: Learns optimal merging weights using target domain data
- **Linear**: Simple average of adapter weights
- **SVD**: Singular value decomposition-based merging
- **TIES**: Trim, Elect Sign & Merge algorithm
- **DARE_TIES**: TIES variant with magnitude-based pruning

## Key Hyperparameters

```yaml
base_model: "vit_base_patch16_224"
batch_size: 32
num_epochs: 5-7
learning_rate: 1e-4 to 1e-5
lora_r: 8
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["qkv", "fc1", "fc2"]
```

## Datasets

The project supports standard domain generalization benchmarks:
- **PACS**: Photo, Art Painting, Cartoon, Sketch
- **VLCS**: VOC2007, LabelMe, Caltech101, SUN09
- **OfficeHome**: Art, Clipart, Product, Real World
- **SVIRO**: Vehicle recognition across domains
- **DomainNet**: Configurable domain setup

## Results

Detailed results and analysis are available in:
- `docs/PACS.md` - PACS benchmark results
- `docs/latest.md` - Latest experimental results
- `docs/RESULTS.ipynb` - Jupyter notebook with visualizations

## License

[Add your license information here]

## Citation

[Add citation information if this is research work]
