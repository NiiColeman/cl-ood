#!/bin/bash 
#SBATCH -A IscrC_FoundCL
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00   # format: HH:MM:SS
#SBATCH -N 2                # 1 node
#SBATCH --ntasks-per-node=4 # 4 tasks out of 32
#SBATCH --gres=gpu:2      # 4 gpus per node out of 4
#SBATCH --mem=64GB          # memory per node out of 494000MB 
#SBATCH --job-name=ood_generalization
#SBATCH --output=/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/outs/office_homr_experiments_%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/ratatouille/ood/outs/office_homr_experiments_%j.err

# Load necessary modules
export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"

export PYTHONUNBUFFERED=1
# Activate the virtual environment
source /leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/l2/bin/activate

# Run the main script
# python main.py --config configs/custom_config.yaml
# python experiments/hyperparamsearch.py
python experiments/lora_hyperparameter_search.py
# python experiments/baseline_experiments.py 
# Deactivate the virtual environment
deactivate