#!/bin/bash

#SBATCH --time=0-00:30:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16G

#SBATCH --job-name=coolchic
#SBATCH --output=slurm_logs/coolchic.out
#SBATCH --error=slurm_logs/coolchic.err

# stack/.2024-06-silent  gcc/12.2.0
# stack/2024-05  gcc/13.2.0
# stack/2024-06  gcc/12.2.0

# This combination of modules worked!
# module load stack/2024-04
# module load gcc/8.5.0

module load stack/2024-06
module load gcc/12.2.0
module load python/3.11.6
module load eth_proxy

# setup

# running
# uv run python coolchic/encode.py --config=cfg/exps/debug/debug.yaml
uv run python coolchic/encode.py --config=cfg/exps/2024-11-12/basic.yaml
