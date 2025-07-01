#!/bin/bash
dataset=clic20-pro-valid
sbatch sbatch-files/run-finetuning.sh config_00 ${dataset}
sbatch sbatch-files/run-finetuning.sh config_01 ${dataset}
sbatch sbatch-files/run-finetuning.sh config_02 ${dataset}
sbatch sbatch-files/run-finetuning.sh config_03 ${dataset}
sbatch sbatch-files/run-finetuning.sh config_04 ${dataset}
sbatch sbatch-files/run-finetuning.sh config_05 ${dataset}
