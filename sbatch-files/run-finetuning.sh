#!/bin/bash
#SBATCH --mail-type=FAIL # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/jborrell/net_scratch/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/jborrell/net_scratch/jobs/%j.err # where to store error messages
#SBATCH --mem-per-gpu=24GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu10,tikgpu[02-03],tikgpu08,artongpu[01-07],hardin01,lbbgpu01

ETH_USERNAME=jborrell
PROJECT_NAME=cool-chic
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}

set -o errexit

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"

cd ${DIRECTORY}

uv run coolchic/hypernet/finetune.py

# Send more noteworthy information to the output log
echo "Finished at: $(date)"
exit 0
