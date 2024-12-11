#!/bin/bash
#SBATCH --mail-type=FAIL # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/jborrell/net_scratch/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/jborrell/net_scratch/jobs/%j.err # where to store error messages
#SBATCH --mem-per-gpu=12G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# only run on tikgpu03: this way we only use 8 gpus.
#SBATCH --exclude=tikgpu10,tikgpu02,tikgpu[04-09],artongpu[01-07],hardin01,lbbgpu01

ETH_USERNAME=jborrell
PROJECT_NAME=cool-chic
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
# DIRECTORY=/itet-stor/${ETH_USERNAME}/home/${PROJECT_NAME}
CONDA_ENVIRONMENT=base
#TODO: change your ETH USERNAME and other stuff from above according + in the #SBATCH output and error the path needs to be double checked!

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
echo 'Failed to create temp directory' >&2
exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log

echo "Running on node: $(hostname)"
echo "In directory: $(pwd)"
echo "Starting on: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"


[[ -f /itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda ]] && eval "$(/itet-stor/${ETH_USERNAME}/net_scratch/conda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENVIRONMENT}
echo "Conda activated"
cd ${DIRECTORY}

# Execute your code
# The source workdir contains a model trained with full parameters and kodim01.
uv run python coolchic/retrain_latents.py --config=cfg/exps/only_latents/vanilla.yaml --source_workdir=results/exps/n_it-grid/n_itr-12000_n_train_loops-5/kodim_config_06/152848_646/

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
