#!/bin/bash
#SBATCH --mail-type=FAIL # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/jborrell/net_scratch/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/jborrell/net_scratch/jobs/%j.err # where to store error messages
#SBATCH --mem-per-gpu=24GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=tikgpu[02-04],artongpu[01-07]

ETH_USERNAME=jborrell
PROJECT_NAME=cool-chic
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
# DIRECTORY=/itet-stor/${ETH_USERNAME}/home/${PROJECT_NAME}
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


cd ${DIRECTORY}

# Execute your code
# Check that 2 args were provided.
if [[ $# -eq 2 ]]; then
  # Print the command being executed verbatim
  echo "Running command: uv run scripts/run_ablation_evals.py --sweep_path=$1 --dataset=clic20-pro-valid --exp_idx=$2"
  uv run scripts/run_ablation_evals.py --sweep_path=$1 --dataset=clic20-pro-valid --exp_idx=$2
else
  # Check that 1 arg was provided.
  if [[ $# -eq 1 ]]; then
    # Print the command being executed verbatim
    echo "Running command: uv run scripts/run_ablation_evals.py --sweep_path=$1 --dataset=clic20-pro-valid"
    uv run scripts/run_ablation_evals.py --sweep_path=$1 --dataset=clic20-pro-valid
  else
    echo "Illegal number of parameters"
    echo "Usage: $0 sweep_path [exp_idx]"
    exit 1
  fi
fi

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
