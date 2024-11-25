#!/bin/bash
#SBATCH --mail-type=FAIL,END # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/jborrell/net_scratch/jobs/%j.out # where to store the output (%j is the JOBID), subdirectory "jobs" must exist
#SBATCH --error=/itet-stor/jborrell/net_scratch/jobs/%j.err # where to store error messages
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4


ETH_USERNAME=jborrell
PROJECT_NAME=cool-chic
DIRECTORY=/itet-stor/${ETH_USERNAME}/net_scratch/${PROJECT_NAME}
# DIRECTORY=/itet-stor/${ETH_USERNAME}/home/${PROJECT_NAME}
CONDA_ENVIRONMENT=base
mkdir -p ${DIRECTORY}/jobs
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
for i in {1..24}
do
	job_number=$(printf "%02d" $i)  # Format the number as two digits
	echo "Launching job $job_number"
	uv run python coolchic/encode.py --config="cfg/exps/2024-11-15/kodim${job_number}.yaml"
done

# Send more noteworthy information to the output log
echo "Finished at: $(date)"

# End the script with exit code 0
exit 0
