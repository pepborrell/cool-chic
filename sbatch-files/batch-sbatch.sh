#!/bin/bash
# Check if the number of arguments is exactly 1
if [ "$#" -ne 1 ]; then
    echo "Error: This script requires the path to the config dir to launch all jobs from."
    echo "Example: $0 cfg/exps/2025-14-42/"
    exit 1
fi
# Format the path to the config dir
if [[ $1 == */ ]]; then
  config_path=$1
else
  config_path="${1}/"
fi
for (( i=0; i<=14; i++ )); do
	job_number=$(printf "%02d" $i)  # Format the number as two digits
	echo "Launching job $job_number"
	sbatch --job-name=kodim${job_number} submit-coolchic-encoding.sh ${config_path}kodim_config_${job_number}.yaml
done
