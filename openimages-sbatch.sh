#!/bin/bash
for (( i=0; i<=1000; i++ )); do
	job_number=$(printf "%02d" $i)  # Format the number as two digits
	echo "Launching job $job_number"
	sbatch --job-name=kodim${job_number} run-latent-script.sh ${config_path}kodim_config_${job_number}.yaml
done
