#!/bin/bash
for (( i=0; i<=13; i+=2 )); do
	job_number1=$(printf "%02d" $i)  # Format the number as two digits
	job_number2=$(printf "%02d" $((i+1)))
	echo "Launching job $job_number1 and $job_number2"
	sbatch --job-name=kodim${job_number1} submit-job.sh cfg/exps/2024-11-21/kodim_config_${job_number1}.yaml cfg/exps/2024-11-21/kodim_config_${job_number2}.yaml
done
sbatch --job-name=kodim14 submit-job.sh cfg/exps/2024-11-21/kodim_config_14.yaml
