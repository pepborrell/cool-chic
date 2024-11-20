#!/bin/bash
# for i in {1..24} # do not run this, as it will run too many tasks
for i in {17..24}
do
	job_number=$(printf "%02d" $i)  # Format the number as two digits
	echo "Launching job $job_number"
	sbatch --job-name=kodim${job_number} submit-job.sh cfg/exps/2024-11-15/kodim${job_number}.yaml
done
