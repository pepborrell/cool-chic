#!/bin/bash
for (( i=101; i<=1000; i++ )); do
	if ((i % 100 == 0)); then
		job_number=$(printf "%02d" $i)  # Format the number as two digits
		echo "Launching job $job_number"
	fi
	sbatch submit-simpler-enc.sh cfg/exps/openimages/encode.yaml $i
done
