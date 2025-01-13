#!/bin/bash
for (( i=0; i<=100; i++ )); do
	job_number=$(printf "%02d" $i)  # Format the number as two digits
	echo "Launching job $job_number"
	sbatch submit-simpler-enc.sh cfg/exps/openimages/encode.yaml $i
done
