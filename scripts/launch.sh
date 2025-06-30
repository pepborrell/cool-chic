# If $2 contains a slash at the end, remove it
if [[ $2 == */ ]]; then
	dir=${2%/}
else
	dir=$2
fi
sbatch $1 ${dir}/config_00.yaml
sbatch $1 ${dir}/config_01.yaml
sbatch $1 ${dir}/config_02.yaml
sbatch $1 ${dir}/config_03.yaml
sbatch $1 ${dir}/config_04.yaml
sbatch $1 ${dir}/config_05.yaml
