#!/bin/bash
#SBATCH --job-name=op-set
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=small
#SBATCH --output=./jobs/op-set_%A_%a.out

m=$SLURM_ARRAY_TASK_ID
Nr=48
dL=48
L=$(( m + dL ))
mdir="./Rotating_SphShells/bg_SolarTa/Mat_m${m}_${Nr}x${L}/ops"

srun python ./dbench_generator.py -res $m $L $Nr -dir $mdir

