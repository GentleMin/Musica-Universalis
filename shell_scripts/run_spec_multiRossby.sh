#!/bin/bash
#SBATCH --job-name=spec-var-Ro
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=small
#SBATCH --output=./jobs/spec-Ro_%A_%a.out

m=$SLURM_ARRAY_TASK_ID
Nr=24
dL=32

srun python ./diff_rot_spec.py $m -res $Nr $dL -symm 1

