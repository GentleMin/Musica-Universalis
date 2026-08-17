#!/bin/bash
#SBATCH --job-name=op-set
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=small
#SBATCH --output=./jobs/op-set_%A_%a.out

m=$SLURM_ARRAY_TASK_ID
Nr=24
dL=30
L=$(( m + dL ))
mdir="./out/bg_SolarTa_LS18-SH16T9/Mat_m${m}_${Nr}x${L}/ops"

srun python ./mhd_dr_generator.py -res $m $L $Nr -dir $mdir

m=$SLURM_ARRAY_TASK_ID
Nr=30
dL=40
L=$(( m + dL ))
mdir="./out/bg_SolarTa_LS18-SH16T9/Mat_m${m}_${Nr}x${L}/ops"

srun python ./mhd_dr_generator.py -res $m $L $Nr -dir $mdir


