#!/bin/bash
#SBATCH --job-name=MHDScanRo
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output=./jobs/scanRo_%A_%a.out

sym_v=1
sym_b=0
Ek=1.19e-3
Em=1.19e-3
Le=4e-2
m=$SLURM_ARRAY_TASK_ID
N=30
dL=40

srun python mhd_dr_scanner.py $m \
    -res $N $dL -symmv $sym_v -symmb $sym_b -Ek $Ek -Em $Em -Le $Le


