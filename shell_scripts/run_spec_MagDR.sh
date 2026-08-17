#!/bin/bash
#SBATCH --job-name=spec
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=small
#SBATCH --output=./jobs/spec_%A_%a.out

sym_v=0
sym_b=1
N=48
dL=48
m=$SLURM_ARRAY_TASK_ID
L=$(( m + dL ))
mat_dir="./out/bg_SolarTa_LS18-SH16T9/Mat_m${m}_${N}x${L}"

Ek=1.19e-3
Em=1.19e-3
Le=4.0e-2
Ro=0.00

srun python src/full_spec_MagDR.py \
    -src "$mat_dir/ops" -dest $mat_dir \
    -Ek $Ek -Em $Em -Le $Le -Ro $Ro \
    -symmv $sym_v -symmb $sym_b -N $N -dL $dL -d3op

sym_v=0
sym_b=1
N=54
dL=54
m=$SLURM_ARRAY_TASK_ID
L=$(( m + dL ))
mat_dir="./out/bg_SolarTa_LS18-SH16T9/Mat_m${m}_${N}x${L}"

Ek=1.19e-3
Em=1.19e-3
Le=4.0e-2
Ro=0.00

srun python src/full_spec_MagDR.py \
    -src "$mat_dir/ops" -dest $mat_dir \
    -Ek $Ek -Em $Em -Le $Le -Ro $Ro \
    -symmv $sym_v -symmb $sym_b -N $N -dL $dL -d3op


