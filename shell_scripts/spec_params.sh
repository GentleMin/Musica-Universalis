#!/bin/bash

sym_v=1
sym_b=0
N=48
dL=48
m=1
L=$(( m + dL ))
mat_dir="./Rotating_SphShells/bg_SolarTa/Mat_m${m}_${N}x${L}"


Ek=1.19e-3
Em=1.19e-3
Le=4.0e-2

python src/full_spec.py \
    -src "$mat_dir/ops" -dest $mat_dir \
    -Ek $Ek -Em $Em -Le $Le \
    -symmv $sym_v -symmb $sym_b -N $N -dL $dL -d3op

