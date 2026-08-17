#!/bin/bash

bgdir="./Rotating_SphShells/bg_SolarTa"
Nr=48
dL=48

for m in {1..40}
do
    L=$(( m + dL ))
    matdir="$bgdir/Mat_m${m}_${Nr}x${L}"
    mkdir $matdir/ops
    mv $matdir/*.npz $matdir/ops/.
done

