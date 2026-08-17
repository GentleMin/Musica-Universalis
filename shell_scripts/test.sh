sym_v=1
sym_b=0
Ek=1.19e-3
Em=1.19e-3
Le=4e-2
m=4
N=24
dL=30

python mhd_dr_scanner.py $m \
    -res $N $dL -symmv $sym_v -symmb $sym_b -Ek $Ek -Em $Em -Le $Le


