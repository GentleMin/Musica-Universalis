# -*- coding: utf-8 -*-


import os
import numpy as np
from scipy import sparse


def reform_vectors():
    basedir = "./results/Boussinesq_solarDR_benchmark/"
    mat_list = [os.path.join(basedir, fname) for fname in os.listdir(basedir) if fname[:3] == 'Mat']
    for mat_dir in mat_list:
        print(mat_dir)
        vec_list = [os.path.join(mat_dir, fname) for fname in os.listdir(mat_dir) if fname[:4] == 'evec']
        if "perm.npz" not in os.listdir(os.path.join(mat_dir, "ops")):
            continue
        P = sparse.load_npz(os.path.join(mat_dir, "ops/perm.npz"))
        for vec_file in vec_list:
            print(vec_file)
            vec_view = np.load(vec_file)
            vec_view = P.T @ (P.T @ vec_view)
            np.save(vec_file, vec_view)


if __name__ == "__main__":
    reform_vectors()


