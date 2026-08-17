# -*- coding: utf-8 -*-


import os, re
import h5py
import numpy as np


file_re = re.compile(r"freq-dr([0-9\.]+)_([0-9]+)x([0-9]+)x([0-9]+)_v([01]).npy")
def parse_fname(filename):
    match_obj = file_re.match(filename)
    if match_obj is None:
        return None
    meta = {
        'Ro': float(match_obj.group(1)),
        'm': int(match_obj.group(2)),
        'L': int(match_obj.group(3)),
        'N': int(match_obj.group(4)),
        'sym': int(match_obj.group(5)),
        'file': filename,
    }
    return meta

result_dir = "./Rotating_SphShells/hydro_DR_LS18/Incompressible"
out_file = "./Rotating_SphShells/hydro_DR_LS18/Incompressible/spec-var-Ro_incompressible.h5"

n_ds = 0
for m_name in os.listdir(result_dir):
    if m_name[0] != 'm':
        continue
    m_path = os.path.join(result_dir, m_name)
    for f_name in os.listdir(m_path):
        meta = parse_fname(f_name)
        if meta is None:
            continue
        w = np.load(os.path.join(m_path, f_name))
        h5_path = f"Ek1.00e-04/Ro{meta['Ro']:.2f}/m{meta['m']:02d}/symm_{meta['sym']}/{meta['L']}x{meta['N']}"
        with h5py.File(out_file, 'a') as fp:
            fp.create_dataset(h5_path, data=w)
        n_ds += 1
        print(f"\rFile saved: {h5_path}", end='', flush=True)
    print(f"\r{m_name} processed, {n_ds} datasets added.", flush=True)

