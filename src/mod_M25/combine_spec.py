# -*- coding: utf-8 -*-


import os, re, time
import h5py
import numpy as np


file_re = re.compile(r"freq-s-dr([0-9\.]+)-r([0-9\.]+)_([0-9]+)x([0-9]+)x([0-9]+).npz")
def parse_fname(filename):
    match_obj = file_re.match(filename)
    if match_obj is None:
        return None
    meta = {
        'Ro': float(match_obj.group(1)),
        'ro': float(match_obj.group(2)),
        'm': int(match_obj.group(3)),
        'L': int(match_obj.group(4)),
        'N': int(match_obj.group(5)),
        'file': filename,
    }
    return meta

# result_dir = "./results/Incompressible_solarDR"
# result_dir = "./results/Boussinesq_solarDR"
# result_dir = "./results/anelastic_solarDR"
result_dir = "./results/Boussinesq_solarDR_benchmark"

n_ds = 0

for f_name in os.listdir(result_dir):
    meta = parse_fname(f_name)
    if meta is None:
        continue
    w_obj = np.load(os.path.join(result_dir, f_name))
    w = w_obj['eigi'] - 1j*w_obj['eigr']
    # out_file = f"./results/spec-var-Ro-r{meta['ro']:.3f}_incompressible.h5"
    # out_file = f"./results/spec-var-Ro-r{meta['ro']:.3f}_Boussinesq.h5"
    out_file = f"./results/spec-var-Ro-r{meta['ro']:.3f}_Boussinesq_benchmark.h5"
    h5_path = f"Ro{meta['Ro']:.2f}/m{meta['m']:02d}/{meta['L']}x{meta['N']}"
    with h5py.File(out_file, 'a') as fp:
        try:
            fp.create_dataset(h5_path, data=w)
            n_ds += 1
            print(f"\rFile saved: {h5_path}", end='', flush=True)
        except:
            print(f"\rFile skipped; dataset {h5_path} exists!", end='', flush=True)
    time.sleep(0.01)

print(f"\n{n_ds} datasets added.", flush=True)


# for m_name in os.listdir(result_dir):
#     if m_name[0] != 'm':
#         continue
#     m_path = os.path.join(result_dir, m_name)
#     for f_name in os.listdir(m_path):
#         meta = parse_fname(f_name)
#         if meta is None:
#             continue
#         w_obj = np.load(os.path.join(m_path, f_name))
#         w = w_obj['eigi'] - 1j*w_obj['eigr']
#         out_file = f"./results/spec-var-Ro-r{meta['ro']:.3f}_incompressible.h5"
#         h5_path = f"Ek1.00e-04/Ro{meta['Ro']:.2f}/m{meta['m']:02d}/{meta['L']}x{meta['N']}"
#         with h5py.File(out_file, 'a') as fp:
#             try:
#                 fp.create_dataset(h5_path, data=w)
#                 n_ds += 1
#                 print(f"\rFile saved: {h5_path}", end='', flush=True)
#             except:
#                 print(f"\rFile skipped; dataset {h5_path} exists!", end='', flush=True)
#     print(f"\r{m_name} processed, {n_ds} datasets added.", flush=True)

