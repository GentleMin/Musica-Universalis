# -*- coding: utf-8 -*-


import os
import re
import h5py
import numpy as np
import argparse
import time


matdir_re = re.compile(r'Mat_m([0-9]+)_([0-9]+)x([0-9]+)')
evfile_re = re.compile(r'spec_Ek([^_]+)_Em([^_]+)_Le([^_]+)_Ro([^_]+)_v(\d)-b(\d).npy')


def parse_matdir(matdir):
    match_obj = matdir_re.match(matdir)
    if match_obj is None:
        return None
    meta = {
        # 'bg': match_obj.group(1),
        'N': int(match_obj.group(2)),
        'L': int(match_obj.group(3)),
        'm': int(match_obj.group(1)),
        # 'bc': match_obj.group(5)
    }
    return meta


def parse_evfile(evfile):
    match_obj = evfile_re.match(evfile)
    if match_obj is None:
        return None
    meta = {
        'Ek': float(match_obj.group(1)),
        'Em': float(match_obj.group(2)),
        'Le': float(match_obj.group(3)),
        'Ro': float(match_obj.group(4)),
        'symv': int(match_obj.group(5)),
        'symb': int(match_obj.group(6)),
    }
    return meta


def test_re():
    rootdir = 'Sphere-MHD-spec-36'
    dir_list = os.listdir(rootdir)
    for matdir in dir_list:
        meta = parse_matdir(matdir)
        if meta is None:
            continue
        print(matdir, meta)
        f_list = os.listdir(os.path.join(rootdir, matdir))
        for evfile in f_list:
            meta = parse_evfile(evfile)
            if meta is None:
                continue
            print('\t', evfile, meta)


def save_evs(src, dest, meta_mat, meta_ev):
    ev = np.load(src)
    # print('\n', dest, '\n', flush=True)
    with h5py.File(dest, 'a') as fp:
        path = '/'.join([
            f"Ro{meta_ev['Ro']:.2f}",
            f"Ek{meta_ev['Ek']:.2e}",
            f"Em{meta_ev['Em']:.2e}",
            f"Le{meta_ev['Le']:.2e}",
            f"m{meta_mat['m']:02d}",
            f"sym_v{meta_ev['symv']}_b{meta_ev['symb']}",
            f"{meta_mat['N']}x{meta_mat['L']}"
        ])
        if path in fp:
            saved = False
        else:
            fp.create_dataset(path, data=ev)
            saved = True
    return saved


def combine_spectra(root_dirs, dest):
    for rootdir in root_dirs:
        dir_list = sorted(os.listdir(rootdir))
        print(rootdir)
        for matdir in dir_list:
            meta_mat = parse_matdir(matdir)
            if meta_mat is None:
                continue
            print(matdir)
            f_list = os.listdir(os.path.join(rootdir, matdir))
            for evfile in f_list:
                meta_ev = parse_evfile(evfile)
                if meta_ev is None:
                    continue
                # if meta_ev['Le'] == 1e-3 or meta_ev['Le'] == 1e-1:
                #     continue
                fpath = os.path.join(rootdir, matdir, evfile)
                saved = save_evs(fpath, dest, meta_mat, meta_ev)
                if saved:
                    print(f'\r\t{evfile} saved                   ', flush=True, end='')
                else:
                    print(f'\r\t{evfile} skipped (already exists)', flush=True, end='')
                time.sleep(0.02)
            print()


if __name__ == '__main__':

    # test_re()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs='+')
    parser.add_argument('-o')
    args = parser.parse_args()

    combine_spectra(
        # ['Sphere-Malkus-spec-48', 'Sphere-Malkus-spec-54'],
        # 'Outputs_Stability/spec_sphereMHD_Malkus_SF.h5'
        args.i, args.o
    )
