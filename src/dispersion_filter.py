# -*- coding: utf-8 -*-

"""
Utility for filtering dispersion results
"""

import os
import h5py, time
import multiprocessing as mp
import numpy as np
from functools import wraps
from .utils import utils


def proc_wrapper(fun):
    @wraps(fun)
    def proc_fun(*args, **kwds):
        pid = os.getpid()
        starttime = time.perf_counter()
        results = fun(*args, **kwds)
        endtime = time.perf_counter()
        proc_info = {'pid': pid, 'elapsed': endtime - starttime}
        return proc_info, results
    return proc_fun


@proc_wrapper
def filter_single_m(fspec, m, mdata, par_arr, verbose=True):
    w_par = list()
    with h5py.File(fspec, 'r') as fp:
        for par in par_arr:
            gp_path = f"Ro{mdata['Ro']:.2f}/Ek{mdata['Ek']:.2e}/Em{mdata['Em']:.2e}/Le{par:.2e}/m{m:02d}/sym_{mdata['symm']}"
            gp = fp[gp_path]
            w_base = gp[f"{mdata['res'][1]}x{m+mdata['res'][0]}"][()]
            w_cmp = gp[f"{mdata['res_cmp'][1]}x{m+mdata['res_cmp'][0]}"][()]
            drift, _ = utils.eigen_drift(w_base, w_cmp)
            w_item = {
                'spec': w_base, 'drift': drift,
                'label': f"m={m:2d}, Le={par:.2e}, symm={mdata['symm']}",
            }
            w_par.append(w_item)
            if verbose:
                # print(f"{w_item['label']} -> {np.sum(w_item['idx'])} eigenvalues filtered.", end='', flush=True)
                print(f"{w_item['label']} finished.", flush=True)
    return m, w_par


@proc_wrapper
def sleep(fspec, m_val, mdata, par_arr, *args, **kwargs):
    time.sleep(5)
    return m_val, 1


def filter_spec_m_multiproc(fspec, mdata, m_arr, par_arr, ncpus=4):

    w_list = [None for _ in m_arr]

    def log_results(result):
        proc_info, (m, w_m) = result
        i_m = np.argmin(np.abs(m - m_arr))
        w_list[i_m] = w_m
        print(f"Computation m = {m:2d} @ PID {proc_info['pid']} finished in {proc_info['elapsed']:.2f}s", flush=True)
    
    pool = mp.Pool(processes=ncpus)
    for m_val in m_arr:
        # print(m_val)
        pool.apply_async(filter_single_m, args=(fspec, m_val, mdata, par_arr), kwds={'verbose': False}, callback=log_results)
        # pool.apply_async(sleep, args=(fspec, m_val, mdata, par_arr), kwds={'verbose': False}, callback=log_results)
    pool.close()
    pool.join()

    return w_list

