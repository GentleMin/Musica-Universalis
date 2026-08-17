import os, argparse
import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='+')
    parser.add_argument('-m', type=int)
    args = parser.parse_args()
    print(args)
    filelist = args.file

    spec_list = list()
    labl_list = list()
    for filename in filelist:
        eigvals = np.load(filename)
        if filename[-4:] == 'npz':
            spec_list.append(eigvals['eigi'] - 1j*eigvals['eigr'])
        else:
            spec_list.append(eigvals)
        labl_list.append(filename[:-4])
    styles = [
        {'marker': 'o', 's': 50, 'facecolor': 'none', 'edgecolor': 'tab:blue'},
        {'marker': '+', 's': 50, 'color': 'm'},
        {'marker': 's', 's': 50, 'facecolor': 'none', 'edgecolor': 'blue'},
        {'marker': 'x', 's': 50, 'color': 'r'},
        {'marker': 'D', 's': 50, 'facecolor': 'none', 'edgecolor': 'k'}
    ]

    fig, axes = plt.subplots(nrows=2, figsize=(8, 10), layout='constrained', sharex=True, sharey=True)

    ax = axes[0]
    for iw, w in enumerate(spec_list):
        w_view = w[np.imag(w) < 0]
        ax.scatter(np.abs(np.imag(w_view)), -np.real(w_view), **styles[iw], label=labl_list[iw], zorder=5)
    ax.grid(which='major')
    ax.set_title('Prograde')
    
    ax = axes[1]
    for iw, w in enumerate(spec_list):
        w_view = w[np.imag(w) > 0]
        ax.scatter(np.abs(np.imag(w_view)), -np.real(w_view), **styles[iw], label=labl_list[iw], zorder=5)
    ax.grid(which='major')
    ax.set_title('Retrograde')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_yscale('symlog', linthresh=1e-3)
    ax.autoscale(enable=True, axis='y')
    ax.get_ylim()
    ax.autoscale(enable=False, axis='y')
    # ax.set_xlim(1e+1, 3e+3)
    # ax.set_ylim(1e-1, 1e+3)
    # ax.set_xlim(1e-4, 1e+0)
    # ax.set_ylim(1e-1, 1e+3)
    # ax.set_xlim(1e-1, 1e+1)
    # ax.set_ylim(1e-1, 1e+3)
    
    if args.m is not None:
        l_range = np.arange(args.m, args.m + 16)
        ax.vlines(2*args.m/l_range/(l_range+1), 1e-9, 1e+9, label='Rossby wave frequencies', 
            colors='silver', linestyles='--', zorder=4)
    
    ax.legend(loc=3)
    # figsave(fig, os.path.join(bench_dir, 'spec-QD_m2_Le1.0e-03_Lu1.0e+01_Pm1_stress-free_mc'))
    plt.show()


if __name__ == '__main__':
    main()

