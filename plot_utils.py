from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from beam import Beam


def plot_disp_mid(my_beam: Beam, v):
    plt.figure()
    plt.plot(my_beam.x, v[v.shape[0]//2, :])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v_{mid}$')
    plt.savefig('figs/single/Displacement_mid.png')
    plt.close()


def plot_bm_mid(my_beam: Beam, bm, bm_static):
    plt.figure()
    plt.plot(my_beam.x, bm[bm.shape[0]//2, :], label='dynamic')
    plt.plot(my_beam.x, bm_static[bm_static.shape[0]//2, :], label='static')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$BM_{mid}$')
    plt.savefig('figs/single/BM_mid.png')
    plt.close()


def plot_disp_mid_tot(my_beam: Beam, v_force, v_free, t_free):
    plt.figure()
    plt.plot(my_beam.t,
             v_force[v_force.shape[0]//2, :], label='Forced response')
    t0 = my_beam.t[-1]
    plt.plot(t0 + t_free, v_free[v_free.shape[0]//2, :], label='Free response')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v_{mid}$')
    plt.savefig('figs/single/Displacement_tot.png')
    plt.legend()
    plt.close()


def plot_mode_contr(mode_contr, ident: str):
    n_modes = len(mode_contr)
    n_modes = np.arange(1, n_modes+1)
    plt.figure()
    plt.plot(n_modes, mode_contr, 'o')
    plt.xlabel('Mode')
    plt.savefig(f'figs/single/Mode contribution_{ident}.png')
    plt.close()


def plot_sweep_alpha(vs_mid, bms_mid, vs_max, bms_max, alphas):
    plt.figure()
    plt.plot(alphas, vs_mid, label='DAF')
    plt.plot(alphas, vs_max, label='FDAF')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$AF_v$')
    plt.savefig('figs/single/DAF_v.png')
    plt.close()

    plt.figure()
    plt.plot(alphas, bms_mid, label='DAF')
    plt.plot(alphas, bms_max, label='FDAF')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$AF_{BM}$')
    plt.savefig('figs/single/DAF_BM.png')
    plt.close()


def plot_sweep_alpha_ver(vs_mid, bms_mid, vs_ver, bms_ver, alphas):
    plt.figure()
    plt.plot(alphas, vs_mid, label='DAF_Python')
    plt.plot(alphas, vs_ver, label='DAF_Matlab')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$DAF_v$')
    plt.close()

    plt.figure()
    plt.plot(alphas, bms_mid, label='DAF_Python')
    plt.plot(alphas, bms_ver, label='DAF_Matlab')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$DAF_{BM}$')
    plt.show()


def plot_heatmap_disp(x, t_tot, c, v, idxs, colors, dx):
    X, T = np.meshgrid(x, t_tot, indexing='ij')
    # Contour plot x-t
    plt.figure()

    pcm = plt.pcolormesh(T, X, v, shading='auto', cmap='viridis')

    # plot forces lines
    for j in range(len(idxs)):
        color = colors[j]
        for i in range(len(idxs[0])):
            ti = t_tot[idxs[0][j][i]:idxs[1][j][i]]
            plt.plot(ti, c*(ti-ti[0]), '--', color=color)

    loct = np.argmax(v, axis=1)
    locx = np.argmax(loct)
    COP = locx * dx
    plt.plot(t_tot, np.ones_like(t_tot)*COP, '--w', label='COP')

    plt.colorbar(pcm, label=r'$v(x,t)$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.grid(True)
    plt.savefig('figs/single/v_map.png')
    plt.close()


def plot_heatmap_bm(x, t_tot, c, bm, idxs, colors, dx):
    X, T = np.meshgrid(x, t_tot, indexing='ij')
    # Contour plot x-t
    plt.figure()

    pcm = plt.pcolormesh(T, X, bm, shading='auto', cmap='viridis')

    # plot forces lines
    for j in range(len(idxs)):
        color = colors[j]
        for i in range(len(idxs[0])):
            ti = t_tot[idxs[0][j][i]:idxs[1][j][i]]
            plt.plot(ti, c*(ti-ti[0]), '--', color=color)

    # add COP BM

    plt.colorbar(pcm, label=r'$BM(x,t)$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.grid(True)
    plt.savefig('figs/single/BM_map.png')
    plt.close()
