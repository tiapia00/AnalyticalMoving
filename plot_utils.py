import matplotlib.pyplot as plt
import numpy as np

from beam import Beam

def plot_disp_mid(my_beam: Beam, v):
    plt.figure()
    plt.plot(my_beam.x, v[v.shape[0]//2, :])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v_{mid}$')
    plt.savefig('figs/multi/Displacement at mid-span.png')

def plot_bm_mid(my_beam: Beam, bm, bm_static):
    plt.figure()
    plt.plot(my_beam.x, bm[bm.shape[0]//2, :], label='dynamic')
    plt.plot(my_beam.x, bm_static[bm_static.shape[0]//2, :], label='static')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$BM_{mid}$')
    plt.savefig('figs/multi/BM at mid-span.png')

def plot_disp_mid_tot(my_beam: Beam, v_force, v_free, t_free):
    plt.figure()
    plt.plot(my_beam.t, v_force[v_force.shape[0]//2, :], label='Forced response')
    t0 = my_beam.t[-1]
    plt.plot(t0 + t_free, v_free[v_free.shape[0]//2, :], label='Free response')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v_{mid}$')
    plt.legend()
    plt.savefig('figs/multi/F+F Displacement at mid-span.png')

def plot_mode_contr(mode_contr, ident: str):
    n_modes = len(mode_contr)
    n_modes = np.arange(1, n_modes+1)
    plt.figure()
    plt.plot(n_modes, mode_contr, 'o')
    plt.xlabel('Mode')
    plt.savefig('figs/multi/Mode contribution' + ident + '.png')

def plot_sweep_alpha(vs_mid, bms_mid, vs_max, bms_max, alphas):
    plt.figure()
    plt.plot(alphas, vs_mid, label='DAF')
    plt.plot(alphas, vs_max, label='FDAF')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$DAF_v$')
    plt.show()

    plt.figure()
    plt.plot(alphas, bms_mid, label='DAF')
    plt.plot(alphas, bms_max, label='FDAF')
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$DAF_{BM}$')
    plt.show()

def plot_multi_disp_mid(t_tot, v, tis, vis, idx_forced, colors):
    plt.figure()
    plt.plot(t_tot, v[v.shape[0] // 2, :], label='v', color='black')
    for j in range(len(tis)):
        for i in range(len(tis[j])):
            plt.plot(tis[i][j], vis[i][j][vis[i][j].shape[0] // 2, :], '--', color=colors[i])
            plt.plot(tis[i][j][idx_forced], 0, 'o', color=colors[i])
        # color is related to the magnitude, so associated with the first index
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v_{mid}$')
    plt.savefig('figs/multi/Displacement at midspan.png')

def plot_multi_bm_mid(t_tot, bm, tis, bmis, idx_forced, colors):
    plt.figure()
    plt.plot(t_tot, bm[bm.shape[0] // 2, :], label='bm', color='black')
    for j in range(len(tis)):
        for i in range(len(tis[j])):
            plt.plot(tis[i][j], bmis[i][j][bmis[i][j].shape[0] // 2, :], '--', color=colors[i])
            plt.plot(tis[i][j][idx_forced], 0, 'o', color=colors[i])
        # color is related to the magnitude, so associated with the first index
    plt.xlabel(r'$t$')
    plt.ylabel(r'$BM_{mid}$')
    plt.savefig('figs/multi/BM at midspan.png')

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

    max_index = np.unravel_index(np.argmax(v), v.shape)
    locx = max_index[0]
    COP = locx * dx
    plt.plot(t_tot, np.ones_like(t_tot)*COP, '--w', label='COP')

    plt.colorbar(pcm, label=r'$v(x,t)$')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.grid(True)
    plt.savefig('figs/multi/2D v_mid map.png')

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

    max_index = np.unravel_index(np.argmax(bm), bm.shape)
    locx = max_index[0]
    COP = locx * dx
    plt.plot(t_tot, np.ones_like(t_tot)*COP, '--w', label='COP')

    cbar = plt.colorbar(pcm, label=r'$BM(x,t)$')
    cbar.formatter.set_powerlimits((0, 0))

    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.grid(True)
    plt.savefig('figs/multi/2D BM map.png')
