import matplotlib.pyplot as plt
import numpy as np

from beam import Beam

def plot_disp_mid(my_beam: Beam, v):
    plt.figure()
    plt.plot(my_beam.x, v[v.shape[0]//2, :])
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v_{mid}$')
    plt.title('Displacement at mid-span')
    plt.show()

def plot_bm_mid(my_beam: Beam, bm, bm_static):
    plt.figure()
    plt.plot(my_beam.x, bm[bm.shape[0]//2, :], label='dynamic')
    plt.plot(my_beam.x, bm_static[bm_static.shape[0]//2, :], label='static')
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    plt.xlabel(r'$t$')
    plt.ylabel(r'$BM_{mid}$')
    plt.title('BM at mid-span')
    plt.show()

def plot_disp_mid_tot(my_beam: Beam, v_force, v_free, t_free):
    plt.figure()
    plt.plot(my_beam.t, v_force[v_force.shape[0]//2, :], label='Forced response')
    t0 = my_beam.t[-1]
    plt.plot(t0 + t_free, v_free[v_free.shape[0]//2, :], label='Free response')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$v_{mid}$')
    plt.title('Displacement at mid-span')
    plt.legend()
    plt.show()

def plot_mode_contr(mode_contr, ident: str):
    n_modes = len(mode_contr)
    n_modes = np.arange(1, n_modes+1)
    plt.figure()
    plt.plot(n_modes, mode_contr, 'o')
    plt.xlabel('Mode')
    plt.title('Mode contribution ' + ident)
    plt.show()

def plot_sweep_alpha(vs_mid, bms_mid, alphas):
    plt.figure()
    plt.plot(alphas, vs_mid)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$DAF_v$')
    plt.show()

    plt.figure()
    plt.plot(alphas, bms_mid)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$DAF_{BM}$')
    plt.show()
