from pathlib import Path
from plot_utils import (
    plot_disp_mid,
    plot_bm_mid,
    plot_disp_mid_tot,
    plot_mode_contr,
    plot_sweep_alpha
    )
import numpy as np
from beam import Beam
from utils import sweep_alpha, verify_results
from scipy.io import savemat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# SI units

# Input data
length = 25
c = 40
T = length/c
P = 1e4
E = 3.5e10
J = 3.8349*0.7
mu = 18358
n_modes = 50
damp_ratio = 0
t_free = 0.5

generate_verify = True
data_mat = {
    'l': float(length),
    'c': float(c),
    'P': float(P),
    'E': float(E),
    'J': float(J),
    'mu': float(mu),
    'damp_ratio': float(damp_ratio),
}


nx = 101
nt = 101

my_beam = Beam(length, mu, E, J, damp_ratio, n_modes, nx, nt, P, c)
print(my_beam.alpha)
v, contr_v = my_beam.get_v(my_beam.alpha, True)
bm, contr_bm = my_beam.get_bm(my_beam.alpha, True)
bm_static = my_beam.get_bm(0)

plot_disp_mid(my_beam, v)
plot_mode_contr(contr_v, 'v')

plot_bm_mid(my_beam, bm, bm_static)
plot_mode_contr(contr_bm, 'BM')

t_free = np.linspace(0, t_free, nt)
v0 = v[:, -1].reshape(-1, 1)
v0_dot = my_beam.get_v_dot(my_beam.alpha)
v_free, bm_free = my_beam.get_free_response(
    v0,
    v0_dot[:, -1].reshape(-1, 1),
    t_free)
plot_disp_mid_tot(my_beam, v, v_free, t_free)

v_mid = v[v.shape[0]//2, :]
bm_mid = bm[bm.shape[0]//2, :]

script_path = r'C:\Users\mattiaan\Documents\MATLAB\VBI-2D'

file_path = Path('Verification.mat')
if file_path.is_file():
    if generate_verify:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        n_els = np.arange(40, 100, 10)
        print(n_els)
        eng.cd(script_path, nargout=0)
        eng.addpath(eng.genpath(script_path))

        errs_v = []
        errs_m = []

        for n_el in n_els:
            data_mat['nel'] = np.float64(n_el)
            savemat('data.mat', data_mat)
            eng.main_single(nargout=0)
            err_v, err_m = verify_results(v_mid, bm_mid, my_beam.v0, my_beam.M0, my_beam.t, True)
            errs_v.append(err_v)
            errs_m.append(err_m)
        eng.quit()

        plt.figure()
        plt.plot(n_els, np.array(errs_v)*100, label='err_v')
        plt.plot(n_els, np.array(errs_m)*100, label='err_BM')
        plt.legend()
        plt.xlabel(r'$n_{ele}$')
        plt.ylabel(r'[%]')
        plt.savefig('figs/single/sens_nele.png')

cs = np.linspace(0, 200)
alphas = cs*np.pi/length/my_beam.return_omega_j(1)
vs_mid, bms_mid, vs_max, bms_max = sweep_alpha(my_beam, damp_ratio, nx, nt, P, cs)
plot_sweep_alpha(vs_mid, bms_mid, vs_max, bms_max, alphas)

#vsver, bmsver = sweep_alpha_matlab(alphas, script_path, data_mat, my_beam.v0, my_beam.M0)
#plot_sweep_alpha_ver(vs_mid, bms_mid, vsver, bmsver, alphas)
