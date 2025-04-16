from plot_utils import plot_disp_mid, plot_bm_mid, plot_disp_mid_tot, plot_mode_contr, plot_sweep_alpha
import numpy as np
from beam import Beam
from utils import sweep_alpha_mid, sweep_alpha_max, verify_results
import matlab.engine
from scipy.io import savemat

# SI units

# Input data
l = 25
c = 300
T = l/c
P = 1e4
E = 3.5e10
J = 3.8349*0.7
mu = 18358
n_modes = 10
damp_ratio = 0
t_free = 0.5

regenerate_ver = False
if regenerate_ver:
    data_mat = {
        'l': float(l),
        'c': float(c),
        'P': float(P),
        'E': float(E),
        'J': float(J),
        'mu': float(mu),
        'damp_ratio': float(damp_ratio)
    }

nx = 101
nt = 101

my_beam = Beam(l, mu, E, J, damp_ratio, n_modes, nx, nt, P, c)
v, contr_v = my_beam.get_v(my_beam.alpha, True)
bm, contr_bm = my_beam.get_bm(my_beam.alpha, True)
bm_static = my_beam.get_bm(0)

plot_disp_mid(my_beam, v)
plot_mode_contr(contr_v, 'v')

plot_bm_mid(my_beam, bm, bm_static)
plot_mode_contr(contr_bm, 'BM')

t_free = np.linspace(0, t_free, nt)
v0 = v[:,-1].reshape(-1,1)
v0_dot = my_beam.get_v_dot(my_beam.alpha)
v_free, bm_free = my_beam.get_free_response(v0, v0_dot[:,-1].reshape(-1,1), t_free)
plot_disp_mid_tot(my_beam, v, v_free, t_free)

alphas = np.linspace(0, 1, 100)
vs_mid, bms_mid = sweep_alpha_mid(my_beam, alphas)
vs_max, bms_max = sweep_alpha_max(my_beam, alphas)
plot_sweep_alpha(vs_mid, bms_mid, vs_max, bms_max, alphas)

v_mid = v[v.shape[0]//2, :]
bm_mid = bm[bm.shape[0]//2, :]

if regenerate_ver:
    script_path=r'C:\Users\mattiaan\Documents\MATLAB\VBI-2D'

    eng = matlab.engine.start_matlab()
    eng.cd(script_path, nargout=0)
    eng.addpath(eng.genpath(script_path))
    eng.main(nargout=0)
    eng.quit()

verify_results(v_mid, bm_mid, my_beam.v0, my_beam.M0, my_beam.t)
