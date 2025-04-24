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

# SI units

# Input data
length = 25
c = 30
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
    'damp_ratio': float(damp_ratio)
}

if generate_verify:
    import matlab.engine
    savemat('data.mat', data_mat)

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
if generate_verify:
    eng = matlab.engine.start_matlab()
    eng.cd(script_path, nargout=0)
    eng.addpath(eng.genpath(script_path))
    eng.main_single(nargout=0)
    eng.quit()

file_path = Path('Verification.mat')
if file_path.is_file():
    verify_results(v_mid, bm_mid, my_beam.v0, my_beam.M0, my_beam.t)

# Careful: here modifying attributes of Beam class
cs = np.linspace(0, 200)
alphas = cs*np.pi/length/my_beam.return_omega_j(1)
vs_mid, bms_mid, vs_max, bms_max = sweep_alpha(my_beam, damp_ratio, nx, nt, P, cs)
plot_sweep_alpha(vs_mid, bms_mid, vs_max, bms_max, alphas)

#vsver, bmsver = sweep_alpha_matlab(alphas, script_path, data_mat, my_beam.v0, my_beam.M0)
#plot_sweep_alpha_ver(vs_mid, bms_mid, vsver, bmsver, alphas)
