from utils import build_time_array, get_multi_v_bm, verify_results
from plot_utils import(
    plot_multi_disp_mid,
    plot_heatmap_disp,
    plot_heatmap_bm,
    plot_multi_bm_mid)
from beam import Beam
import numpy as np
from scipy.io import savemat
import matlab.engine
import os

ni = np.array([2, 2])
di = np.array([4, 1.5])
dij = np.array([1])

Ptot = 2.8e4 * 9.81
perc_back = 0.7
Pi = np.array([(1-perc_back)*Ptot, perc_back*Ptot])

c = 30
l = 25
E = 3.5e10
h = 0.75
b = 11
J = b*h**3/12
mu = 20897.25
n_modes = 50
nx = 101
nt = 101
damp_ratio = 0
colors = ['red', 'blue']
generate_verify = True

data_mat = {
    'l': float(l),
    'c': float(c),
    'P1': float(Pi[0]),
    'P2': float(Pi[1]),
    'd1': float(di[0]),
    'd2': float(di[1]),
    'd12': float(dij[0]),
    'E': float(E),
    'J': float(J),
    'mu': float(mu),
    'damp_ratio': float(damp_ratio)
}
savemat('data_multi.mat', data_mat)

my_beam = Beam(l, mu, E, J, damp_ratio, n_modes, nx, nt, Pi[0], c)
print(my_beam.alpha)
dx = my_beam.x[1] - my_beam.x[0]

omega = np.pi*c/l
v0i = Pi*l**3/(np.pi**4*E*J)
M0i = Pi*l/4
alpha = omega/my_beam.return_omega_j(1)

# idxs[0][0] -> P1 entering
t_tot, idxs = build_time_array(my_beam, Pi, di, ni, dij, nt, c)
v, bm, tis, vis, bmis, idx_forced = get_multi_v_bm(my_beam, t_tot, idxs, Pi)

# tis[i][j]
# i = 0 -> forces with magnitude P1
# i = 1 -> force with magnitude P2
# j = 0 -> 1st force
# j = 1 -> 2nd force

plot_multi_disp_mid(t_tot, v, tis, vis, idx_forced, colors)
plot_multi_bm_mid(t_tot, bm, tis, bmis, idx_forced, colors)
plot_heatmap_disp(my_beam.x, t_tot, c, v, idxs, colors, dx)
plot_heatmap_bm(my_beam.x, t_tot, c, bm, idxs, colors, dx)

if generate_verify:
    script_path = r'C:\Users\mattiaan\Documents\MATLAB\VBI-2D'
    eng = matlab.engine.start_matlab()
    eng.cd(script_path, nargout=0)
    eng.addpath(eng.genpath(script_path))
    eng.main_multi(nargout=0)
    eng.quit()

v0ii = ni * v0i
v0ii = np.sum(v0ii)

M0ii = ni * M0i
M0ii = np.sum(M0ii)
print(M0ii)

verify_results(v[v.shape[0]//2, :], bm[bm.shape[0]//2, :], v0ii, M0ii, t_tot)
