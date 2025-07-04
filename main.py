from utils import build_time_array, get_multi_v_bm, verify_results
from plot_utils import(
    plot_multi_disp_mid,
    plot_heatmap_disp,
    plot_heatmap_bm,
    plot_multi_bm_mid)
from beam import Beam
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ni = np.array([2, 2])
di = np.array([4, 1.5])
dij = np.array([1])

Ptot = 2.8e4 * 9.81
perc_back = 0.7
Pi = np.array([(1-perc_back)*Ptot, perc_back*Ptot])

c = 30
length = 25
E = 3.5e10
h = 0.75
b = 11
J = b*h**3/12
mu = 20897.25
n_modes = 10
nx = 400
nt = 101
damp_ratio = 0
colors = ['red', 'blue']
generate_verify = True
regenerate_sweep = False

data_mat = {
    'length': float(length),
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

my_beam = Beam(length, mu, E, J, damp_ratio, n_modes, nx, nt, Pi, c)
print(my_beam.alpha)
dx = my_beam.x[1] - my_beam.x[0]

omega = np.pi*c/length
alpha = omega/my_beam.return_omega_j(1)

# idxs[0][0] -> P1 entering
my_beam.set_loading_config(di, ni, dij)
my_beam.compute_time_array()
v, bm, tis, vis, bmis, idx_forced = my_beam.compute_multi_response()

# tis[i][j]
# i = 0 -> forces with magnitude P1
# i = 1 -> force with magnitude P2
# j = 0 -> 1st force
# j = 1 -> 2nd force

t_tot = my_beam.t_tot
idxs = my_beam.idxs

plot_multi_disp_mid(t_tot, v, tis, vis, idx_forced, colors)
plot_multi_bm_mid(t_tot, bm, tis, bmis, idx_forced, colors)
plot_heatmap_disp(my_beam.x, t_tot, c, v, idxs, colors, dx)
plot_heatmap_bm(my_beam.x, t_tot, c, bm, idxs, colors, dx)

v0ii = ni * my_beam.v0
v0ii = np.sum(v0ii)

M0ii = ni * my_beam.M0
M0ii = np.sum(M0ii)

DAF0 = np.max(bm[bm.shape[0]//2, :])/M0ii

verify_results(v[v.shape[0]//2, :], bm[bm.shape[0]//2, :], v0ii, M0ii, t_tot)

alpha0 = my_beam.alpha
c0 = c
di0 = di[0]

if regenerate_sweep:
    DAFBM = []
    dis = np.arange(2, 5, 0.5)
    c0s = np.arange(20, 40, 3)

    for dii in dis:
        di[0] = dii
        DAFBM_c = []
        for c in c0s:
            my_beam = Beam(length, mu, E, J, damp_ratio, n_modes, nx, nt, Pi[0], c)
            t_tot, idxs = build_time_array(my_beam, Pi, di, ni, dij, nt, c)
            v, bm, tis, vis, bmis, idx_forced = get_multi_v_bm(my_beam, t_tot, idxs, Pi)
            DAFBM_c.append(np.max(bm[bm.shape[0]//2, :]))

        DAFBM.append(DAFBM_c)

    DAFBM = np.array(DAFBM)/M0ii
    grid_d, grid_c = np.meshgrid(dis, c0s, indexing='ij')

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(grid_d, grid_c/c0 * alpha0, DAFBM, cmap='viridis', alpha=0.7)
    ax.scatter(di0, alpha0, DAF0, color='r', marker='o')

    ax.set_xlabel(r'$d_0$ [m]')
    ax.set_ylabel(r'$\alpha$')
    ax.set_zlabel(r'$DAF_{BM}$')
    ax.grid(True)
    plt.savefig('figs/multi/2Dd0alpha.png')
    plt.close()
