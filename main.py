"""
- correct shift? plot single contributions
"""

from load_els import LoadElement, LoadSystem
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

m_axles = np.array([3e2, 3e3])
m_carriage = 3e3
di = np.array([1])
dij = np.array([2])

vehs = []
vehs.append(LoadElement(m_axles, m_carriage, di))
vehs.append(LoadElement(m_axles, m_carriage, di))

load_configuration = LoadSystem(vehs, dij)

c = 10
length = 25
E = 3.5e10
h = 3
b = 11
J = b*h**3/12
mu = 1000
n_modes = 10
nx = 400
nt = 400
damp_ratio = 0
colors = ['red', 'blue']
generate_verify = True
regenerate_sweep = False
g = 9.81

my_beam = Beam(length, mu, E, J, damp_ratio, n_modes, nx, nt, c)
print(my_beam.alpha)
dx = my_beam.x[1] - my_beam.x[0]

omega = np.pi*c/length
alpha = omega/my_beam.return_omega_j(1)

t_global, v, bm = my_beam.compute_multi_response(load_configuration)

plt.figure()
plt.plot(t_global, v[v.shape[0]//2])
plt.xlabel(r'$t$')
plt.ylabel(r'$U_2$')
plt.title('Midspan displacement')
plt.show()

# Comparison with total weight applied at midspan
total_mass = load_configuration.total_mass()

v0_total = my_beam.get_v0(total_mass*g)
M0_total = my_beam.get_M0(total_mass*g)

DAF0 = np.max(bm[bm.shape[0]//2, :])/M0_total

verify_results(v[v.shape[0]//2, :], bm[bm.shape[0]//2, :], v0_total, M0_total, t_global)

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
