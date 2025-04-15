from utils import build_time_array, build_multi_disp
from plot_utils import plot_multi_disp_mid, plot_heatmap_disp
from beam import Beam
import numpy as np

ni = np.array([2, 2])
di = np.array([1.5, 1.5])
dij = np.array([2])

Pi = np.array([10, 100])

c = 300
l = 25
E = 3.5e10
J = 3.8349
mu = 18358
n_modes = 10
nx = 101
nt = 101
damp_ratio = 0
colors = ['red', 'blue']

my_beam = Beam(l, mu, E, J, damp_ratio, n_modes, nx, nt, Pi[0], c)
dx = my_beam.x[1] - my_beam.x[0]

omega = np.pi*c/l
v0i = Pi*l**3/(np.pi**4*E*J)
M0i = Pi*l/4
alpha = omega/my_beam.return_omega_j(1)

# idxs[0][0] -> P1 entering
t_tot, idxs = build_time_array(my_beam, Pi, di, ni, dij, nt, c)
v, tis, vis = build_multi_disp(my_beam, t_tot, idxs, Pi)
plot_multi_disp_mid(t_tot, v, tis, vis, colors)
plot_heatmap_disp(my_beam.x, t_tot, c, v, idxs, colors, dx)

# tis[i][j]
# i = 0 -> forces with magnitude P1
# i = 1 -> force with magnitude P2
# j = 0 -> 1st force
# j = 1 -> 2nd force