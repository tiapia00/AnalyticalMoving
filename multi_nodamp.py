from nodamp import return_omega_j, get_v
import numpy as np
import matplotlib.pyplot as plt

# Input set
# MKS standard units
l = 25
c = 30

ni = [2, 2]
di = [10, 100]
d12 = 100
n_points = 100

Pi = [10, 100]
E = 210000
d = 60
mu = 18358e-3
j_end = 40

omega = np.pi*c/l
J = np.pi/32*d**4
v0i = []
for i in range(len(Pi)):
    v0i.append(2*Pi[i]*l**3/(np.pi**4*E*J))
alpha = omega/return_omega_j(1, E, J, l, mu)
print(f'alpha={alpha}')

# Settings plot
colors = ['red', 'blue']

x = np.linspace(0, l, 1000)

if ni[1] > 0:
    dtot = (ni[0]-1)*di[0] + d12 +  (ni[1]-1)*di[1]
else:
    dtot = (ni[0]-1)*di[0]

T_tot = (l+dtot)/c

t0 = 0
entering = []

# Stack lists
# each list(enter) represents a Pi
for j in range(0, len(Pi)):
    enter = [t0]
    for i in range(0, ni[j]-1):
        t = t0 + di[j]/c
        enter.append(t)
        t0 = t
    t0 += d12/c
    entering.append(enter)

t0 = l/c
exiting = []
for j in range(0, len(Pi)):
    exit = [t0]
    for i in range(0, ni[j]-1):
        t = t0 + di[j]/c
        exit.append(t)
        t0 = t
    t0 += d12/c
    exiting.append(exit)

# time array cannot be built correctly if max(entering) > min(exiting) 
if max(entering) > min(exiting):
    raise SystemExit("Time array cannot be built properly")
"""
plt.figure()
plt.plot(t_single, vi[0][x.shape[0]//2, :])
plt.plot(t_single, vi[1][x.shape[0]//2, :])
plt.show()
"""
t = []
idxs_entering = []
for i in range(len(entering)):
    t_frame = np.linspace(entering[i][0], entering[i][1], n_points)
    t.append(t_frame)
t_end = t[0][-1]
t_trans = np.linspace(t_end, t_end + d12/c, n_points)
t.insert(1, t_trans)
t_tot = np.array(t).reshape(-1)
idxs_entering = ((0, n_points), (n_points*2, n_points*3-1))

t_enex = np.linspace(t_tot[-1], exiting[0][0], n_points)
t_tot = np.concatenate((t_tot, t_enex))
t = []
for i in range(len(exiting)):
    t_frame = np.linspace(exiting[i][0], exiting[i][1], n_points)
    t.append(t_frame)
t_end = t[0][-1]
t_trans = np.linspace(t_end, t_end + d12/c, n_points)
t.insert(1, t_trans)
t = np.array(t).reshape(-1)
idxs_exiting = (np.array([[0, n_points],
    [n_points*2, n_points*3-1]])
    + idxs_entering[-1][-1] + 1 + n_points)
idxs_exiting = idxs_exiting.tolist()
idxs_exiting = tuple(idxs_exiting)
idxs = [idxs_entering, idxs_exiting]

t_tot = np.concatenate((t_tot, t))
v = np.zeros((x.shape[0], t_tot.shape[0]))

# j : force
# i : indexes
# idxs[0][0] -> P1 entering

plt.figure()
for j in range(len(entering)):
    color = colors[j]
    for i in range(len(entering[0])):
        t0 = t_tot[idxs[0][j][i]]
        t_single = t_tot[idxs[0][j][i]:idxs[1][j][i]] - t0
        vi = get_v(x, t_single, j_end, alpha, omega, v0i[j], l, E, J, mu)
        """
        plt.figure()
        plt.plot(t_single, vi[vi.shape[0]//2, :])
        plt.show()
        """
        vmid = vi[vi.shape[0]//2, :]
        plt.plot(t_single + t0, vmid, label=f'$P{j}{i}$', color=color)
        v[:, idxs[0][j][i]:idxs[1][j][i]] += vi
plt.plot(t_tot, v[v.shape[0]//2, :], label='v', color='black')
plt.legend()
# Plot entering for P1
for j in range(len(idxs)):
    color = colors[j]
    for i in range(len(idxs[i])):
        plt.plot(t_tot[idxs[0][j][i]], 0, 'o', color=color)
        plt.plot(t_tot[idxs[1][j][i]], 0, 'o', color=color)

plt.title('Mid-span deformation')
plt.xlabel('$t$')
plt.ylabel(r'$v_{mid}$')
plt.show()

X, T = np.meshgrid(x, t_tot)
# Contour plot x-t
plt.figure()
pcm = plt.pcolormesh(T.T, x.T, v, shading='auto', cmap='viridis')

# plot forces lines
for j in range(len(entering)):
    color = colors[j]
    for i in range(len(entering[0])):
        ti = t_tot[idxs[0][j][i]:idxs[1][j][i]]
        plt.plot(ti, c*(ti-ti[0]), '--', color=color)

plt.colorbar(pcm, label=r'$v(x,t)$')
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
plt.grid(True)
plt.title('2D deformation map')
plt.show()
