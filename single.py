# Given the coordinate and beam parameters, plot the response and the DAF

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import interp1d

def return_omega_j(j, E, J, l, mu):
    omega_j = j**2*np.pi**2/l**2*(E*J/mu)**(1/2)
    return omega_j

def get_v(x: np.ndarray,
          t: np.ndarray,
          j_end: int,
          alpha: float,
          omega: float,
          v0: float,
          l: float,
          E: float,
          J: float,
          mu: float,
          omega_d: float = 0,
          return_contr: bool = 0):

    # x, t: 1D arrays
    gridx, gridt = np.meshgrid(x, t)
    v = np.zeros_like(gridx)
    beta = omega_d / return_omega_j(1, E, J, l, mu)

    v_contr = []
    for j in range(1, j_end):
        omega_j = return_omega_j(j, E, J, l, mu)
        if alpha == j:
            v_j = 1/(2*j**4)*(np.exp(-omega_d*gridt)*np.sin(
                    j*omega*gridt)-j**2/beta*np.cos(j*omega*gridt)*(1-
                    np.exp(-omega_d*gridt)))*np.sin(j*np.pi*gridx/l)
        else:
            v_j = np.sin(j*np.pi*gridx/l)*1/(j**2*(j**2-alpha**2))*(
                    np.sin(j*omega*gridt) - alpha/j*np.exp(-omega_d*gridt)*np.sin(omega_j*gridt))

        v += v_j
        v_contr.append(np.mean(v_j.T[v_j.shape[0]//2, :]))

    v *= v0
    v_contr = np.array(v_contr)
    if return_contr:
        return v.T, v_contr
    else:
        return v.T


def get_M(x: np.ndarray,
          t: np.ndarray,
          j_end: int,
          alpha: float,
          omega: float,
          M0: float,
          l: float,
          E: float,
          J: float,
          mu: float,
          omega_d: float):

    gridx, gridt = np.meshgrid(x, t)
    M = np.zeros_like(gridx)
    beta = omega_d / return_omega_j(1, E, J, l, mu)

    for j in range(1, j_end):
        omega_j = return_omega_j(j, E, J, l, mu)
        if alpha == j:
            M_j = 4*M0/(np.pi**2*j**2)*(np.exp(-omega_d*gridt)*np.sin(
                    j*omega*gridt)-j**2/beta*np.cos(j*omega*gridt)*(1-
                    np.exp(-omega_d*gridt)))*np.sin(j*np.pi*gridx/l)
        else:
            M_j = np.sin(j*np.pi*gridx/l)*1/(j**2*(1-alpha**2/j**2))*(
                    np.sin(j*omega*gridt) - alpha/j*np.exp(-omega_d*gridt)*np.sin(omega_j*gridt))
            M_j *= 8/(np.pi)**2*M0

        M += M_j

    return M.T

#simply supported beam
#c: speed of the load (m/s)
#P: load
#circular cross-section
if __name__ == "__main__":
    # MKS units
    l = 25
    c = 30
    T = l/c
    P = 1e4
    E = 3.5e10
    J = 3.8349
    mu = 18358
    j_end = 20
    damp_ratio = 1e-2

    omega = np.pi*c/l
    v0 = 2*P*l**3/(np.pi**4*E*J)
    M0 = P*l/4
    omega0 = return_omega_j(1, E, J, l, mu)
    alpha = omega/omega0

    omega_d = omega0*(1-damp_ratio**2)**(1/2)

    x = np.linspace(0, l, 1000)
    t = np.linspace(0, T, 100)
    v, v_contr = get_v(x, t, j_end, alpha, omega, v0, l, E, J, mu, omega_d, True)
    print(f"alpha = {alpha:.2e}")

    #Plot midspan displacement in time
    v_mid = v[v.shape[0]//2, :]

    plt.figure()
    plt.plot(t, v_mid)
    plt.xlabel('t[s]')
    plt.ylabel('v_mid[mm]')
    plt.title('Total deflection')
    plt.show()

    M = get_M(x, t, j_end, alpha, omega, M0, l, E, J, mu, omega_d)
    M_static = get_M(x, t, j_end, 0, omega, M0, l, E, J, mu, omega_d)
    M_mid = M[M.shape[0]//2, :]
    plt.figure()
    plt.plot(t, M_mid, label='dynamic')
    plt.plot(t, M_static[M.shape[0]//2, :], label='static')
    plt.ticklabel_format(axis='y', scilimits=(0,0))
    plt.xlabel('t')
    plt.ylabel('M')
    plt.title('Mid-span bending moment')
    plt.legend()
    plt.show()

    # Verification
    mat_ver = scipy.io.loadmat('Verification\\alow.mat')
    v_ver = mat_ver['U_xt']
    M_ver = mat_ver['BM_xt']
    node_mid = mat_ver['node_midspan'].item()
    t_ver = mat_ver['t_ver'].squeeze()

    v_ver_mid = v_ver[node_mid, :]
    M_ver_mid = M_ver[node_mid, :]

    v_ver_interp = interp1d(t_ver, -v_ver_mid, kind='linear')
    # Fryba uses + for downside displacements
    v_ver_res = v_ver_interp(t)

    M_ver_interp = interp1d(t_ver, M_ver_mid, kind='linear')
    M_ver_res = M_ver_interp(t)

    err_v = np.mean((v_ver_res - v_mid)/v0)
    err_M = np.mean((M_ver_res - M_mid)/M0)

    print(f'err_v = {np.abs(err_v)*100:.2f}%')
    print(f'err_M = {np.abs(err_M)*100:.2f}%')

    # Plot contribution of each mode
    js = np.arange(v_contr.shape[0])
    plt.figure()
    plt.plot(js, v_contr, 'o')
    plt.xlabel('j')
    plt.ylabel('vj')
    plt.title('Mode contributions')

    #Adimensionalized plot
    v_adim = v_mid/v0
    t_adim = c*t/l
    plt.figure()
    plt.plot(t_adim, v_adim)
    plt.xlabel("c*t/l")
    plt.ylabel("v_mid/v0")
    plt.xlim((0,1))
    plt.title("Admensional plot")
    plt.show()
    print(f"DAF max: {np.max(v_adim):.2f}")

    M_alpha = []
    v_alpha = []

    alpha_arr = np.arange(0, 2, 0.1)
    for alpha in alpha_arr:
        M = get_M(x, t, j_end, alpha, omega, M0, l, E, J, mu, omega_d)/M0
        v = get_v(x, t, j_end, alpha, omega, v0, l, E, J, mu, omega_d)/v0

        M_max = np.max(np.abs(M[M.shape[0]//2, :]))
        v_max = np.max(np.abs(v[v.shape[0]//2, :]))
        M_alpha.append(M_max)
        v_alpha.append(v_max)

    M_alpha = np.array(M_alpha)
    v_alpha = np.array(v_alpha)

    plt.figure()
    plt.plot(alpha_arr, M_alpha)
    plt.ticklabel_format(axis='y', scilimits=(0,0))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$M_{max}/M_0$')
    plt.show()

    plt.figure()
    plt.plot(alpha_arr, v_alpha)
    plt.ticklabel_format(axis='y', scilimits=(0,0))
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$v_{max}/v_0$')
    plt.show()
