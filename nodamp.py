# Given the coordinate and beam parameters, plot the response and the DAF

import numpy as np
import matplotlib.pyplot as plt

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

    v_contr = []
    for j in range(1, j_end):
        omega_j = return_omega_j(j, E, J, l, mu)
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

    for j in range(1, j_end):
        omega_j = return_omega_j(j, E, J, l, mu)
        M_j = np.sin(j*np.pi*gridx/l)*1/(j**2*(1-alpha**2/j**2))*(
                np.sin(j*omega*gridt) - alpha/j*np.exp(-omega_d*gridt)*np.sin(omega_j*gridt))
        M += M_j

    M *= 8/(np.pi)**2*M0
    return M.T

#simply supported beam
#c: speed of the load (m/s)
#P: load
#circular cross-section
if __name__ == "__main__":
    # MKS units
    l = 25
    c = 200
    T = l/c
    P = 1e4
    E = 1.5e10
    d = 2.5
    mu = 18358
    j_end = 20
    damp_ratio = 1e-2

    omega = np.pi*c/l
    J = np.pi/32*d**4
    v0 = 2*P*l**3/(np.pi**4*E*J)
    M0 = P*l/4
    omega0 = return_omega_j(1, E, J, l, mu)
    alpha = omega/omega0

    omega_d = omega0*(1-damp_ratio**2)**(1/2)

    x = np.linspace(0, l, 1000)
    t = np.linspace(0, T, 100)
    v, v_contr = get_v(x, t, j_end, alpha, omega, v0, l, E, J, mu, omega_d, True)
    print(f"alpha = {alpha}")

    #Plot midspan displacement in time
    v_mid = v[v.shape[0]//2, :]

    plt.figure()
    plt.plot(t, v_mid)
    plt.xlabel('t[s]')
    plt.ylabel('v_mid[mm]')
    plt.title('Total deflection')
    plt.show()

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
    print(f"DAF max: {np.max(v_adim)}")

    M = get_M(x, t, j_end, alpha, omega, M0, l, E, J, mu, omega_d)
    M_static = get_M(x, t, j_end, 0, omega, M0, l, E, J, mu, omega_d)
    plt.figure()
    plt.plot(t, M[M.shape[0]//2, :], label='dynamic')
    plt.plot(t, M_static[M.shape[0]//2, :], label='static')
    plt.ticklabel_format(axis='y', scilimits=(0,0))
    plt.xlabel('t')
    plt.ylabel('M')
    plt.title('Mid-span bending moment')
    plt.legend()
    plt.show()
