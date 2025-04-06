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
          mu: float):

    # x, t: 1D arrays
    gridx, gridt = np.meshgrid(x, t)
    v = np.zeros_like(gridx)

    for j in range(1, j_end):
        omega_j = return_omega_j(j, E, J, l, mu)
        v_j = np.sin(j*np.pi*gridx/l)*1/(j**2*(j**2-alpha**2))*(
                np.sin(j*omega*gridt) - alpha/j*np.sin(omega_j*gridt))
        v += v_j

    v *= v0
    return v.T

#simply supported beam
#c: speed of the load (m/s)
#P: load
#circular cross-section
if __name__ == "__main__":
    l = 1000    #mm
    c = 300 #mm/s
    T = l/c
    P = 100    #N
    E = 210000  #MPa
    d = 60  #mm
    mu = 3 #kg/mm
    j_end = 40

    omega = np.pi*c/l
    J = np.pi/32*d**4
    v0 = 2*P*l**3/(np.pi**4*E*J)
    alpha = omega/return_omega_j(1, E, J, l, mu)

    x = np.linspace(0, l, 1000)
    t = np.linspace(0, T, 100)
    v = get_v(x, t, j_end, alpha, omega, v0, l, E, J, mu)
    print(f"alpha = {alpha}")

    #Plot midspan displacement in time
    v_mid = v[v.shape[0]//2, :]

    plt.figure()
    plt.plot(t, v_mid)
    plt.xlabel('t[s]')
    plt.ylabel('v_mid[mm]')
    plt.title('Total deflection')
    plt.show()

    #Adimensionalized plot
    v_adim = v_mid/v0
    t_adim = c*t/l
    plt.plot(t_adim, v_adim)
    plt.xlabel("c*t/l")
    plt.ylabel("v_mid/v0")
    plt.xlim((0,1))
    plt.title("Admensional plot")
    plt.show()
    print(f"DAF max: {np.max(v_adim)}")
