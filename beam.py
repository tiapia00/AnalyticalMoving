import numpy as np

class Beam:
    def __init__(self, l, mu, E, J, xi, n_modes, nx, nt, load, c):
        self.l = l
        self.mu = mu
        self.E = E
        self.J = J
        self.n_modes = n_modes

        self.x = np.linspace(0, l, nx)
        self.t = np.linspace(0, l/c, nt)
        self.omega = c * np.pi/l
        self.v0 = 2*load*l**3/(np.pi**4*E*J)
        self.M0 = load*l/4
        self.alpha = self.omega/self.return_omega_j(1)
        self.omega_d = self.return_omega_j(1) * xi

    def return_omega_j(self, j):
        omega_j = j ** 2 * np.pi ** 2 / self.l ** 2 * (self.E * self.J / self.mu) ** (1 / 2)
        return omega_j

    def get_v(self, alpha, return_contr: bool = 0):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        v = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        v_contr = []
        for j in range(1, self.n_modes + 1):
            omega_j = self.return_omega_j(j)
            v_j = np.sin(j * np.pi * gridx/self.l)
            if j == alpha:
                if beta != 0:
                    v_j *= (np.exp(-self.omega_d*gridt)*np.sin(
                        j*self.omega*gridt)- j**2/beta * np.cos(j*self.omega*gridt) * (1-
                                                                             np.exp(-self.omega_d*gridt)))
                else:
                    v_j *= (np.sin(j * self.omega * gridt) -
                            j * self.omega * gridt * np.cos(j * self.omega * gridt))
                v_j *= 1/(2*j**4)
            else:
                v_j *= (np.sin(self.omega * j * gridt) -
                        alpha/j * np.exp(-self.omega_d * gridt) * np.sin(omega_j * gridt))
                v_j *= 1/(j**2*(j**2-alpha**2))

            v += v_j
            v_contr.append(np.mean(v_j[v_j.shape[0]//2, :]))

        v *= self.v0
        v_contr = np.array(v_contr)

        if return_contr:
            return v, v_contr
        else:
            return v

    def get_v_dot(self, alpha):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        v = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        for j in range(1, self.n_modes + 1):
            omega_j = self.return_omega_j(j)
            v_j = np.sin(j * np.pi * gridx/self.l)
            if j == alpha:
                if beta != 0:
                    pass
                else:
                    v_j *= (j*self.omega*np.cos(j * self.omega * gridt) -
                            (j * self.omega * np.cos(j * self.omega * gridt) -
                             j**2*self.omega**2*gridt*np.sin(j * self.omega * gridt)))
                v_j *= 1/(2*j**4)
            else:
                v_j *= (j*self.omega*np.cos(self.omega * j * gridt) -
                        alpha/j * (-self.omega_d *np.exp(-self.omega_d * gridt) * np.sin(omega_j * gridt) +
                                   np.exp(-self.omega_d * gridt) * omega_j * np.cos(omega_j * gridt)))
                v_j *= 1/(j**2*(j**2-alpha**2))

            v += v_j

        v *= self.v0

        return v

    def get_bm(self, alpha, return_contr: bool = 0):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        bm = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        bm_contr = []
        for j in range(1, self.n_modes+1):
            omega_j = self.return_omega_j(j)
            bm_j = np.sin(j * np.pi * gridx/self.l)
            if alpha == j:
                if beta != 0:
                    bm_j *= 1/(np.pi**2*j**2)*(np.exp(-self.omega_d*gridt)*np.sin(
                        j*self.omega*gridt)-j**2/beta*np.cos(j*self.omega*gridt)*(1-
                                                                        np.exp(-self.omega_d*gridt)))
                else:
                    bm_j *= 1/(np.pi**2*j**2)*(np.sin(j*self.omega*gridt)-
                                              j*self.omega*gridt*np.cos(j*self.omega*gridt))
                bm_j *= 4*self.M0
            else:
                bm_j *= (np.sin(self.omega * j * gridt)
                        - alpha/j * np.exp(-self.omega_d * gridt) * np.sin(omega_j * gridt))
                bm_j *= 1/(j**2*(1-alpha**2/j**2))
                bm_j *= 8/(np.pi**2)
                bm_j *= self.M0

            bm_contr.append(np.mean(bm_j[bm_j.shape[0]//2, :])/self.M0)
            bm += bm_j

        if return_contr:
            return bm, bm_contr
        else:
            return bm

    def get_modes_shapes(self):
        phi = np.zeros((len(self.x), self.n_modes))  # Initialize matrix

        for j in range(1, self.n_modes + 1):
            phi[:, j - 1] = np.sin(j * np.pi * self.x / self.l)

        return phi

    def get_free_response(self, v0, v0_dot, t_free):
        phi = self.get_modes_shapes()

        a = np.trapezoid(v0 * phi, self.x, axis=0)
        b = np.trapezoid(v0_dot * phi, self.x, axis=0)

        a *= 2 / self.l
        b *= 2 / self.l

        for j in range(1, b.shape[0] + 1):
            b[j - 1] *= 1 / self.return_omega_j(j)

        gridx, gridt = np.meshgrid(self.x, t_free, indexing='ij')
        v = np.zeros((len(self.x), len(t_free)))

        for j in range(0, a.shape[0]):
            omega_j = self.return_omega_j(j + 1)
            vj = a[j] * np.cos(omega_j * gridt) + b[j] * np.sin(omega_j * gridt)
            vj *= np.sin((j + 1) * np.pi * gridx / self.l)
            v += vj
        return v
