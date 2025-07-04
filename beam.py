import numpy as np
from typing import Optional, Tuple, List

class Beam:
    def __init__(self, length, mu, E, J, xi, n_modes, nx, nt, Pi, c):
        self.length = length
        self.mu = mu
        self.E = E
        self.J = J
        self.n_modes = n_modes

        self.x = np.linspace(0, length, nx)
        self.nt = nt
        self.t = np.linspace(0, length/c, self.nt)
        self.c = c
        self.omega = self.c * np.pi/length
        self.v0 = 2*Pi*length**3/(np.pi**4*E*J)
        self.M0 = Pi*length/4
        self.alpha = self.omega/self.return_omega_j(1)
        self.omega_d = self.return_omega_j(1) * xi

        self.di: Optional[np.ndarray] = None
        self.ni: Optional[np.ndarray] = None
        self.dij: Optional[np.ndarray] = None

        self.t_tot: Optional[np.ndarray] = None
        self.idxs: Optional[List[Tuple[int, int]]] = None
        self.v: Optional[np.ndarray] = None
        self.bm: Optional[np.ndarray] = None
        self.tis: Optional[List] = None
        self.vis: Optional[List] = None
        self.bmis: Optional[List] = None
        self.idx_forced: Optional[int] = None


    def return_omega_j(self, j):
        omega_j = j ** 2 * np.pi ** 2 / self.length ** 2 * (self.E * self.J / self.mu) ** (1 / 2)
        return omega_j

    def get_v(self, alpha, v0, return_contr: bool = 0):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        v = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        v_contr = []
        for j in range(1, self.n_modes + 1):
            omega_j = self.return_omega_j(j)
            v_j = np.sin(j * np.pi * gridx/self.length)
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

        v *= v0
        v_contr = np.array(v_contr)

        if return_contr:
            return v, v_contr
        else:
            return v


    def get_v_dot(self, alpha, v0):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        v = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        for j in range(1, self.n_modes + 1):
            omega_j = self.return_omega_j(j)
            v_j = np.sin(j * np.pi * gridx/self.length)
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

        v *= v0

        return v


    def set_loading_config(self, di, ni, dij):
        self.di = di
        self.ni = ni
        self.dij = dij


    def compute_time_array(self):
        required = ['di', 'ni', 'dij']
        for attr in required:
            if not hasattr(self, attr):
                raise AttributeError(f"Beam object missing required attribute '{attr}' for time array generation.")

        t0 = 0
        entering = []

        for j in range(len(self.ni)):
            enter = [t0]
            for i in range(self.ni[j] - 1):
                t = t0 + self.di[j] / self.c
                enter.append(t)
                t0 = t
            t0 += self.dij[0] / self.c
            entering.append(enter)

        t0 = self.length / self.c
        exiting = []
        for j in range(len(self.ni)):
            exit = [t0]
            for i in range(self.ni[j] - 1):
                t = t0 + self.di[j] / self.c
                exit.append(t)
                t0 = t
            t0 += self.dij[0] / self.c
            exiting.append(exit)

        if max([max(e) for e in entering]) > min([min(e) for e in exiting]):
            raise SystemExit("Time array cannot be built properly")

        t = []
        for i in range(len(entering)):
            t_frame = np.linspace(entering[i][0], entering[i][1], self.nt)
            t.append(t_frame)
        t_end = t[0][-1]
        t_trans = np.linspace(t_end, t_end + self.dij[0] / self.c, self.nt)
        t.insert(1, t_trans)
        t_tot = np.array(t).reshape(-1)
        idxs_entering = ((0, self.nt), (self.nt * 2, self.nt * 3 - 1))

        t_enex = np.linspace(t_tot[-1], exiting[0][0], self.nt)
        t_tot = np.concatenate((t_tot, t_enex))

        t = []
        for i in range(len(exiting)):
            t_frame = np.linspace(exiting[i][0], exiting[i][1], self.nt)
            t.append(t_frame)
        t_end = t[0][-1]
        t_trans = np.linspace(t_end, t_end + self.dij[0] / self.c, self.nt)
        t.insert(1, t_trans)
        t = np.array(t).reshape(-1)

        idxs_exiting = (np.array([[0, self.nt],
                                  [self.nt * 2, self.nt * 3 - 1]])
                        + idxs_entering[-1][-1] + 1 + self.nt)
        idxs_exiting = tuple(tuple(sublist) for sublist in idxs_exiting.tolist())
        idxs = [idxs_entering, idxs_exiting]

        self.t_tot = np.concatenate((t_tot, t))
        self.idxs = idxs

        return self.t_tot, self.idxs


    def get_bm(self, alpha, M0, return_contr: bool = 0):
        gridx, gridt = np.meshgrid(self.x, self.t, indexing='ij')

        bm = np.zeros_like(gridx)
        beta = self.omega_d / self.return_omega_j(1)

        bm_contr = []
        for j in range(1, self.n_modes+1):
            omega_j = self.return_omega_j(j)
            bm_j = np.sin(j * np.pi * gridx/self.length)
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
                bm_j *= M0

            bm_contr.append(np.mean(bm_j[bm_j.shape[0]//2, :])/self.M0)
            bm += bm_j

        if return_contr:
            return bm, bm_contr
        else:
            return bm


    def compute_multi_response(self):
        if not hasattr(self, 't_tot') or not hasattr(self, 'idxs'):
            raise ValueError("Missing time array or load configuration. "
                             "Call set_loading_config() and compute_time_array() first.")

        assert self.t_tot is not None
        assert self.idxs is not None
        assert self.ni is not None
        assert self.di is not None
        assert self.dij is not None

        v = np.zeros((self.x.shape[0], self.t_tot.shape[0]))
        bm = np.zeros((self.x.shape[0], self.t_tot.shape[0]))
        tis, vis, bmis = [], [], []

        for j in range(len(self.idxs[0])):
            v0 = self.v0[j]
            M0 = self.M0[j]
            tj, vj, bmj = [], [], []

            for i in range(len(self.idxs[0][0])):
                t0 = self.t_tot[self.idxs[0][j][i]]
                t_single = self.t_tot[self.idxs[0][j][i]:self.idxs[1][j][i]] - t0
                idx_forced = len(t_single)
                self.t = t_single

                vi = self.get_v(self.alpha, v0)
                bmi = self.get_bm(self.alpha, M0)

                v[:, self.idxs[0][j][i]:self.idxs[1][j][i]] += vi
                bm[:, self.idxs[0][j][i]:self.idxs[1][j][i]] += bmi

                v0i = vi[:, -1].reshape(-1, 1)
                v0i_dot = self.get_v_dot(self.alpha, v0)[:, -1].reshape(-1, 1)
                vi_free, bm_free = self.get_free_response(
                    v0i, v0i_dot,
                    self.t_tot[self.idxs[1][j][i]:] - self.t_tot[self.idxs[1][j][i]]
                )

                v[:, self.idxs[1][j][i]:] += vi_free
                bm[:, self.idxs[1][j][i]:] += bm_free

                tji = np.concatenate((t_single + t0, self.t_tot[self.idxs[1][j][i]:]))
                tj.append(tji)
                vj.append(np.concatenate((vi, vi_free), axis=1))
                bmj.append(np.concatenate((bmi, bm_free), axis=1))

            tis.append(tj)
            vis.append(vj)
            bmis.append(bmj)

        return v, bm, tis, vis, bmis, idx_forced

    def get_modes_shapes(self):
        phi = np.zeros((len(self.x), self.n_modes))  # Initialize matrix

        for j in range(1, self.n_modes + 1):
            phi[:, j - 1] = np.sin(j * np.pi * self.x / self.length)

        return phi

    def get_free_response(self, v0, v0_dot, t_free):
        phi = self.get_modes_shapes()

        a = np.trapezoid(v0 * phi, self.x, axis=0)
        b = np.trapezoid(v0_dot * phi, self.x, axis=0)

        a *= 2 / self.length
        b *= 2 / self.length

        for j in range(1, b.shape[0] + 1):
            b[j - 1] *= 1 / self.return_omega_j(j)

        gridx, gridt = np.meshgrid(self.x, t_free, indexing='ij')

        v = np.zeros((len(self.x), len(t_free)))
        bm = np.zeros((len(self.x), len(t_free)))

        for j in range(0, a.shape[0]):
            omega_j = self.return_omega_j(j + 1)
            vj = a[j] * np.cos(omega_j * gridt) + b[j] * np.sin(omega_j * gridt)
            vj *= np.sin((j + 1) * np.pi * gridx / self.length)
            v += vj

            bmj = a[j] * np.sin(omega_j * gridt) - b[j] * np.cos(omega_j * gridt)
            bmj *= np.sin((j + 1) * np.pi * gridx / self.length)
            bmj *= (j+1)**2 * np.pi**2 / self.length**2
            bmj *= self.E*self.J
            bm += bmj
        return v, bm
