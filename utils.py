from beam import Beam
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def sweep_alpha_mid(my_beam: Beam, alphas: np.ndarray):
    vs_mid = []
    bms_mid = []
    omega_init = my_beam.omega
    for alpha in alphas:
        if alpha != 0:
            my_beam.omega = omega_init * alpha
        # So alpha and its definition stay consistent

        v = my_beam.get_v(alpha)
        bm = my_beam.get_bm(alpha)

        v_mid = v[v.shape[0]//2, :]
        bm_mid = bm[bm.shape[0]//2, :]

        vs_mid.append(np.max(v_mid)/my_beam.v0)
        bms_mid.append(np.max(bm_mid)/my_beam.M0)

    vs_mid = np.array(vs_mid)
    bms_mid = np.array(bms_mid)

    return vs_mid, bms_mid

def sweep_alpha_max(my_beam: Beam, alphas: np.ndarray):
    vs_mid = []
    bms_mid = []
    omega_init = my_beam.omega
    for alpha in alphas:
        if alpha != 0:
            my_beam.omega = omega_init * alpha
        # So alpha and its definition stay consistent

        v = my_beam.get_v(alpha)
        bm = my_beam.get_bm(alpha)

        vs_mid.append(np.max(v)/my_beam.v0)
        bms_mid.append(np.max(bm)/my_beam.M0)

    vs_mid = np.array(vs_mid)
    bms_mid = np.array(bms_mid)

    return vs_mid, bms_mid

def verify_results(v_mid, bm_mid, v0, M0, t):
    mat_ver = scipy.io.loadmat('Verification.mat')
    v_ver = mat_ver['U_xt']
    bm_ver = mat_ver['BM_xt']
    node_mid = mat_ver['node_midspan'].item()
    t_ver = mat_ver['t_ver'].squeeze()

    v_ver_mid = v_ver[node_mid, :]
    bm_ver_mid = bm_ver[node_mid, :]

    interp_order = 'linear'

    v_ver_interp = interp1d(t_ver, -v_ver_mid, kind=interp_order)
    # Fryba uses + for downside displacements
    v_ver_res = v_ver_interp(t)

    bm_ver_interp = interp1d(t_ver, bm_ver_mid, kind=interp_order)
    bm_ver_res = bm_ver_interp(t)

    err_v = np.mean((v_ver_res - v_mid)/v0)
    err_M = np.mean((bm_ver_res - bm_mid)/M0)

    plt.figure()
    plt.plot(t, v_mid/v0, label='calculated')
    plt.plot(t, v_ver_res/v0, label='verification')
    plt.xlabel('t')
    plt.ylabel('v/v0')
    plt.title('Verification mid-span displacement')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t, bm_mid/M0, label='calculated')
    plt.plot(t, bm_ver_res/M0, label='verification')
    plt.xlabel('t')
    plt.ylabel('M/M0')
    plt.title('Verification mid-span BM')
    plt.legend()
    plt.show()

    print(f'err_v = {np.abs(err_v)*100:.2f}%')
    print(f'err_M = {np.abs(err_M)*100:.2f}%')

def build_time_array(my_beam: Beam, Pi, di, ni, dij, nt, c):
    t0 = 0
    entering = []

    # Stack lists
    # each list(enter) represents a Pi
    for j in range(0, len(Pi)):
        enter = [t0]
        for i in range(0, ni[j] - 1):
            t = t0 + di[j] / c
            enter.append(t)
            t0 = t
        t0 += dij[0] / c
        entering.append(enter)

    t0 = my_beam.l / c
    exiting = []
    for j in range(0, len(Pi)):
        exit = [t0]
        for i in range(0, ni[j] - 1):
            t = t0 + di[j] / c
            exit.append(t)
            t0 = t
        t0 += dij[0] / c
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
        t_frame = np.linspace(entering[i][0], entering[i][1], nt)
        t.append(t_frame)
    t_end = t[0][-1]
    t_trans = np.linspace(t_end, t_end + dij[0] / c, nt)
    t.insert(1, t_trans)
    t_tot = np.array(t).reshape(-1)
    idxs_entering = ((0, nt), (nt * 2, nt * 3 - 1))

    t_enex = np.linspace(t_tot[-1], exiting[0][0], nt)
    t_tot = np.concatenate((t_tot, t_enex))
    t = []
    for i in range(len(exiting)):
        t_frame = np.linspace(exiting[i][0], exiting[i][1], nt)
        t.append(t_frame)
    t_end = t[0][-1]
    t_trans = np.linspace(t_end, t_end + dij[0] / c, nt)
    t.insert(1, t_trans)
    t = np.array(t).reshape(-1)
    idxs_exiting = (np.array([[0, nt],
                              [nt * 2, nt * 3 - 1]])
                    + idxs_entering[-1][-1] + 1 + nt)
    idxs_exiting = idxs_exiting.tolist()
    idxs_exiting = tuple(idxs_exiting)
    idxs_exiting = tuple(tuple(sublist) for sublist in idxs_exiting)
    idxs = [idxs_entering, idxs_exiting]

    t_tot = np.concatenate((t_tot, t))

    return t_tot, idxs

def get_multi_v_bm(my_beam: Beam, t_tot, idxs, Pi):
    v = np.zeros((my_beam.x.shape[0], t_tot.shape[0]))
    bm = np.zeros((my_beam.x.shape[0], t_tot.shape[0]))
    v0_init = my_beam.v0
    M0_init = my_beam.M0
    tis = []
    vis = []
    for j in range(len(idxs[0])):
        my_beam.v0 = v0_init * Pi[j]
        my_beam.M0 = M0_init * Pi[j]
        tj = []
        vj = []
        for i in range(len(idxs[0][0])):
            t0 = t_tot[idxs[0][j][i]]
            t_single = t_tot[idxs[0][j][i]:idxs[1][j][i]] - t0
            my_beam.t = t_single
            vi = my_beam.get_v(my_beam.alpha)
            bmi = my_beam.get_bm(my_beam.alpha)

            v[:, idxs[0][j][i]:idxs[1][j][i]] += vi
            bm[:, idxs[0][j][i]:idxs[1][j][i]] += bmi

            v0i = vi[:, -1].reshape(-1,1)
            v0i_dot = my_beam.get_v_dot(my_beam.alpha)[:,-1].reshape(-1,1)
            vi_free, bm_free = my_beam.get_free_response(v0i, v0i_dot, t_tot[idxs[1][j][i]:] - t_tot[idxs[1][j][i]])

            v[:, idxs[1][j][i]:] += vi_free
            bm[:, idxs[1][j][i]:] += bm_free

            tji = np.concatenate((t_single + t0, t_tot[idxs[1][j][i]:]))
            tj.append(tji)
            vji = np.concatenate((vi, vi_free), axis=1)
            vj.append(vji)

            # plt.figure()
            # plt.plot(t_tot[idxs[0][j][i]:idxs[1][j][i]], vi[vi.shape[0]//2, :])
            # plt.plot(t_tot[idxs[1][j][i]:], vi_free[vi_free.shape[0]//2, :])
            # plt.show()

        tis.append(tj)
        vis.append(vj)

    return v, bm, tis, vis
