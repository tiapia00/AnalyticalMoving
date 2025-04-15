from beam import Beam
import numpy as np
import scipy.io
from scipy.interpolate import interp1d

def sweep_alpha(my_beam: Beam, alphas: np.ndarray):
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

def verify_results(v_mid, bm_mid, v0, M0, t):
    mat_ver = scipy.io.loadmat('Verification.mat')
    v_ver = mat_ver['U_xt']
    M_ver = mat_ver['BM_xt']
    node_mid = mat_ver['node_midspan'].item()
    t_ver = mat_ver['t_ver'].squeeze()

    v_ver_mid = v_ver[node_mid, :]
    M_ver_mid = M_ver[node_mid, :]

    interp_order = 'linear'

    v_ver_interp = interp1d(t_ver, -v_ver_mid, kind=interp_order)
    # Fryba uses + for downside displacements
    v_ver_res = v_ver_interp(t)

    M_ver_interp = interp1d(t_ver, M_ver_mid, kind=interp_order)
    M_ver_res = M_ver_interp(t)

    err_v = np.mean((v_ver_res - v_mid)/v0)
    err_M = np.mean((M_ver_res - bm_mid)/M0)

    print(f'err_v = {np.abs(err_v)*100:.2f}%')
    print(f'err_M = {np.abs(err_M)*100:.2f}%')
