import enum as enum
import numpy as np
import sigpy as sp
import cupy as cp
from recon_utils import get_random_circle_shift, sos


class SensitivityDataSource(enum.Enum):
    CALIBRATION_SHOTS = 0
    FULL_LOW_BVALUE_IMAGE = 1


def calibration_data_sorting(ksp_cal_z_sxyc, sequence_opts):
    print(ksp_cal_z_sxyc.shape)

    n_cal_shots = sequence_opts["n_cal_scans"]
    cal_shot_split = sequence_opts["cal_shot_split"]
    print('n_cal_shots = ', n_cal_shots, 'cal_shot_split = ', cal_shot_split)

    ksp_cal_sz_xyc_raw = np.transpose(ksp_cal_z_sxyc, (2, 0, 1, 3, 4, 5))

    ksp_cal_sz_xyc = ksp_cal_sz_xyc_raw[0::cal_shot_split, ...] + ksp_cal_sz_xyc_raw[1::cal_shot_split, ...]
    print(ksp_cal_sz_xyc.shape)

    n_cal_shots = ksp_cal_sz_xyc.shape[0]

    ksp_cal_sz_xyc[1::2, ...] = np.flip(ksp_cal_sz_xyc[1::2, ...], axis=-3)

    return ksp_cal_sz_xyc, n_cal_shots

def fista_2d_fse_recon(ksp_zcs_xy, im_phasenav_zsxy, sens_zcxy, mask_zs_xy, 
                       im_phasenav_magnitude_weights_zsxy=None, do_sample_density_preconditioner=False, max_iter=100, lamda=5e-4):
    
    nz, nc, ns, _, nx, ny = ksp_zcs_xy.shape
    recon_device = sp.get_device(ksp_zcs_xy)

    xp = recon_device.xp

    F = sp.linop.FFT((nc, ns, 1, nx, ny), axes=(-2, -1))

    support_mask_xy = xp.sum(xp.reshape(mask_zs_xy[0, :, 0, ...], (ns, 1, ny)), axis=0)
    samp_index = xp.nonzero(support_mask_xy[0, :])
    support_mask_xy = xp.zeros((nx, ny), xp.float32)
    support_mask_xy[:, xp.min(samp_index[0]):xp.max(samp_index[0])] = 1

    Wav = sp.linop.Wavelet((nx, ny, 1), axes=(0, 1)) 
    im_z_xy_recon = xp.zeros((nz, 1, nx, ny), dtype=np.complex64)
    im_z_xy_adjoint_recon = xp.zeros_like(im_z_xy_recon)
    T = sp.linop.Transpose((nx, ny, 1), axes=(2, 0, 1))

    for z in range(nz):
        phasenav_s_xy = xp.exp(1j * xp.angle(im_phasenav_zsxy[z, ...]))[:, xp.newaxis, :]

        if im_phasenav_magnitude_weights_zsxy is not None:
            phasenav_s_xy = phasenav_s_xy * im_phasenav_magnitude_weights_zsxy[z, :, xp.newaxis, ...]

        P = sp.linop.Multiply((1, nx, ny), phasenav_s_xy)
        sens_c__xy = sens_zcxy[z, ...][:, xp.newaxis, xp.newaxis, ...]
        S = sp.linop.Multiply((ns, 1, nx, ny), sens_c__xy)
        mask_s_xy = mask_zs_xy[z, ...]
        D = sp.linop.Multiply((nc, ns, 1, nx, ny), mask_s_xy)

        if do_sample_density_preconditioner:
            dcf___xy = xp.mean(mask_s_xy, axis=0, keepdims=False)
            dcf___xy[dcf___xy == 0] = 1
            W = sp.linop.Multiply((nc, ns, 1, nx, ny), 1.0 / xp.sqrt(xp.reshape(dcf___xy, (1, 1, 1, 1, ny))))
        else:
            W = sp.linop.Identity((nc, ns, 1, nx, ny))

        E = sp.linop.Compose((W, D, F, S, P, T))        
        EHE = sp.linop.Compose((E.H, E))

        ksp_csexy = xp.ascontiguousarray(ksp_zcs_xy[z, ...])
        Y = W * ksp_csexy
        EHY = E.H * Y
        im_z_xy_adjoint_recon[z, ...] = xp.transpose(EHY, (2, 0, 1))

        max_eigenvalue = sp.app.MaxEig(EHE, device=recon_device, max_iter=30, dtype=xp.complex64, show_pbar=False).run()
        fista_step_size = .99 / max_eigenvalue

        prox_spatial_wav = sp.prox.UnitaryTransform(sp.prox.L1Reg(Wav.oshape, lamda), Wav)

        def proxg_circ_wavelet(alpha, x):
            # average multiple circleshifts
            n_circshifts = 4
            y = 0
            for _ in range(n_circshifts):
                C = get_random_circle_shift((nx, ny, 1), (4, 4, 4))
                Cx = C * x                
                y += C.H * prox_spatial_wav(alpha, Cx)
            return y / n_circshifts

        def gradf(x):
            return E.H * (E * x - Y)

        if lamda > 0:
            x = xp.array(EHY)
            alg = sp.alg.GradientMethod(gradf, x, alpha=fista_step_size, proxg=proxg_circ_wavelet, accelerate=True, max_iter=max_iter)
            while not alg.done():
                alg.update()
        else:
            x = xp.zeros_like(EHY)
            alg = sp.alg.ConjugateGradient(EHE, EHY, x, max_iter=max_iter)
            while not alg.done():
                alg.update()
        
        im_recon = xp.transpose(x, (2, 0, 1))
        im_z_xy_recon[z, ...] = sp.ifft(sp.fft(im_recon, axes=(-1, -2)) * support_mask_xy, axes=(-1, -2))
        
    return im_z_xy_recon, im_z_xy_adjoint_recon


def extract_full_kspace(ksp_vz_sxrc, fse_control_table_esv, ny):
    nv, nz, _, nshots, nx, etl, nc = ksp_vz_sxrc.shape
    ksp_vz_sxyc = np.zeros((nv, nz, 1, nshots, nx, ny, nc), dtype=np.complex64)
    for v in range(nv):
        for s in range(nshots):
            for fse_echo in range(etl):
                pe_index = int(fse_control_table_esv[fse_echo, s, v])
                ksp_vz_sxyc[v, :, :, s, :, pe_index, :] = ksp_vz_sxrc[v, :, :, s, :, fse_echo, :]
    return ksp_vz_sxyc
