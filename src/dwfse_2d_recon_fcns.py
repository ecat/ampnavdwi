import enum as enum
import numpy as np
import sigpy as sp
import cupy as cp
from recon_utils import get_random_circle_shift, sos


class SensitivityDataSource(enum.Enum):
    CALIBRATION_SHOTS = 0
    FULL_LOW_BVALUE_IMAGE = 1


def calibration_data_sorting(ksp_cal_zsxyc, sequence_opts):
    print(ksp_cal_zsxyc.shape)

    n_cal_shots = sequence_opts["n_cal_scans"]
    cal_shot_split = sequence_opts["cal_shot_split"]
    print('n_cal_shots = ', n_cal_shots, 'cal_shot_split = ', cal_shot_split)

    ksp_cal_szxyc_raw = np.transpose(ksp_cal_zsxyc, (1, 0, 2, 3, 4))

    ksp_cal_szxyc = ksp_cal_szxyc_raw[0::cal_shot_split, ...] + ksp_cal_szxyc_raw[1::cal_shot_split, ...]
    print(ksp_cal_szxyc.shape)

    n_cal_shots = ksp_cal_szxyc.shape[0]

    ksp_cal_szxyc[1::2, ...] = np.flip(ksp_cal_szxyc[1::2, ...], axis=-3)

    return ksp_cal_szxyc, n_cal_shots

def fista_2d_fse_recon(ksp_zcsxy, im_phasenav_zsxy, sens_zcxy, mask_zsxy, 
                       im_phasenav_magnitude_weights_zsxy=None, do_sample_density_preconditioner=False, max_iter=100, lamda=5e-4):
    
    nz, nc, ns, nx, ny = ksp_zcsxy.shape
    recon_device = sp.get_device(ksp_zcsxy)

    xp = recon_device.xp

    F = sp.linop.FFT((nc, ns, nx, ny), axes=(-2, -1))

    support_mask_xy = xp.sum(xp.reshape(mask_zsxy[0, :, 0, ...], (ns, 1, ny)), axis=0)
    samp_index = xp.nonzero(support_mask_xy[0, :])
    support_mask_xy = xp.zeros((nx, ny), xp.float32)
    support_mask_xy[:, xp.min(samp_index[0]):xp.max(samp_index[0])] = 1

    Wav = sp.linop.Wavelet((nx, ny), axes=(0, 1)) 
    im_zxy_recon = xp.zeros((nz, nx, ny), dtype=np.complex64)
    im_zxy_adjoint_recon = xp.zeros_like(im_zxy_recon)

    for z in range(nz):
        phasenav_sxy = xp.exp(1j * xp.angle(im_phasenav_zsxy[z, ...]))

        if im_phasenav_magnitude_weights_zsxy is not None:
            phasenav_sxy = phasenav_sxy * im_phasenav_magnitude_weights_zsxy[z, :, ...]

        P = sp.linop.Multiply((1, nx, ny), phasenav_sxy)
        sens_c_xy = sens_zcxy[z, ...][:, xp.newaxis, ...]
        S = sp.linop.Multiply((ns, nx, ny), sens_c_xy)
        mask_s_xy = mask_zsxy[z, ...]
        D = sp.linop.Multiply((nc, ns, nx, ny), mask_s_xy)

        if do_sample_density_preconditioner:
            dcf___xy = xp.mean(mask_s_xy, axis=0, keepdims=False)
            dcf___xy[dcf___xy == 0] = 1
            W = sp.linop.Multiply((nc, ns, nx, ny), 1.0 / xp.sqrt(xp.reshape(dcf___xy, (1, 1, 1, ny))))
        else:
            W = sp.linop.Identity((nc, ns, nx, ny))

        E = sp.linop.Compose((W, D, F, S, P))        
        EHE = sp.linop.Compose((E.H, E))

        ksp_csexy = xp.ascontiguousarray(ksp_zcsxy[z, ...])
        Y = W * ksp_csexy
        EHY = E.H * Y
        im_zxy_adjoint_recon[z, ...] = EHY

        max_eigenvalue = sp.app.MaxEig(EHE, device=recon_device, max_iter=30, dtype=xp.complex64, show_pbar=False).run()
        fista_step_size = .99 / max_eigenvalue

        prox_spatial_wav = sp.prox.UnitaryTransform(sp.prox.L1Reg(Wav.oshape, lamda), Wav)

        def proxg_circ_wavelet(alpha, x):
            # average multiple circleshifts
            n_circshifts = 4
            y = 0
            for _ in range(n_circshifts):
                C = get_random_circle_shift((nx, ny), (4, 4))
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
        
        im_recon = x
        im_zxy_recon[z, ...] = sp.ifft(sp.fft(im_recon, axes=(-1, -2)) * support_mask_xy, axes=(-1, -2))
        
    return im_zxy_recon, im_zxy_adjoint_recon


def extract_full_kspace(ksp_vzsxrc, fse_control_table_esv, ny):
    nv, nz, nshots, nx, etl, nc = ksp_vzsxrc.shape
    ksp_vzsxyc = np.zeros((nv, nz, nshots, nx, ny, nc), dtype=np.complex64)
    for v in range(nv):
        for s in range(nshots):
            for fse_echo in range(etl):
                pe_index = int(fse_control_table_esv[fse_echo, s, v])
                ksp_vzsxyc[v, :, s, :, pe_index, :] = ksp_vzsxrc[v, :, s, :, fse_echo, :]
    return ksp_vzsxyc
