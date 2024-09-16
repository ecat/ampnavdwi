# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import os, sys

is_jupyter = hasattr(sys, 'ps1') # https://stackoverflow.com/questions/2356399/tell-if-python-is-in-interactive-mode

    
from utils import calculate_adc_map
from recon_utils import get_window, WindowType, sos
from shot_rejection import PhaseNavNormalizationType, PhaseNavPerVoxelNormalizationType, walsh_method_new, normalize_phasenav_weights
from dwfse_2d_recon_fcns import SensitivityDataSource, calibration_data_sorting, fista_2d_fse_recon, extract_full_kspace
from coil_compression import get_cc_matrix, apply_cc_matrix
from dwfse_2d_recon_manager import add_dwfse_2d_recon_command_line_args_to_parser, dwfse_2d_cflExchanger
from recon_manager import PathManager
from recon_utils import get_fermi_filter
# -

import numpy as np
import cupy as cp
import sigpy as sp
import sigpy.plot as pl
import sigpy.mri as mr
import time
import matplotlib.pyplot as plt
import warnings
import argparse
# %matplotlib widget

# +
if is_jupyter:
    
    #cfl_root_dir = '/home/sdkuser/workspace/data/subject1_n2m0_stab_8nex/'
    #cfl_root_dir = '/home/sdkuser/workspace/data/subject1_n2m1_nostab_15nex/'
    cfl_root_dir = '/home/sdkuser/workspace/data/subject1_n2m0_nostab_15nex/'
    #cfl_root_dir = '/home/sdkuser/workspace/data/subject2_n2m0_stab_8nex/'
    #cfl_root_dir = '/home/sdkuser/workspace/data/subject2_n2m1_nostab_15nex/'
    #cfl_root_dir = '/home/sdkuser/workspace/data/subject2_n2m0_nostab_15nex/'
    
    recon_out_dir = None
    data_dir = None
    device_id = 0 # change this to -1 if you dont have a gpu
    
    args = None
else:
    parser = argparse.ArgumentParser(description='DW fse 2D Reconstruction') 
    parser.add_argument('--data-dir', help='data directory one above cfls', required=True)

    parser = add_dwfse_2d_recon_command_line_args_to_parser(parser)
    
    parser.add_argument('--out-dir', default=None, type=str, required=True)
    
    args = parser.parse_args()

    cfl_root_dir = None
    data_dir = args.data_dir
    device_id = args.device
    recon_out_dir = args.out_dir

if recon_out_dir is None or data_dir is None:
    assert cfl_root_dir is not None, 'Specify root dir for managing cfls'
    print('root_dir ' + cfl_root_dir)

data_dir = PathManager.get_raw_cfl_path(cfl_root_dir) if data_dir is None else data_dir
print('data_dir ' + data_dir)

recon_out_dir = PathManager.get_recon_cfl_path(cfl_root_dir) if recon_out_dir is None else recon_out_dir
print('recon_out_dir ' + recon_out_dir)

cfl_exchanger = dwfse_2d_cflExchanger()

recon_device = sp.Device(device_id)
recon_device.use()
to_recon_device = lambda x: sp.to_device(x, recon_device)
to_cpu = lambda x: sp.to_device(x, sp.cpu_device)
xp = recon_device.xp

print('Data directory: ' + str(cfl_root_dir))

# +
ksp_vz_sxrc, mask_vz_sxy, kspace_cal_z_sxrc, phase_encode_table_esv, phase_encode_table_cal_es, sequence_opts = cfl_exchanger.load_raw_cfls(data_dir)

mask_vzsxy = mask_vz_sxy[:, :, 0, ...]
ksp_vzsxrc = ksp_vz_sxrc[:, :, 0, ...]
kspace_cal_zsxrc = kspace_cal_z_sxrc[:, 0, ...]

nv, nz, ns, nx, fse_etl, nc = ksp_vzsxrc.shape
ny = sequence_opts["matrix_size"][1]
valid_shot_slicer_v = sequence_opts["valid_shots_range"]


ksp_vzsxyc = extract_full_kspace(ksp_vzsxrc, phase_encode_table_esv, ny)
ksp_cal_zsxyc = np.squeeze(extract_full_kspace(kspace_cal_zsxrc[np.newaxis, ...], phase_encode_table_cal_es[..., np.newaxis], ny), axis=0)

del ksp_vz_sxrc, kspace_cal_z_sxrc

do_fermi_filter = True
if do_fermi_filter:
    # do inline so that avoids big copy
    filter_along_y = lambda x: np.multiply(x, np.reshape(get_fermi_filter(ny), (1, 1, 1, -1, 1)), x)
    filter_along_y(ksp_vzsxyc)




# +
start = time.time()

xp = recon_device.xp
print('Run reconstruction with', recon_device)

nv, nz, ns, nx, ny, nc = ksp_vzsxyc.shape
print('matrix size: nv nz ns nx ny nc', nv, nz, ns, nx, ny, nc)
print('ns_valid ', [len(v) for v in valid_shot_slicer_v]) # may not align with ns if montecarlo

# +
# load calibration data
ksp_cal_szxyc, n_cal_shots = calibration_data_sorting(ksp_cal_zsxyc, sequence_opts)

if is_jupyter:
    im_cal_szxyc = sp.ifft(ksp_cal_szxyc, axes=(-2, -3))
    im_cal_szxy = sos(im_cal_szxyc, axis_in=-1)
    im_cal_inner_product_z_xyc = im_cal_szxyc[0, ...] * np.conj(im_cal_szxyc[1, ...])
    sl_to_plot = 0
    pl.ImagePlot(im_cal_szxy[0, ...], z=-3, x=-2, y=-1, title='cal 0')

# +
dcf_vz_xy_ = np.sum(mask_vzsxy[..., np.newaxis], 2, keepdims=True)
dcf_vz_xy_[dcf_vz_xy_ == 0] = 1
ksp_naive_vzxyc = np.sum(ksp_vzsxyc / dcf_vz_xy_, axis=2)
im_naive_vzxyc = sp.ifft(ksp_naive_vzxyc, axes=(-3, -2))
im_naive_vzxy = sos(im_naive_vzxyc, axis_in=-1)

normalize_scale = np.max(np.abs(im_naive_vzxy))
print('normalize scale', normalize_scale)
ksp_vzsxyc = ksp_vzsxyc / normalize_scale

if is_jupyter:
    sl_to_show = range(0, nz, 8)

    for vv in range(nv):
        for sl in sl_to_show:
            im_xy_to_show = im_naive_vzxy[vv, sl, ...]
            pl.ImagePlot(im_xy_to_show / np.max(im_xy_to_show), x=-2, y=-1, 
                        title='density compensated sos recon vol ' + str(vv) + ' all echos ', vmax=.7)
# -

# plot sampling pattern
if is_jupyter:
    for vv in range(nv):
        plt.figure()
        plt.subplot(211)
        plt.imshow(np.squeeze(np.abs(mask_vzsxy[vv, 0, ...])))
        plt.xlabel('shots')
        plt.ylabel('ky')
        plt.subplot(212)    
        plt.plot(np.squeeze(np.sum(np.abs(mask_vzsxy[vv, 0, ...]), 0)), 'x')
        plt.ylim([-.1, None])
        plt.show()

# +
# coil compression
do_cc = True
ncalibx, ncaliby = 24, min(24, sequence_opts["cal_shot_split"] * sequence_opts["etl"]) # only include area guaranteed to be fully sampled
ncc = max(nc // 3, min(8, nc))

if do_cc:
    ksp_center_zxyc = sp.resize(ksp_cal_szxyc[0, ...], (nz, ncalibx, ncaliby, nc))
    ksp_cc_vzsxyc = np.zeros((nv, nz, ns, nx, ny, ncc), dtype=xp.complex64)
    ksp_cc_cal_szxyc = np.zeros((n_cal_shots, nz, nx, ny, ncc), dtype=xp.complex64)
        
    for z in range(nz):
        cc_matrix = get_cc_matrix(ksp_center_zxyc[z, ...], ncc, do_plot=False)
        ksp_cc_vzsxyc[:, z, ...] = apply_cc_matrix(ksp_vzsxyc[:, z, ...], cc_matrix)
        ksp_cc_cal_szxyc[:, z, ...] = apply_cc_matrix(ksp_cal_szxyc[:, z, ...], cc_matrix)

    ksp_vzsxyc = sp.to_device(ksp_cc_vzsxyc, recon_device)
    ksp_cal_szxyc = sp.to_device(ksp_cc_cal_szxyc, recon_device)
    print('Coil compression from ', str(nc), ' to ', str(ncc))
    nc = ncc
else:
    ksp_vzsxyc = sp.to_device(ksp_vzsxyc, recon_device)
    ksp_cal_szxyc = sp.to_device(ksp_cal_szxyc, recon_device)
    print('Skipped coil compression')


ksp_density_compensated_vzxyc = np.sum(sp.to_device(ksp_vzsxyc, sp.cpu_device) / dcf_vz_xy_, axis=2) # keep on cpu to save memory

# +
# estimate coil sensitivity
sens_estimation_flag = SensitivityDataSource.FULL_LOW_BVALUE_IMAGE if sequence_opts["phase_nav_extent"] > 12 else SensitivityDataSource.CALIBRATION_SHOTS
print('Using ' + sens_estimation_flag.name + ' for sensitivity estimation ')
# only phase nav extent is guaranteed to be fully sampled
sens_calib_size = (24, sequence_opts["phase_nav_extent"]) if sens_estimation_flag == SensitivityDataSource.FULL_LOW_BVALUE_IMAGE else (24, ncaliby) 
print('Sens calib size ' + str(sens_calib_size))

# window if using VFA to reduce ringing
do_calib_window = (sequence_opts["vfa_s_tar"] > 0) and sens_estimation_flag == SensitivityDataSource.CALIBRATION_SHOTS
print('Do calib window ' + str(do_calib_window))
cal_data_window_xy = get_window(WindowType.BOX, sens_calib_size, (nx, ny)).reshape(1, 1, 1, nx, ny) if do_calib_window else 1.

if sens_estimation_flag == SensitivityDataSource.FULL_LOW_BVALUE_IMAGE:
    ksp_for_sens_estimation_zcxy = np.transpose(ksp_density_compensated_vzxyc[0, ...], (0, 3, 1, 2))
    ksp_for_sens_estimation_szcxy = np.tile(ksp_for_sens_estimation_zcxy[xp.newaxis, ...], (n_cal_shots, 1, 1, 1, 1))

elif sens_estimation_flag == SensitivityDataSource.CALIBRATION_SHOTS:    
    ksp_for_sens_estimation_szcxy = np.transpose(ksp_cal_szxyc, (0, 1, 4, 2, 3)) 
    
sens_calzcxy = xp.zeros((n_cal_shots, nz, nc, nx, ny), dtype=np.complex64)
for cal in range(n_cal_shots):
    for z in range(nz): 
        sens_calzcxy[cal, z, ...] = mr.app.EspiritCalib(sp.to_device(ksp_for_sens_estimation_szcxy[cal, z, ...], recon_device), 
                                                        calib_width=min(sens_calib_size), device=recon_device, kernel_width=7, crop=0.95, thresh=0.05, show_pbar=False).run()


im_mask_calzxy = sos(sens_calzcxy, axis_in=2) > 0 # compute mask as union of both polarities to make it a bit more accurate
im_mask_zxy = xp.sum(im_mask_calzxy, 0, keepdims=False) == im_mask_calzxy.shape[0]

sens_zcxy_for_recon = sens_calzcxy[-1, ...] * im_mask_zxy[:, np.newaxis, ...]
ksp_cc_zxyc_b0 = sp.to_device(ksp_density_compensated_vzxyc[0, ...], recon_device)
im_conj_phase_zxy_b0 = xp.sum(sp.ifft(ksp_cc_zxyc_b0, axes=(-2, -3)) * xp.conj(xp.transpose(sens_zcxy_for_recon, (0, 2, 3, 1))), axis=-1)

if is_jupyter:
    sl_to_plot = int(nz // 2)
    pl.ImagePlot(sp.ifft(ksp_for_sens_estimation_szcxy[0, sl_to_plot, ...], axes=(-2, -1)), z=-3, x=-2, y=-1, mode='m')

    sens_to_show = to_cpu(sens_calzcxy[0, ...])
    pl.ImagePlot(np.abs(sens_to_show[sl_to_plot, :, ...]), z=-3, x=-2, y=-1, title='magnitude sens')
    pl.ImagePlot(np.angle(sens_to_show[sl_to_plot, :, ...]), z=-3, x=-2, y=-1, title='angle sens')

    pl.ImagePlot(im_mask_zxy, z=-3, x=-2, y=-1)
    pl.ImagePlot(np.abs(im_conj_phase_zxy_b0), z=-3, x=-2, y=-1, vmax=.75*np.max(np.abs(im_conj_phase_zxy_b0)), 
                 title='density compensated conjugate phase coil combination')

# +
# reconstruct phase navigators
phasenav_recon_x, phasenav_recon_y = ny // 2, ny // 2
phasenav_window_x, phasenav_window_y = 32, (2 + sequence_opts['phase_nav_extent'])
phasenav_window_type = WindowType.TRIANGLE


print('Phasenav window ' + str(phasenav_window_type) + ' window size ' + str((phasenav_window_x, phasenav_window_y)) + 
      ' phasenav recon size ' + str((phasenav_recon_x, phasenav_recon_y)))

phasenav_window = sp.to_device(get_window(phasenav_window_type, (phasenav_window_x, phasenav_window_y), (nx, ny)), device=recon_device)

im_phasenav_vzsxy = xp.zeros((nv, nz, ns, nx, ny), dtype=np.complex64)

sens_downsample_zcxy = sp.ifft(sp.resize(sp.fft(sens_zcxy_for_recon, axes=(-2, -1)), (nz, nc, phasenav_recon_x, phasenav_recon_y)), axes=(-2, -1)) 

for v in range(nv):
    valid_shot_slicer = valid_shot_slicer_v[v]
    ksp_zsxyc = ksp_vzsxyc[v, ...]
    # reconstuct only low-res data
    ksp_lowres_zsxyc = sp.resize(ksp_zsxyc, (nz, ns, phasenav_recon_x, phasenav_recon_y, nc))    
    ksp_lowres_zscxy = xp.ascontiguousarray(xp.transpose(ksp_lowres_zsxyc, (0, 1, 4, 2, 3)))

    for z in range(nz):
        sens_downsampled_cxy = sens_downsample_zcxy[z, ...]
        ksp_lowres_scxy = ksp_lowres_zscxy[z, ...]
        for shot in valid_shot_slicer: # only reconstruct valid shots
            ksp_lowres_cxy = ksp_lowres_scxy[shot, ...]
            app = mr.app.SenseRecon(ksp_lowres_cxy, sens_downsampled_cxy, device=recon_device, max_iter=5, show_pbar=False)
            im_phasenav_lowres_xy = app.run()
            ksp_phasenav_lowres_xy = sp.resize(sp.fft(im_phasenav_lowres_xy, axes=(-1, -2)), (nx, ny)) * phasenav_window
            im_phasenav_vzsxy[v, z, shot, ...] = sp.ifft(ksp_phasenav_lowres_xy, axes=(-1, -2))

im_phasenav_vzsxy = im_phasenav_vzsxy * im_mask_zxy[np.newaxis, :, np.newaxis, ...]


# +
im_phasenav_weights_vzsxy = xp.zeros((nv, nz, ns, nx, ny), dtype=np.complex64)
im_normalized_phasenav_weights_vzsxy = xp.zeros((nv, nz, ns, nx, ny), dtype=np.complex64)
phasenav_weight_native_window_shape = (16, 16)
downsample_factor_for_phasenav_weights = 1
nx_phasenav_weight, ny_phasenav_weight = tuple(w // downsample_factor_for_phasenav_weights for w in (nx, ny))
phasenav_weight_window_shape = tuple(w // downsample_factor_for_phasenav_weights for w in phasenav_weight_native_window_shape)
phasenav_weight_window_stride = (2, 2)


for v in range(nv):
    valid_shot_slicer = valid_shot_slicer_v[v]
    ns_valid = len(valid_shot_slicer)
    for z in range(nz):
        # make sure to only use valid shots for calculating weight normalization otherwise scaling will be off
        im_phasenav_sxy = im_phasenav_vzsxy[v, z, valid_shot_slicer, ...] 

        downsample = lambda x: sp.ifft(sp.resize(sp.fft(x, axes=(1, 2)), (ns_valid, nx_phasenav_weight, ny_phasenav_weight)), axes=(1, 2))
        upsample = lambda x: sp.ifft(sp.resize(sp.fft(x, axes=(1, 2)), (ns_valid, nx, ny)), axes=(1, 2))
        im_phasenav_downsampled_sxy = downsample(im_phasenav_sxy)
        im_phasenav_downsampled_weights_sxy, _ = walsh_method_new(im_phasenav_downsampled_sxy, phasenav_weight_window_shape, phasenav_weight_window_stride)
        im_phasenav_weights_sxy = upsample(im_phasenav_downsampled_weights_sxy)
        
        sxy_to__xy_s = sp.linop.Compose((sp.linop.Reshape((1, nx, ny, 1, ns_valid), (nx, ny, ns_valid)), sp.linop.Transpose(im_phasenav_weights_sxy.shape, (1, 2, 0))))
        N_shots_to_scale_against = 1
        im_normalized_phasenav_weights_sxy = sxy_to__xy_s.H * normalize_phasenav_weights(sxy_to__xy_s * im_phasenav_weights_sxy, 
                                                                        PhaseNavNormalizationType.NOOP, 
                                                                        PhaseNavPerVoxelNormalizationType.PERCENTILE_PER_RESPIRATORY_PHASE, 
                                                                        percentiles_per_r=xp.array([N_shots_to_scale_against / ns]))

        im_phasenav_weights_vzsxy[v, z, valid_shot_slicer, ...] = im_phasenav_weights_sxy
        im_normalized_phasenav_weights_vzsxy[v, z, valid_shot_slicer, ...] = im_normalized_phasenav_weights_sxy

im_phasenav_weights_vzsxy = im_phasenav_weights_vzsxy * im_mask_zxy[np.newaxis, :, np.newaxis, ...]  
im_normalized_phasenav_weights_vzsxy = im_normalized_phasenav_weights_vzsxy * im_mask_zxy[np.newaxis, :, np.newaxis, ...]  
# -

if is_jupyter:
    sl_to_plot = nz // 2
    #pl.ImagePlot(im_phasenav_vzsxy[0, sl_to_plot, ...], mode='m', z=-3, x=-2, y=-1, title='b0 phasenav images')
    pl.ImagePlot(im_phasenav_vzsxy[1, sl_to_plot, ...], mode='m', z=-3, x=-2, y=-1, title='dw phasenav images')
    pl.ImagePlot(im_phasenav_weights_vzsxy[1, sl_to_plot, ...], mode='m', z=-3, x=-2, y=-1, title='dw phasenav weights')
    pl.ImagePlot(im_normalized_phasenav_weights_vzsxy[1, sl_to_plot, ...], mode='m', z=-3, x=-2, y=-1, title='dw phasenav weights normalized')
    pl.ImagePlot(im_phasenav_vzsxy[1, sl_to_plot, ...], mode='p', z=-3, x=-2, y=-1, title='dw phasenav phase')

# +
# combine shots recon
im_vzxy_recon = xp.zeros((nv, nz, nx, ny), dtype=xp.complex64)
mask_vzsxy = to_recon_device(mask_vzsxy)

im_phasenav_for_recon_vzsxy = xp.copy(im_phasenav_vzsxy)
im_phasenav_for_recon_vzsxy[0, ...] = xp.ones((nz, ns, nx, ny), dtype=np.complex64)

do_magnitude_shot_weighting = True
lambda_wav_independent = 0 if ((args is None) or (args.lambda_reg < 0)) else args.lambda_reg
print('lambda independent ', lambda_wav_independent)

for vv in range(0, nv):
    # first recon all echos independently
    valid_shot_slicer = valid_shot_slicer_v[vv]

    # need to remove volume first otherwise shots gets bumped to front when indexing with list object, this doesn't happen when using slice object
    ksp_zsxyc = ksp_vzsxyc[vv]
    ksp_zcsxy = xp.transpose(ksp_zsxyc[:, valid_shot_slicer, ...], (0, 4, 1, 2, 3))
    im_phasenav_zsxy = im_phasenav_for_recon_vzsxy[vv][:, valid_shot_slicer, ...]  
    mask_zsxy = mask_vzsxy[vv][:, valid_shot_slicer, ...]      

    if do_magnitude_shot_weighting and vv > 0:
        im_phasenav_magnitude_zsxy = im_normalized_phasenav_weights_vzsxy[vv][:, valid_shot_slicer, ...]
    else:
        im_phasenav_magnitude_zsxy = None

    im_vzxy_recon[vv, ...], _ = fista_2d_fse_recon(ksp_zcsxy, im_phasenav_zsxy, sens_zcxy_for_recon, mask_zsxy, 
                                im_phasenav_magnitude_zsxy, max_iter=50, lamda=lambda_wav_independent)
# -


if is_jupyter:
    pl.ImagePlot(im_vzxy_recon[0, 0::2, ...].get(), z=-3, title='recon b0 images')

    for z in range(1, nz, 2):
        slicer_a = (0, z, slice(None), slice(None))
        slicer_b = (1, z, slice(None), slice(None))
        pl.ImagePlot(np.concatenate((im_vzxy_recon[slicer_a].get(), 2 * im_vzxy_recon[slicer_b].get()), axis=1), title='slice ' + str(z))

# generate adc map
diff_bval = sequence_opts["bvalue"] - sequence_opts["first_volume_bvalue"]
im_vzxy_recon = np.squeeze(np.abs(im_vzxy_recon.get()))
adc_map_zxy = calculate_adc_map(im_vzxy_recon[0, ...], im_vzxy_recon[1, ...], diff_bval) * im_mask_zxy.get()

# +
get_tag = lambda: ''
print('saving to ', recon_out_dir)

cfl_exchanger.save_clinic_cfls(recon_out_dir, list(map(to_cpu, [im_vzxy_recon, adc_map_zxy])))

# save memory by only saving reduced number of shots if reconstructing subsets of data
do_save_reduced_number_of_shots = len(valid_shot_slicer_v[-1]) < ns
vols_to_save = [-1] if do_save_reduced_number_of_shots else slice(None, None, None)
shots_to_save = valid_shot_slicer_v[-1] if do_save_reduced_number_of_shots else slice(None, None, None)
cfl_exchanger.save_debug_cfls(recon_out_dir, list(map(to_cpu, [im_phasenav_vzsxy[vols_to_save, :, shots_to_save, ...], 
                                                im_normalized_phasenav_weights_vzsxy[vols_to_save, :, shots_to_save, ...], 
                                                im_phasenav_weights_vzsxy[vols_to_save, :, shots_to_save, ...],
                                                sens_calzcxy])))
