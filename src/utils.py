import os
import sys

import numpy as np
import sigpy as sp
import cupy as cp
import warnings
import matplotlib.pyplot as plt

def montage(im_xyz, grid_cols=4, normalize=False, title=None, do_show=False):
    assert(im_xyz.ndim == 3), "input must have three dimensions"
    im_xyz = sp.to_device(im_xyz)
    nx, ny, nz = im_xyz.shape  
    if (nz % grid_cols != 0):
        warnings.warn("Number of requested grid columns does not evenly divide nz, some slices will be absent")

    slicer = tuple((slice(None), slice(None), slice(row * grid_cols, min((row+1) * grid_cols, nz))) for row in range(0, nz//grid_cols))    
    im_to_show = np.vstack(tuple(np.reshape(np.transpose(im_xyz[s], (0, 2, 1)), (nx, -1)) for s in slicer))
    scale = np.max(np.abs(np.ravel(im_to_show))) if normalize else 1.
    
    plt.figure()
    plt.imshow(im_to_show / scale, cmap='gray', aspect=1)

    if title is not None:
        plt.title(title)

    if do_show:
        plt.show()

    return im_to_show

def print_and_clear_cupy_memory(do_clear=True):

    mempool = cp.get_default_memory_pool()
    print("Before clear " + str(mempool.used_bytes()))

    if do_clear:
        mempool.free_all_blocks()
        print("After clear " + str(mempool.used_bytes()))

def print_and_clear_cupy_fft_cache(do_print=True, do_clear=False):

    cache = cp.fft.config.get_plan_cache()
    if do_print:
        cache.show_info()

    if do_clear:    
        cache.clear()
        print("Cleared FFT Cache")

def calculate_adc_map(im_xyz_b0, im_xyz_dw, diff_bval):
    
    with np.errstate(divide='ignore', invalid='ignore'):
        im_divide_xyz = np.true_divide(np.abs(im_xyz_dw), np.abs(im_xyz_b0))
        im_log_xyz = np.log(im_divide_xyz)
        im_log_xyz[im_log_xyz == np.inf] = 0
        im_log_xyz = np.nan_to_num(im_log_xyz)
        
    adc_map = np.abs(-1/diff_bval * im_log_xyz)
    adc_map[adc_map > 5e-3] = 0
    
        
    return np.abs(adc_map)    

