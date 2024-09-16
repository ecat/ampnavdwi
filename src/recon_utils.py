import numpy as np
import sigpy as sp
import cv2
import random
import enum
import warnings
from scipy import stats


def get_random_circle_shift(shape, circle_shift_limits):
    randint_minus_x_to_x = lambda x: random.randint(0, x) - x//2
    random_shift = [randint_minus_x_to_x(circle_shift_limits[i]) for i in range(len(circle_shift_limits))]
    return sp.linop.Circshift(shape, random_shift)


# apply closing in the first two dimensions  
def morph_close(im_xyz, kernel_size, iterations=1, im_reference_mask_zxy=None):
    return morph(im_xyz, kernel_size, cv2.MORPH_CLOSE, iterations, im_reference_mask_zxy)

def morph_erode(im_xyz, kernel_size, iterations=1, im_reference_mask_zxy=None):
    return morph(im_xyz, kernel_size, cv2.MORPH_ERODE, iterations, im_reference_mask_zxy)

def morph_open(im_xyz, kernel_size, iterations=1, im_reference_mask_zxy=None):
    return morph(im_xyz, kernel_size, cv2.MORPH_OPEN, iterations, im_reference_mask_zxy)

def morph(im_xyz, kernel_size, morph_flag, iterations=1, im_reference_mask_xyz=None):
    nx, ny, nz = im_xyz.shape

    im_closed_xyz = np.zeros_like(im_xyz)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    for zz in range(nz):
        im_xy = im_xyz[..., zz].astype(np.uint8)
        im_reference_mask_xy = 1 if im_reference_mask_xyz is None else im_reference_mask_xyz[..., zz].astype(np.uint8)
        # continuously apply the reference mask to prevent weird regions from growing
        for n in range(iterations):
            im_xy = cv2.morphologyEx(im_xy, morph_flag, kernel, iterations=1) * im_reference_mask_xy

        im_closed_xyz[..., zz] = im_xy
        
    return im_closed_xyz

def upsample(x, full_matrix_size):
    ksp = sp.fft(x)
    ksp = uncenter_crop(ksp, full_matrix_size)
    return sp.ifft(ksp)

def uncenter_crop(x, full_matrix_size):
    warnings.warn('Use sp.resize instead')
    assert(len(full_matrix_size) == x.ndim)    
    assert np.all(np.logical_or(np.array(full_matrix_size) % 2 == 0, np.array(full_matrix_size) == -1)), "Does not support odd matrix sizes, use -1 for full"
    inner_matrix_size = x.shape
    output_matrix_size = [x.shape[dim] if full_matrix_size[dim] == -1 else full_matrix_size[dim] for dim in range(x.ndim)]
    xp = sp.get_array_module(x)
    with sp.get_device(x):        
        y_full = xp.zeros(output_matrix_size, dtype=x.dtype)
        slicer = (slice(full_matrix_size[dim]//2 - inner_matrix_size[dim]//2, full_matrix_size[dim]//2 + inner_matrix_size[dim]//2) if full_matrix_size[dim] > 0 else slice(None) for dim in range(0, x.ndim))
        y_full[tuple(slicer)] = x
    return y_full

def get_hamming_filter(n, wf=2):
    hamming_filter = 0.54 + 0.46 * np.cos(2 * np.pi * np.linspace(-1, 1, n) / wf)
    return hamming_filter

def get_fermi_filter(n, rf=1, wf=.2):
    # https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.1910370514?saml_referrer
    fermi_filter_one_side = 1 / (1 + np.exp((np.linspace(0, 1, n // 2) - rf) / wf))
    fermi_filter = np.hstack((np.flip(fermi_filter_one_side), fermi_filter_one_side)) / np.max(fermi_filter_one_side)
    return fermi_filter

def box_window(inner_matrix_size, full_matrix_size, dtype=np.complex64):
    # returns a mask with center is ones
    assert(len(full_matrix_size) == len(inner_matrix_size))
    return sp.resize(np.ones(inner_matrix_size, dtype=dtype), full_matrix_size)

def center_crop(x, crop_size):
    warnings.warn('Use sp.resize instead')
    # center crops the array x to a dimension specified by crop size, if crop_size has a -1 it will take the whole axis    
    assert(x.ndim == len(crop_size))
    assert(all(tuple(crop_size[dim] <= x.shape[dim] for dim in range(0, x.ndim))))    
    slicer = tuple(slice((x.shape[dim] - crop_size[dim])//2 if crop_size[dim] > 0 else None, (x.shape[dim] + crop_size[dim])//2 if crop_size[dim] > 0 else None) for dim in range(0, x.ndim))
    return x[slicer]


class WindowType(enum.Enum):
    BOX = 0
    TRIANGLE = 1
    GAUSSIAN = 2
    HOUSE = 3

def get_window(window_type, window_fwhm, full_matrix_size):
    if window_type == WindowType.BOX:
        window = box_window(window_fwhm, full_matrix_size)
    elif window_type == WindowType.TRIANGLE:
        window = triangle_window(window_fwhm, full_matrix_size)
    elif window_type == WindowType.GAUSSIAN:
        window = gauss_window(window_fwhm, (1, 1), full_matrix_size) 
    elif window_type == WindowType.HOUSE:
        window = house_window(window_fwhm, full_matrix_size)
    else:
        raise ValueError("Invalid window_type")

    return window

def house_window(fwhm_size, full_matrix_size):
    # triangle_window_dims is the size of the full triangle
    # returns a ramp within inner_matrix_size that is 0.5 at point inner_matrix_size, zeros outside
    return sp.resize(sp.resize(triangle_window(fwhm_size, full_matrix_size), fwhm_size), full_matrix_size)

def triangle_window(fwhm_size, full_matrix_size=None):    
    # returns a ramp triangle window within 2 * fwhm_size, zeros everywhere else    
    if full_matrix_size is None:
        full_matrix_size = tuple(2 * x for x in fwhm_size)

    triangle_size = tuple(2 * x for x in fwhm_size)
    assert(all(tuple((fwhm_size[dim] <= full_matrix_size[dim],) for dim in range(len(fwhm_size)))))
    coords = np.meshgrid(*tuple(((np.linspace(-1, 1, triangle_size[sz]),) for sz in range(len(triangle_size)))), indexing='ij')

    triangle_window = np.ones(triangle_size, dtype=np.float32)
    for dim in range(len(coords)):    
        triangle_window = triangle_window * np.abs(1 - np.abs(coords[dim]))
    
    return sp.resize(triangle_window, full_matrix_size)

def gauss_window(inner_matrix_size, inner_matrix_sigma, full_matrix_size=None):
    # inner_matrix_sigma is standard deviation across the inner matrix shape 
    if full_matrix_size is None:
        full_matrix_size = inner_matrix_size    
    coords = np.meshgrid(*tuple(((np.linspace(-1, 1, inner_matrix_size[sz]),) for sz in range(len(inner_matrix_size)))), indexing='ij')

    gauss_window = np.ones(inner_matrix_size, dtype=np.float32)    

    for dim, sigma in enumerate(inner_matrix_sigma):
        gauss_window = gauss_window * np.exp(-np.square(coords[dim]) / (2 * (sigma ** 2)))

    return sp.resize(gauss_window, full_matrix_size)

def fermi_filter(matrix_size, r=None, w=10):
    """
    Returns a fermi filter operator to apply to k-space data of a specified size.
    
    Assumes the k-space data is centered.
    
    Parameters
    ----------
    matrix_size : tuple of ints
        Specifies the size of the k-space on which to apply the Fermi Filter
    r : int or tuple of ints (Default is matrix_size/2)
        Specifies the radius of the filter
    w : int or tuple of ints
        Specifies the width of the filter
    
    Returns
    -------
    H : numpy array
        An array that specifies the element-wise coefficients for the specified
        Fermi filter. Apply by doing y * H.
    """
    if r is None:
        r = tuple(int(i/2) for i in matrix_size)
    elif isinstance(r, int):
        r = tuple(r for i in matrix_size)
        
    H = np.ones(matrix_size)
    axes = [i for i in range(H.ndim)]
    c = 0
    for n in matrix_size:
        h = np.linspace(-int(n/2), int(n/2)-1, num=n)
        h = (1 + np.exp((h - r[c])/w)) ** -1
        Hn = np.broadcast_to(h, tuple(np.roll(matrix_size[::-1], c)))
        Hn = np.moveaxis(Hn.T, axes, np.roll(axes, -c))
        H = H * Hn
        c += 1
    
    return H

def whiten(x, cov):
    """
    Applies a Cholesky whitening transform to the data.
    
    Parameters
    ----------
    x : numpy/cupy array
        Input array whose first dimension corresponds to the number of coils
    cov : numpy/cupy array
        Covariance matrix of the noise present in x
        
    Returns
    -------
    y : numpy/cupy array
        A whitened version of x with the same dimensions
    """
    
    # Check that the whitening transform can be applied
    if x.shape[-2] != cov.shape[0]:
        raise ValueError("The first dimension of x and the provided covariance matrix do not match.")
    
    device = sp.get_device(x)
    xp = device.xp
    cov = sp.to_device(cov, device)
        
    # Get the whitening transform operator
    L = xp.linalg.cholesky(xp.linalg.inv(cov))
    LH = L.conj().T
    
    # Apply the whitening transform and return the result
    y = LH @ x
    return y

def sos(matrix_in, axis_in, keepdims=False):
    # Return square root sum of squares
    device = sp.get_device(matrix_in)
    xp = device.xp
    with device:
        matrix_out = xp.sqrt(xp.sum(xp.abs(matrix_in) ** 2, axis=axis_in, keepdims=keepdims))
    return matrix_out

def match_device(a, b):
    # copies a to the same device as b
    return sp.to_device(a, sp.backend.get_device(b))

