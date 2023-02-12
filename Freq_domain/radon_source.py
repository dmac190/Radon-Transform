# -*- coding: utf-8 -*-
"""
radon.py - Radon and inverse radon transforms

Based on code of Justin K. Romberg
(http://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)
J. Gillam and Chris Griffin.

References:
    -B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
    the Discrete Radon Transform With Some Applications", Proceedings of
    the Fourth IEEE Region 10 International Conference, TENCON '89, 1989.
    -A. C. Kak, Malcolm Slaney, "Principles of Computerized Tomographic
    Imaging", IEEE Press 1988.
"""
from __future__ import division
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from ._warps_cy import _warp_fast
from ._radon_transform import sart_projection_update
from .. import util
from warnings import warn


__all__ = ["radon", "iradon", "iradon_sart"]


def radon(image, theta=None, circle=False):
    """
    Calculates the radon transform of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like, dtype=float
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, dtype=float, optional (default np.arange(180))
        Projection angles (in degrees).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.

    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.

    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)

    if circle:
        radius = min(image.shape) // 2
        c0, c1 = np.ogrid[0:image.shape[0], 0:image.shape[1]]
        reconstruction_circle = ((c0 - image.shape[0] // 2) ** 2
                                 + (c1 - image.shape[1] // 2) ** 2)
        reconstruction_circle = reconstruction_circle <= radius ** 2
        if not np.all(reconstruction_circle | (image == 0)):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = []
        for d in (0, 1):
            if image.shape[d] > min(image.shape):
                excess = image.shape[d] - min(image.shape)
                slices.append(slice(int(np.ceil(excess / 2)),
                                    int(np.ceil(excess / 2)
                                        + min(image.shape))))
            else:
                slices.append(slice(None))
        slices = tuple(slices)
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = util.pad(image, pad_width, mode='constant',
                                constant_values=0)
    # padded_image is always square
    assert padded_image.shape[0] == padded_image.shape[1]
    radon_image = np.zeros((padded_image.shape[0], len(theta)))
    center = padded_image.shape[0] // 2

    shift0 = np.array([[1, 0, -center],
                       [0, 1, -center],
                       [0, 0, 1]])
    shift1 = np.array([[1, 0, center],
                       [0, 1, center],
                       [0, 0, 1]])

    def build_rotation(theta):
        T = np.deg2rad(theta)
        R = np.array([[np.cos(T), np.sin(T), 0],
                      [-np.sin(T), np.cos(T), 0],
                      [0, 0, 1]])
        return shift1.dot(R).dot(shift0)

    for i in range(len(theta)):
        rotated = _warp_fast(padded_image, build_rotation(theta[i]))
        radon_image[:, i] = rotated.sum(0)
    return radon_image


def _sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return util.pad(sinogram, pad_width, mode='constant', constant_values=0)

def iradon(radon_image, theta=None, output_size=None,
           filter="ramp", interpolation="linear", circle=False):
    """
    Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : array_like, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array_like, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int
        Number of rows and columns in the reconstruction.
    filter : str, optional (default ramp)
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    interpolation : str, optional (default 'linear')
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        m, n = radon_image.shape
        theta = np.linspace(0, 180, n, endpoint=False)
    else:
        theta = np.asarray(theta)
    if len(theta) != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")
    interpolation_types = ('linear', 'nearest', 'cubic')
    if not interpolation in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
    if not output_size:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = radon_image.shape[0]
        else:
            output_size = int(np.floor(np.sqrt((radon_image.shape[0]) ** 2
                                               / 2.0)))
    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)

    th = (np.pi / 180.0) * theta
    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = \
        max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = util.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Construct the Fourier filter
    f = fftfreq(projection_size_padded).reshape(-1, 1)   # digital frequency
    omega = 2 * np.pi * f                                # angular frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter
    if filter == "ramp":
        pass
    elif filter == "shepp-logan":
        # Start from first element to avoid divide by zero
        fourier_filter[1:] = fourier_filter[1:] * np.sin(omega[1:]) / omega[1:]
    elif filter == "cosine":
        fourier_filter *= np.cos(omega)
    elif filter == "hamming":
        fourier_filter *= (0.54 + 0.46 * np.cos(omega / 2))
    elif filter == "hann":
        fourier_filter *= (1 + np.cos(omega / 2)) / 2
    elif filter is None:
        fourier_filter[:] = 1
    else:
        raise ValueError("Unknown filter: %s" % filter)
    # Apply filter in Fourier domain
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))

    # Resize filtered image back to original size
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    # Determine the center of the projections (= center of sinogram)
    mid_index = radon_image.shape[0] // 2

    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2

    # Reconstruct image by interpolation
    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_filtered.shape[0]) - mid_index
        if interpolation == 'linear':
            backprojected = np.interp(t, x, radon_filtered[:, i],
                                      left=0, right=0)
        else:
            interpolant = interp1d(x, radon_filtered[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            backprojected = interpolant(t)
        reconstructed += backprojected
    if circle:
        radius = output_size // 2
        reconstruction_circle = (xpr ** 2 + ypr ** 2) <= radius ** 2
        reconstructed[~reconstruction_circle] = 0.

    return reconstructed * np.pi / (2 * len(th))