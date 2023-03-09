"""Entropy functions"""
# Borrowed from https://github.com/raphaelvallat/antropy/blob/master/antropy/entropy.py

from builtins import range

import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import periodogram

from src.utils.ts_utils import make_stationary


@np.vectorize
def _xlog2x(x):
    """Returns x log2 x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    return 0.0 if x == 0 else x * np.log2(x)


def spectral_entropy(
    x, sampling_frequency=1, normalize=True, axis=-1, transform_stationary=False
):
    """Spectral Entropy.
    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    sampling_frequency : float
        Sampling frequency for the FFT, in Hz.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.
    axis : int
        The axis along which the entropy is calculated. Default is -1 (last).
    transform_stationary : bool
        Flag to decide if we should make the series stationary before calculating spectral entropy. Default is False.
    Returns
    -------
    se : float
        Spectral Entropy
    Notes
    -----
    Spectral Entropy is defined to be the Shannon entropy of the power
    spectral density (PSD) of the data:
    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) \\log_2[P(f)]
    Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.
    """
    x = np.asarray(x)
    if transform_stationary:
        x, _ = make_stationary(x, method="detrend")
    # Compute and normalize power spectrum
    _, psd = periodogram(x, sampling_frequency, axis=axis)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -_xlog2x(psd_norm).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se


@njit
def _rolling(a, window):
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


# @njit
@njit
def apply_along_axis_0(func1d, arr):
    """Like calling func1d(arr, axis=0)"""
    if arr.size == 0:
        raise RuntimeError("Must have arr.size > 0")
    ndim = arr.ndim
    if ndim == 0:
        raise RuntimeError("Must have ndim > 0")
    elif 1 == ndim:
        return func1d(arr)
    else:
        result_shape = arr.shape[1:]
        out = np.empty(result_shape, arr.dtype)
        _apply_along_axis_0(func1d, arr, out)
        return out


@njit
def _apply_along_axis_0(func1d, arr, out):
    """Like calling func1d(arr, axis=0, out=out). Require arr to be 2d or bigger."""
    ndim = arr.ndim
    if ndim < 2:
        raise RuntimeError("_apply_along_axis_0 requires 2d array or bigger")
    elif ndim == 2:  # 2-dimensional case
        for i in range(len(out)):
            out[i] = func1d(arr[:, i])
    else:  # higher dimensional case
        for i, out_slice in enumerate(out):
            _apply_along_axis_0(func1d, arr[:, i], out_slice)


@njit
def nb_mean_axis_0(arr):
    return apply_along_axis_0(np.mean, arr)


@njit
def nb_std_axis_0(arr):
    return apply_along_axis_0(np.std, arr)


@njit
def nb_amax_axis_0(arr):
    return apply_along_axis_0(np.amax, arr)


def _into_subchunks(x, subchunk_length, every_n=1):
    """
    Split the time series x into subwindows of length "subchunk_length", starting every "every_n".

    For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix

        0  2  4
        1  3  5
        2  4  6

    with the settings subchunk_length = 3 and every_n = 2
    """
    len_x = len(x)

    assert subchunk_length > 1
    assert every_n > 0

    # how often can we shift a window of size subchunk_length over the input?
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
    return x[indexer]


# @njit
def sample_entropy(x, transform_stationary=False):
    """
    Calculate and return sample entropy of x.

    .. rubric:: References

    |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
    |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray

    :return: the value of this feature
    :return type: float
    """
    # x = np.array(x)

    # if one of the values is NaN, we can not compute anything meaningful
    if np.isnan(x).any():
        return np.nan

    m = 2  # common value for m, according to wikipedia...
    tolerance = 0.2 * np.std(
        x
    )  # 0.2 is a common value for r, according to wikipedia...
    if transform_stationary:
        x, _ = make_stationary(x, method="logdiff")

    # Split time series and save all templates of length m
    # Basically we turn [1, 2, 3, 4] into [1, 2], [2, 3], [3, 4]
    xm = _into_subchunks(x, m)

    # Now calculate the maximum distance between each of those pairs
    #   np.abs(xmi - xm).max(axis=1)
    # and check how many are below the tolerance.
    # For speed reasons, we are not doing this in a nested for loop,
    # but with numpy magic.
    # Example:
    # if x = [1, 2, 3]
    # then xm = [[1, 2], [2, 3]]
    # so we will substract xm from [1, 2] => [[0, 0], [-1, -1]]
    # and from [2, 3] => [[1, 1], [0, 0]]
    # taking the abs and max gives us:
    # [0, 1] and [1, 0]
    # as the diagonal elements are always 0, we substract 1.
    B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])

    # Similar for computing A
    xmp1 = _into_subchunks(x, m + 1)
    A = np.sum(
        [np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1]
    )

    # Return SampEn
    return -np.log(A / B)


@njit
def _phi(x, m, r):
    N = x.shape[0]
    x_re = _rolling(x, m)
    diff = np.abs(x_re.copy().reshape(-1, 1, m) - x_re.copy().reshape(1, -1, m))
    _max = nb_amax_axis_0(np.transpose(diff, (2, 0, 1)))
    _max_mask = _max <= r
    C = np.sum(_max_mask, axis=0) / (N - m + 1)
    return np.sum(np.log(C)) / (N - m + 1.0)


def approximate_entropy(x, m, r, transform_stationary=False):
    """
    Implements a vectorized Approximate entropy algorithm.

        https://en.wikipedia.org/wiki/Approximate_entropy

    For short time-series this method is highly dependent on the parameters,
    but should be stable for N > 2000, see:

        Yentes et al. (2012) -
        *The Appropriate Use of Approximate Entropy and Sample Entropy with Short Data Sets*


    Other shortcomings and alternatives discussed in:

        Richman & Moorman (2000) -
        *Physiological time-series analysis using approximate entropy and sample entropy*

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param m: Length of compared run of data
    :type m: int
    :param r: Filtering level, must be positive
    :type r: float

    :return: Approximate entropy
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if transform_stationary:
        x, _ = make_stationary(x, method="logdiff")
    N = x.size
    r *= np.std(x)
    if r < 0:
        raise ValueError("Parameter r must be positive.")
    if N <= m + 1:
        return 0
    return np.abs(_phi(x, m, r) - _phi(x, m + 1, r))


# ss = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv')
# a10 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')
# rand_small = np.random.randint(0, 100, size=36)
# rand_big = np.random.randint(0, 100, size=136)
# approximate_entropy(ss.value.values, 2, r=0.2)
# import os
# # os.chdir(r"..")
# t = np.load(r"t.sav.npy")
# sample_entropy(t[:50])
