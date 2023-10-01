#cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport exp, pi, cos, sin
from libc.complex cimport complex
cimport numpy as cnp
import numpy as np
from typing import List, Dict

cdef complex complex_exp(double x):
    cdef double re = cos(x)
    cdef double im = sin(x)
    return complex(re, im)

# Custom lfilter in Cython
cpdef cython_lfilter(complex[::1] B, complex[::1] A, complex[::1] x):
    cdef int n = x.shape[0]
    cdef complex[::1] y = np.empty(n, dtype=complex)
    cdef int i, j
    cdef complex sum_val

    for i in range(n):
        sum_val = 0
        for j in range(B.shape[0]):
            if i - j < 0: continue
            sum_val += B[j] * x[i - j]
        for j in range(1, A.shape[0]):
            if i - j < 0: continue
            sum_val -= A[j] * y[i - j]
        y[i] = sum_val / A[0]

    return np.asarray(y)

# Cython version of SDFT
cpdef cython_sdft(complex[::1] signal, int n):
    """
    Compute the Sliding Discrete Fourier Transform (SDFT) of a given signal.
    """
    cdef complex omega = complex_exp(-2 * pi / n)
    cdef complex x_prev = 0 + 0j
    cdef complex[::1] x = np.empty(len(signal), dtype=complex)
    cdef int i

    for i in range(n):
        x_prev += signal[i] * complex_exp(-2 * pi * i / n)
        x[i] = x_prev
    for i in range(n, len(signal)):
        x_prev = x_prev - signal[i - n] + signal[i]
        x[i] = x_prev * omega
        x_prev = x[i]

    return np.asarray(x)

# [!! warning: experimental wrt optimization] Cython version of stable SDFT
cpdef cython_stable_sdft(complex[::1] signal, int N, int k):
    """
    Compute the Stable SDFT Network (SDFT) of a given signal at frequency bin k.
    """
    cdef complex exp_factor = complex_exp(2 * pi * k / N)
    cdef complex cos_factor = -2 * cos(2 * pi * k / N)
    cdef complex[::1] B = np.array([exp_factor, -1, -exp_factor, 1], dtype=complex)
    cdef complex[::1] A = np.array([1, cos_factor, 1], dtype=complex)
    cdef complex[::1] y = cython_lfilter(B, A, signal)
    cdef complex[::1] norm_factor = np.convolve(np.ones(len(signal), dtype=complex), np.asarray(A))[: len(signal)]

    cdef int i
    for i in range(len(norm_factor)):
        if norm_factor[i] == 0:
            norm_factor[i] = 1e-30

    for i in range(len(y)):
        y[i] /= norm_factor[i]

    return np.asarray(y)

# Cython version of psychoacoustic_mapping
cpdef Dict[str, float] cython_psychoacoustic_mapping(double[::1] freqs, double[::1] mags):
    cdef Dict[str, tuple] bands = {
        "Sub-Bass": (20, 120),
        "Bass": (120, 420),
        "Low Mid-Bass": (420, 1000),
        "Mid-Bass": (1000, 3000),
        "Midrange": (3000, 6000),
        "Presence": (6000, 8000),
        "Upper Midrange": (8000, 12000),
        "Brilliance": (12000, 20000)
    }
    cdef Dict[str, float] band_values = {}
    cdef str band
    cdef tuple freq_range
    cdef double f_low, f_high
    cdef double sum_value = 0.0

    for band, freq_range in bands.items():
        f_low, f_high = freq_range
        sum_value = 0.0
        for i in range(len(freqs)):
            if freqs[i] >= f_low and freqs[i] < f_high:
                sum_value += mags[i]
        band_values[band] = sum_value

    return band_values
