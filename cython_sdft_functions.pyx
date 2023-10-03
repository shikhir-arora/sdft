#cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport exp, pi, cos, sin
from libc.complex cimport complex
cimport numpy as np
import numpy as np
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free
from typing import List, Dict

# Custom lfilter in Cython
cdef void cython_lfilter(complex *B, complex *A, complex *x, complex *y, int n):
    cdef int i, j
    cdef complex sum_val
    for i in range(n):
        sum_val = 0
        for j in range(4):  # B has 4 elements
            if i - j < 0:
                continue
            sum_val += B[j] * x[i - j]
        for j in range(1, 3):  # A has 3 elements, and A[0] is 1 so we skip it
            if i - j < 0:
                continue
            sum_val -= A[j] * y[i - j]
        y[i] = sum_val

# Simple convolution to compute norm_factor
cdef void compute_norm_factor(complex *A, complex *norm_factor, int n):
    cdef int i, j
    for i in range(n):
        norm_factor[i] = 0
        for j in range(3):  # A has 3 elements
            if i - j < 0:
                continue
            norm_factor[i] += A[j]

cpdef cython_stable_sdft(complex[::1] signal, int N, int k):
    """
    Compute the Stable SDFT Network (SDFT) of a given signal at frequency bin k.
    """
    cdef int n = len(signal)
    cdef complex *y = <complex *> malloc(n * sizeof(complex))
    cdef complex *norm_factor = <complex *> malloc(n * sizeof(complex))
    cdef complex exp_factor, cos_factor
    cdef complex B[4]
    cdef complex A[3]
    cdef int i

    if y is NULL or norm_factor is NULL:
        raise MemoryError("Could not allocate buffer.")

    exp_factor = <complex> (cos(2 * pi * k / N) + 1j * sin(2 * pi * k / N))
    cos_factor = -2 * cos(2 * pi * k / N)

    B[0] = exp_factor
    B[1] = -1
    B[2] = -exp_factor
    B[3] = 1

    A[0] = 1
    A[1] = cos_factor
    A[2] = 1

    # Call to our optimised cython_lfilter
    cython_lfilter(&B[0], &A[0], &signal[0], y, n)

    # Compute norm_factor
    compute_norm_factor(A, norm_factor, n)

    # Releasing the GIL for the division operation
    with nogil, parallel():
        for i in prange(n):
            if norm_factor[i] == 0:
                norm_factor[i] = 1e-30  # Replace zeros to avoid division by zero
            y[i] /= norm_factor[i]

    # Copy data back to a NumPy array
    cdef np.ndarray[complex, ndim=1] result = np.empty(n, dtype=complex)
    for i in range(n):
        result[i] = y[i]

    # free malloc 
    free(y)
    free(norm_factor)

    return result


# Cython version of SDFT
cpdef cython_sdft(complex[::1] signal, int n):
    """
    Compute the Sliding Discrete Fourier Transform (SDFT) of a given signal.
    """
    cdef complex omega = cos(-2 * pi / n) + 1j * sin(-2 * pi / n)
    cdef complex x_prev = 0 + 0j
    cdef complex[::1] x = np.empty(len(signal), dtype=complex)
    cdef int i

    for i in range(n):
        x_prev += signal[i] * (cos(-2 * pi * i / n) + 1j * sin(-2 * pi * i / n))
        x[i] = x_prev
    for i in range(n, len(signal)):
        x_prev = x_prev - signal[i - n] + signal[i]
        x[i] = x_prev * omega
        x_prev = x[i]

    return np.asarray(x)


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
