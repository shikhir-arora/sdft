# rewrite.pyx

from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel
import numpy as np
from libc.math cimport cos, sin, pi
cimport cython

# Define the quantization function
cdef double Quantize(double value):
    cdef double Factor = 2**15  # Ensure at least 15 bits of precision
    return Factor * round((value / Factor) * 131072) / \
        131072  # 131072 is 2**17 for high precision


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_stable_sdft(complex[:] signal, int N, int k):
    cdef int n = len(signal)
    cdef double * y_real = <double * > malloc(n * sizeof(double))
    cdef double * y_imag = <double * > malloc(n * sizeof(double))
    cdef double * norm_factor = <double * > malloc(n * sizeof(double))
    cdef double exp_factor_real, exp_factor_imag, cos_factor
    cdef double B_real[3], B_imag[3]
    cdef double A[3]
    cdef int i

    if y_real is NULL or y_imag is NULL or norm_factor is NULL:
        raise MemoryError("Could not allocate buffer.")

    # Quantize feed-forward coefficients (outside the loop)
    exp_factor_real = Quantize(cos(2 * pi * k / N))
    exp_factor_imag = Quantize(sin(2 * pi * k / N))
    cos_factor = Quantize(-2 * cos(2 * pi * k / N))

    B_real[0] = exp_factor_real
    B_real[1] = -1.0
    B_real[2] = -exp_factor_real

    B_imag[0] = exp_factor_imag
    B_imag[1] = 0.0
    B_imag[2] = -exp_factor_imag

    A[0] = 1.0
    A[1] = cos_factor
    A[2] = 1.0

    # Initialize norm_factor
    for i in range(n):
        norm_factor[i] = 1.0  # Simplified normalization

    # Apply the filter
    for i in range(n):
        y_real[i] = signal[i].real * B_real[0] - signal[i].imag * B_imag[0]
        y_imag[i] = signal[i].real * B_imag[0] + signal[i].imag * B_real[0]

        if i > 0:
            y_real[i] += signal[i - 1].real * \
                B_real[1] - signal[i - 1].imag * B_imag[1]
            y_imag[i] += signal[i - 1].real * \
                B_imag[1] + signal[i - 1].imag * B_real[1]

        if i > 1:
            y_real[i] += signal[i - 2].real * \
                B_real[2] - signal[i - 2].imag * B_imag[2]
            y_imag[i] += signal[i - 2].real * \
                B_imag[2] + signal[i - 2].imag * B_real[2]

        if i > 0:
            y_real[i] -= y_real[i - 1] * A[1]
            y_imag[i] -= y_imag[i - 1] * A[1]

        if i > 1:
            y_real[i] -= y_real[i - 2] * A[2]
            y_imag[i] -= y_imag[i - 2] * A[2]

    with nogil, parallel():
        for i in prange(n):
            if norm_factor[i] == 0:
                norm_factor[i] = 1e-30  # Avoid division by zero
            y_real[i] /= norm_factor[i]
            y_imag[i] /= norm_factor[i]

    cdef np.ndarray[complex, ndim= 1] result = np.empty(n, dtype=complex)
    for i in range(n):
        result[i] = y_real[i] + 1j * y_imag[i]

    free(y_real)
    free(y_imag)
    free(norm_factor)

    return result


cpdef cython_sdft(complex[::1] signal, int n):
    """
    Compute the Sliding Discrete Fourier Transform (SDFT) of a given signal.
    """
    cdef complex omega = cos(-2 * pi / n) + 1j * sin(-2 * pi / n)
    cdef complex x_prev = 0 + 0j
    cdef complex[::1] x = np.empty(len(signal), dtype=complex)
    cdef int i

    for i in range(n):
        x_prev += signal[i] * (cos(-2 * pi * i / n) +
                               1j * sin(-2 * pi * i / n))
        x[i] = x_prev
    for i in range(n, len(signal)):
        x_prev = x_prev - signal[i - n] + signal[i]
        x[i] = x_prev * omega
        x_prev = x[i]

    return np.asarray(x)

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
