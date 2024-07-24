# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np
from libc.math cimport cos, sin, pi
cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free
from typing import Dict

# Define the quantization function
cdef double Quantize(double value):
    """
    Quantize a given value to ensure at least 15 bits of precision.
    
    Parameters:
        value (double): The value to be quantized.
    
    Returns:
        double: The quantized value.
    """
    cdef double Factor = 2**15  # Ensure at least 15 bits of precision
    return Factor * round((value / Factor) * 131072) / 131072  # 131072 is 2**17 for high precision

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[complex, ndim=1] cython_stable_sdft(complex[::1] signal, int N, int k):
    """
    Compute the Stable Sliding Discrete Fourier Transform (SDFT) of a given signal at a specific frequency bin.

    Parameters:
    signal (complex[::1]): The input complex signal array.
    N (int): The number of points for the Fourier Transform.
    k (int): The frequency bin index.

    Returns:
        np.ndarray[complex, ndim=1]: The stable SDFT of the input signal.
    """
    cdef int n = len(signal)
    cdef double *y_real = <double *> malloc(n * sizeof(double))
    cdef double *y_imag = <double *> malloc(n * sizeof(double))
    cdef double *norm_factor = <double *> malloc(n * sizeof(double))
    cdef double *result_real = <double *> malloc(n * sizeof(double))
    cdef double *result_imag = <double *> malloc(n * sizeof(double))
    cdef double exp_factor_real, exp_factor_imag, cos_factor
    cdef double B_real[3], B_imag[3]
    cdef double A[3]
    cdef int I

    if y_real is NULL or y_imag is NULL or norm_factor is NULL or result_real is NULL or result_imag is NULL:
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
    for I in range(n):
        norm_factor[I] = 1.0

    # Apply the filter
    with nogil, parallel():
        for I in prange(n):
            y_real[I] = signal[I].real * B_real[0] - signal[I].imag * B_imag[0]
            y_imag[I] = signal[I].real * B_imag[0] + signal[I].imag * B_real[0]
            if I > 0:
                y_real[I] += signal[I-1].real * B_real[1] - signal[I-1].imag * B_imag[1]
                y_imag[I] += signal[I-1].real * B_imag[1] + signal[I-1].imag * B_real[1]
            if I > 1:
                y_real[I] += signal[I-2].real * B_real[2] - signal[I-2].imag * B_imag[2]
                y_imag[I] += signal[I-2].real * B_imag[2] + signal[I-2].imag * B_real[2]
            if I > 0:
                y_real[I] -= y_real[I-1] * A[1]
                y_imag[I] -= y_imag[I-1] * A[1]
            if I > 1:
                y_real[I] -= y_real[I-2] * A[2]
                y_imag[I] -= y_imag[I-2] * A[2]
            if norm_factor[I] == 0:
                norm_factor[I] = 1e-30  # Avoid division by zero
            result_real[I] = y_real[I] / norm_factor[I]
            result_imag[I] = y_imag[I] / norm_factor[I]

    # Convert C arrays to NumPy arrays
    cdef np.ndarray[complex, ndim=1] result = np.empty(n, dtype=complex)
    for I in range(n):
        result[I] = result_real[I] + 1j * result_imag[I]

    free(y_real)
    free(y_imag)
    free(norm_factor)
    free(result_real)
    free(result_imag)

    return result

cpdef np.ndarray[complex, ndim=1] cython_sdft(complex[::1] signal, int n):
    """
    Compute the Sliding Discrete Fourier Transform (SDFT) of a given signal.
    
    Parameters:
        signal (complex[::1]): The input complex signal array.
        n (int): The number of points for the Fourier Transform.
    
    Returns:
        np.ndarray[complex, ndim=1]: The SDFT of the input signal.
    """
    cdef complex omega = cos(-2 * pi / n) + 1j * sin(-2 * pi / n)
    cdef complex x_prev = 0 + 0j
    cdef complex[::1] x = np.empty(len(signal), dtype=complex)
    cdef int I

    for I in range(n):
        x_prev += signal[I] * (cos(-2 * pi * I / n) + 1j * sin(-2 * pi * I / n))
        x[I] = x_prev
    for I in range(n, len(signal)):
        x_prev = x_prev - signal[I - n] + signal[I]
        x[I] = x_prev * omega
        x_prev = x[I]

    return np.asarray(x)

cpdef Dict[str, float] cython_psychoacoustic_mapping(double[::1] freqs, double[::1] mags):
    """
    Map frequencies to psychoacoustic bands and sum the magnitudes within each band.
    
    Parameters:
        freqs (double[::1]): Array of frequency values.
        mags (double[::1]): Array of magnitude values corresponding to the frequencies.
    
    Returns:
        Dict[str, float]: Dictionary containing the summed magnitudes for each psychoacoustic band.
    """
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
        for I in range(len(freqs)):
            if freqs[I] >= f_low and freqs[I] < f_high:
                sum_value += mags[I]
        band_values[band] = sum_value

    return band_values

cdef double hz_to_mel(double hz):
    """
    Convert frequency in Hertz to Mel scale.
    
    Parameters:
        hz (double): Frequency in Hertz.
    
    Returns:
        double: Frequency in Mel scale.
    """
    return 2595 * np.log10(1 + hz / 700)

cdef double mel_to_hz(double mel):
    """
    Convert frequency in Mel scale to Hertz.
    
    Parameters:
        mel (double): Frequency in Mel scale.
    
    Returns:
        double: Frequency in Hertz.
    """
    return 700 * (10**(mel / 2595) - 1)

cpdef np.ndarray[double, ndim=1] generate_mel_freqs(int num_bins, int fs):
    """
    Generate an array of frequencies in Hertz corresponding to Mel scale bins.
    
    Parameters:
        num_bins (int): Number of Mel scale bins.
        fs (int): Sampling frequency.
    
    Returns:
        np.ndarray[double, ndim=1]: Array of frequencies in Hertz.
    """
    cdef double mel_max = hz_to_mel(fs // 2)
    cdef np.ndarray[double, ndim=1] mel_bins = np.linspace(0, mel_max, num_bins)
    cdef np.ndarray[double, ndim=1] hz_bins = np.empty(num_bins, dtype=np.double)
    for I in range(num_bins):
        hz_bins[I] = mel_to_hz(mel_bins[I])
    return hz_bins
