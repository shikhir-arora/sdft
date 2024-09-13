# cython: language_level=3, boundscheck=False, wraparound=False

from libc.math cimport exp, pi, cos, sin
cimport numpy as np
import numpy as np
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

cpdef np.ndarray[complex, ndim=1] cython_stable_sdft(complex[::1] signal, int N, int k):
    """
    Compute the Stable SDFT of a given signal at frequency bin k.

    Parameters:
        signal (complex[::1]): Input signal array.
        N (int): Number of points in the DFT.
        k (int): Frequency bin index.

    Returns:
        np.ndarray[complex, ndim=1]: The computed SDFT at frequency bin k.
    """
    cdef int n = signal.shape[0]
    cdef complex exp_factor, cos_factor
    cdef complex *B
    cdef complex *A
    cdef complex *y = <complex *> malloc(n * sizeof(complex))
    cdef complex *norm_factor = <complex *> malloc(n * sizeof(complex))
    cdef int i, j

    if y is NULL or norm_factor is NULL:
        raise MemoryError("Could not allocate buffer.")

    # Calculate filter coefficients
    exp_factor = cos(2 * pi * k / N) + 1j * sin(2 * pi * k / N)
    cos_factor = -2 * cos(2 * pi * k / N)

    # Allocate coefficient arrays
    B = <complex *> malloc(4 * sizeof(complex))
    A = <complex *> malloc(3 * sizeof(complex))

    if B is NULL or A is NULL:
        free(y)
        free(norm_factor)
        raise MemoryError("Could not allocate coefficient buffers.")

    B[0] = exp_factor    # Corresponds to z^0 term
    B[1] = -1            # Corresponds to z^{-1} term
    B[2] = -exp_factor   # Corresponds to z^{-2} term
    B[3] = 1             # Corresponds to z^{-3} term

    A[0] = 1             # Corresponds to z^0 term
    A[1] = cos_factor    # Corresponds to z^{-1} term
    A[2] = 1             # Corresponds to z^{-2} term

    # Apply the filter to the signal
    cdef complex *x = &signal[0]  # Direct pointer to the input signal

    # Initialize output array
    for i in range(n):
        y[i] = 0
        norm_factor[i] = 0

    # Apply the filter
    with nogil:
        for i in range(n):
            cdef complex sum_val = 0
            # Feedforward part
            for j in range(4):
                if i - j >= 0:
                    sum_val += B[j] * x[i - j]
            # Feedback part
            for j in range(1, 3):  # Skip A[0] because it's assumed to be 1
                if i - j >= 0:
                    sum_val -= A[j] * y[i - j]
            y[i] = sum_val

    # Compute normalization factor
    with nogil:
        for i in range(n):
            for j in range(3):
                if i - j >= 0:
                    norm_factor[i] += A[j]

    # Normalize the output
    with nogil, parallel():
        for i in prange(n, schedule='static'):
            if norm_factor[i] == 0:
                norm_factor[i] = 1e-30  # Prevent division by zero
            y[i] /= norm_factor[i]

    # Copy data back to a NumPy array
    result = np.empty(n, dtype=np.complex128)
    for i in range(n):
        result[i] = y[i]

    # Free allocated memory
    free(y)
    free(norm_factor)
    free(B)
    free(A)

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
