import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from rich.console import Console
from rich.style import Style
from scipy.signal import lfilter
import sounddevice as sd
from typing import Dict, List, Tuple, Union, Any
from cython_sdft_functions import (
    cython_sdft,
    cython_stable_sdft,
    cython_psychoacoustic_mapping,
)

def sdft(signal: np.ndarray, n: int) -> np.ndarray:
    """
    Compute the Sliding Discrete Fourier Transform (SDFT) of a given signal.

    Parameters:
        signal (np.ndarray): The input audio signal array.
        n (int): The number of points to use for the Fourier Transform.

    Returns:
        np.ndarray: The SDFT of the input signal.
    """
    omega = np.exp(-1j * 2 * np.pi / n)
    x_prev = 0 + 0j
    x = []
    for i in range(n):
        x_prev += signal[i] * np.exp(-1j * 2 * np.pi * i / n)
        x.append(x_prev)
    for i in range(n, len(signal)):
        x_curr = x_prev - signal[i - n] + signal[i]
        x.append(x_curr * omega)
        x_prev = x_curr
    return np.array(x)

def stable_sdft(signal: np.ndarray, N: int, k: int) -> np.ndarray:
    """
    Compute the Stable Sliding Discrete Fourier Transform (SDFT) of a given signal.

    Parameters:
        signal (np.ndarray): The input audio signal array.
        N (int): The number of points to use for the Fourier Transform.
        k (int): The frequency index for which the SDFT is calculated.

    Returns:
        np.ndarray: The stable SDFT of the input signal at the frequency index k.
    """

    exp_factor = np.exp(2 * np.pi * k / N * 1j)
    cos_factor = -2 * np.cos(2 * np.pi * k / N)

    B = np.array([exp_factor, -1, -exp_factor, 1], dtype=complex)
    A = np.array([1, cos_factor, 1], dtype=complex)
    y = lfilter(B, A, signal)

    norm_factor = np.convolve(np.ones(len(signal), dtype=complex), A, mode="full")[
        : len(signal)
    ]
    norm_factor[norm_factor == 0] = 1e-30  # Replace zeros to avoid division by zero

    np.divide(y, norm_factor, out=y)

    return y

def psychoacoustic_mapping(freqs: np.ndarray, mags: np.ndarray) -> Dict[str, float]:
    """
    Map frequencies to psychoacoustic bands and sum the magnitudes within each band.

    Parameters:
        freqs (np.ndarray): Array of frequency values.
        mags (np.ndarray): Array of magnitude values corresponding to the frequencies.

    Returns:
        Dict[str, float]: Dictionary containing the summed magnitudes for each psychoacoustic band.
    """
    bands = {
        "Sub-Bass": (20, 120),
        "Bass": (120, 420),
        "Low Mid-Bass": (420, 1000),
        "Mid-Bass": (1000, 3000),
        "Midrange": (3000, 6000),
        "Presence": (6000, 8000),
        "Upper Midrange": (8000, 9500),
        "Brilliance": (9500, 16000),
    }

    band_values = {}
    for band, (f_low, f_high) in bands.items():
        # indices = np.where((freqs >= f_low) & (freqs < f_high))
        # band_values[band] = np.mean(mags[indices])
        band_values[band] = np.sum(mags[(freqs >= f_low) & (freqs <= f_high)])

    return band_values

def make_audio_cmap(bands: List[int]) -> Any:
    """
    Create a custom colormap based on the audio frequency bands provided.

    Parameters:
        bands (List[int]): List of frequency bands for which to make the colormap.

    Returns:
        Any: Custom colormap for Matplotlib.
    """
    base_cmap = cm.turbo
    min_freq = min(bands)
    max_freq = max(bands)
    band_splits = np.linspace(min_freq, max_freq, len(bands) + 1)
    splits = (band_splits - min_freq) / (max_freq - min_freq)
    cmap_colors = np.ones((len(splits) - 1, 256, 4))

    for i, (start, end) in enumerate(zip(splits[:-1], splits[1:])):
        cmap_vals = base_cmap(np.linspace(start, end, 256))
        cmap_colors[i, :, :3] = cmap_vals[:, :3]

    new_cmap = np.vstack(cmap_colors)
    return colors.ListedColormap(new_cmap)

cmap = make_audio_cmap([20, 120, 420, 1000, 3000, 6000, 8000, 12000, 18000])

# Initialize plot
def initialize_plot() -> Tuple:
    """
    Initialize the plot for real-time audio visualization.

    Returns:
        Tuple: Matplotlib Figure and Axes objects.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    (line,) = ax1.plot([], [], lw=2, color="c")
    ax1.set_xscale("log")
    ax1.set_xlim(20, 20000)
    ax1.set_ylim(0, 50)
    ax1.grid(True)
    ax1.set_title("Real-Time S(DFT) Magnitude (20 Hz - 20 kHz)", fontsize=16)
    ax1.set_xlabel("Frequency (Hz)", fontsize=14)
    ax1.set_ylabel("Magnitude", fontsize=14)
    ax2.axis("off")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    return fig, ax1, ax2, line

# Callback function for sounddevice
def callback(indata: np.ndarray, frames: int, time: float, status: Any) -> None:
    """
    Callback function for sounddevice to update the audio data.

    Parameters:
        indata (np.ndarray): Incoming audio data.
        frames (int): Number of audio frames.
        time (float): Time information.
        status (Any): Status information.

    Returns:
        None
    """
    global audio_data
    audio_data = indata[:, 0]

# Initialization function for FuncAnimation
def init() -> List:
    """
    Initialization function for Matplotlib FuncAnimation.

    Returns:
        List: List of Matplotlib Artist objects to be updated.
    """
    line.set_data([], [])
    return [line] + lights

# Update functions for FuncAnimation

global console, style
console = Console()
style = Style(color="green", bold=True)

def update_sdft(frame: int) -> List:
    """
    Update function for Matplotlib FuncAnimation to update the plot in real-time.

    Parameters:
        frame (int): The current frame number in the animation.

    Returns:
        List: A list containing the updated line and light objects for the plot.
    """
    global audio_data, cmap
    

    audio_data_complex = audio_data.astype(np.complex128, copy=False)
    sdft_results = cython_sdft(audio_data_complex, 50)  # !! NOTICE !! cython enforced

    half_len = len(sdft_results) // 2
    freqs = np.linspace(20, 20000, half_len)

    valid_magnitudes = np.abs(sdft_results[:half_len], out=sdft_results[:half_len])
    line.set_data(freqs, valid_magnitudes)

    band_values = cython_psychoacoustic_mapping(
        freqs.astype(np.float64), valid_magnitudes.astype(np.float64)
    )
    color_norm = plt.Normalize(0, 50)

    for i, (band, magnitude) in enumerate(band_values.items()):
        if i < len(lights):
            color_value = cmap(color_norm(magnitude))
            lights[i].set_color(color_value)
            console.print(color_value, style="magenta")

    console.print(band_values, style=style)
    return [line] + lights

def update_stable_sdft(frame: int) -> List[Any]:
    """
    Update function for Matplotlib FuncAnimation for each frame in the animation.
    This uses the stable_sdft function and is experimental for this application.
    Special case for k = fq bin 

    Parameters:
        frame (int): The current frame number in the animation.

    Returns:
        List[Any]: A list containing the updated line and light objects for the plot.
    """
    global audio_data, cmap
    fs = 44100  # Sampling frequency
    N = 150
    k = 5
    audio_data_complex = audio_data.astype(np.complex128, copy=False)

    # calculate bin for k=1
    sdft_result_k1 = stable_sdft(audio_data_complex, N, k) # cython super experimental here, use normal CPython for now
    valid_magnitudes = np.abs(sdft_result_k1)

    freqs = np.linspace(0, fs // 2, len(valid_magnitudes))

    line.set_data(freqs, valid_magnitudes)

    band_values = cython_psychoacoustic_mapping(freqs, valid_magnitudes)
    color_norm = plt.Normalize(0, len(valid_magnitudes))

    for i, (band, magnitude) in enumerate(band_values.items()):
        if i < len(lights):
            color_value = cmap(color_norm(magnitude))
            lights[i].set_color(color_value)

    print(band_values)
    return [line] + lights

# Main Execution Code
if __name__ == "__main__":
    audio_data = np.zeros(1024)  # Preallocate audio_data array
    fig, ax1, ax2, line = initialize_plot()  # Initialize plot

    # Create circles (lights) in the second plot
    num_bands = 8
    lights = [
        plt.Circle(
            (0.5 * np.cos(np.pi / 4 * i), 0.5 * np.sin(np.pi / 4 * i)),
            0.1,
            color="black",
        )
        for i in range(num_bands)
    ]
    for light in lights:
        ax2.add_artist(light)

    # Create FuncAnimation object and **choose update function accordingly**  SDFT Network / SDFT)
    ani = FuncAnimation(fig, update_sdft, frames=range(100), init_func=init, blit=True)

    # Start audio stream
    with sd.InputStream(callback=callback):
        plt.show()

