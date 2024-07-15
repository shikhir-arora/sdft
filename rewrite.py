import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import sounddevice as sd
from typing import Dict, List, Any
from rewrite import cython_stable_sdft, cython_sdft, cython_psychoacoustic_mapping, generate_mel_freqs

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

def initialize_plot() -> Any:
    """
    Initialize the plot for real-time audio visualization.

    Returns:
        Any: Matplotlib Figure and Axes objects.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    (line,) = ax1.plot([], [], lw=2, color="c")
    ax1.set_xlim(0, 2595 * np.log10(1 + 22050 / 700))  # Use Mel scale max value for fs/2
    ax1.set_ylim(0, 50)
    ax1.grid(True)
    ax1.set_title("Real-Time Frequency Magnitude (Mel Scale)")
    ax1.set_xlabel("Frequency (Mel)")
    ax1.set_ylabel("Magnitude")
    ax2.axis("off")
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    return fig, ax1, ax2, line

def callback(indata: np.ndarray, frames: int, time: float, status: Any) -> None:
    """
    Callback function for sounddevice to update the audio data.

    Parameters:
        indata (np.ndarray): Incoming audio data.
        frames (int): Number of audio frames.
        time (float): Time information.
        status (Any): Status information.
    """
    global audio_data
    audio_data = indata[:, 0]

def init() -> List:
    """
    Initialization function for Matplotlib FuncAnimation.

    Returns:
        List: List of Matplotlib Artist objects to be updated.
    """
    line.set_data([], [])
    return [line] + lights

def update_sdft(frame: int) -> List:
    """
    Update function for Matplotlib FuncAnimation to update the plot in real-time using regular SDFT.

    Parameters:
        frame (int): The current frame number in the animation.

    Returns:
        List: A list containing the updated line and light objects for the plot.
    """
    global audio_data, cmap
    audio_data_complex = audio_data.astype(np.complex128, copy=False)
    sdft_results = cython_sdft(audio_data_complex, 50)
    half_len = len(sdft_results) // 2
    freqs = generate_mel_freqs(half_len, 44100)
    valid_magnitudes = np.abs(sdft_results[:half_len])
    line.set_data(freqs, valid_magnitudes)
    band_values = cython_psychoacoustic_mapping(freqs, valid_magnitudes)
    color_norm = plt.Normalize(0, 50)

    for i, (band, magnitude) in enumerate(band_values.items()):
        if i < len(lights):
            color_value = cmap(color_norm(magnitude))
            lights[i].set_color(color_value)

    print(band_values)
    return [line] + lights

def update_stable_sdft(frame: int) -> List[Any]:
    """
    Update function for Matplotlib FuncAnimation to update the plot in real-time using stable SDFT.

    Parameters:
        frame (int): The current frame number in the animation.

    Returns:
        List: A list containing the updated line and light objects for the plot.
    """
    global audio_data, cmap
    fs = 44100
    N = 1024
    audio_data_complex = audio_data.astype(np.complex128, copy=False)
    
    all_magnitudes = np.zeros(N // 2)
    for k in range(N // 2):
        sdft_result_k = cython_stable_sdft(audio_data_complex, N, k)
        valid_magnitudes = np.abs(sdft_result_k)
        all_magnitudes[k] = valid_magnitudes[0]
    
    mel_freqs = generate_mel_freqs(N // 2, fs)
    line.set_data(mel_freqs, all_magnitudes)

    # Update psychoacoustic bands
    band_values = cython_psychoacoustic_mapping(mel_freqs, all_magnitudes)
    color_norm = plt.Normalize(0, 50)
    for i, (band, magnitude) in enumerate(band_values.items()):
        if i < len(lights):
            color_value = cmap(color_norm(magnitude))
            lights[i].set_color(color_value)

    return [line] + lights

# Main function
if __name__ == "__main__":
    audio_data = np.zeros(1024)
    fig, ax1, ax2, line = initialize_plot()

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

    ani = FuncAnimation(
        fig, update_stable_sdft, frames=range(100), init_func=init, blit=True
    )

    with sd.InputStream(callback=callback):
        plt.show()
