import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import sounddevice as sd
from typing import Dict, List, Any
from rewrite import cython_stable_sdft, cython_sdft, cython_psychoacoustic_mapping


def sdft(signal: np.ndarray, n: int) -> np.ndarray:
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


def psychoacoustic_mapping(freqs: np.ndarray, mags: np.ndarray) -> Dict[str, float]:
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
        band_values[band] = np.sum(mags[(freqs >= f_low) & (freqs <= f_high)])

    return band_values


def make_audio_cmap(bands: List[int]) -> Any:
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


def callback(indata: np.ndarray, frames: int, time: float, status: Any) -> None:
    global audio_data
    audio_data = indata[:, 0]


def init() -> List:
    line.set_data([], [])
    return [line] + lights


def update_sdft(frame: int) -> List:
    global audio_data, cmap
    audio_data_complex = audio_data.astype(np.complex128, copy=False)
    sdft_results = sdft(audio_data_complex, 50)
    half_len = len(sdft_results) // 2
    freqs = np.linspace(20, 20000, half_len)
    valid_magnitudes = np.abs(sdft_results[:half_len])
    line.set_data(freqs, valid_magnitudes)
    band_values = psychoacoustic_mapping(freqs, valid_magnitudes)
    color_norm = plt.Normalize(0, 50)

    for i, (band, magnitude) in enumerate(band_values.items()):
        if i < len(lights):
            color_value = cmap(color_norm(magnitude))
            lights[i].set_color(color_value)

    print(band_values)
    return [line] + lights


def update_stable_sdft(frame: int) -> List[Any]:
    global audio_data, cmap
    fs = 44100
    N = 150
    k = 5
    audio_data_complex = audio_data.astype(np.complex128, copy=False)
    sdft_result_kx = cython_stable_sdft(audio_data_complex, N, k)
    valid_magnitudes = np.abs(sdft_result_kx)
    freqs = np.linspace(0, fs // 2, len(valid_magnitudes))
    line.set_data(freqs, valid_magnitudes)
    band_values = cython_psychoacoustic_mapping(freqs, valid_magnitudes)
    color_norm = plt.Normalize(0, np.max(valid_magnitudes))

    for i, (band, magnitude) in enumerate(band_values.items()):
        if i < len(lights):
            color_value = cmap(color_norm(magnitude))
            lights[i].set_color(color_value)

    print(band_values)
    return [line] + lights


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
